"""Offline quality of replacing hybrid splats with semi-transparent textured planes.

Usage: python scripts/eval_alpha_layers.py output2/<scene>.ply [...]  (SAFE=1: back-only layers)
Writes PSNR per eye to renderings/alpha_layers/<scene>.json and comparison strips
(R | H | O | C0.5 | CE0.5) at a few eyes. Results: docs/quest-xr-findings.md §5.

Everything is composited by gsplat: real splats as themselves, the hybrid
mesh and the layers ray-cast per eye pixel and turned into tiny per-pixel
Gaussians (eps2d ~0 so they don't leak into neighbours) at the hit depth.
gsplat sorts per tile by depth, so a pass gives the exact per-pixel depth
order; colour/black variants of passes give the viewer's draw orders:

  R   original splats (reference)
  H   current hybrid: mesh + kept splats
  O   layers, oracle per-pixel depth order with the splats
  A   viewer order mesh -> layers (sorted) -> splats
  B   viewer order mesh -> splats -> layers

Layers: per 16-px tile up to CAP planes (RANSAC in disparity) within
PX px parallax at MOVE m; texture = RGBA render of the layer's splats alone
from the input view (premultiplied), padded by PAD px.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sharp.cli.bake_layers import load_gsplat, splat_tensors  # noqa: E402

OUT = Path("renderings") / ("alpha_layers_safe" if os.environ.get("SAFE") else "alpha_layers")
GW, SS = 1024, 2           # fit grid width, supersampling of renders
TILE, PAD, CAP, MOVE, PX, MIN_LAYER = 16, 8, 4, 1.0, 0.5, 8
EPS = 0.01                 # eps2d for every pass (sample Gaussians must not leak)
dev = "cuda"
rng = np.random.default_rng(0)
gs = load_gsplat()


# ---------------------------------------------------------------- layer fit
def fit_layers(u, w, d, tol):
    """Returns per-splat layer id (-1 none), plane params (L,3) in grid px, tile (L,2)."""
    A = np.stack([u, w, np.ones_like(u)], 1)
    key = np.floor(u / TILE).astype(int) * 100000 + np.floor(w / TILE).astype(int)
    o = np.argsort(key)
    groups = np.split(o, np.flatnonzero(np.diff(key[o])) + 1)
    lid = -np.ones(len(u), int)
    planes, tiles = [], []
    for g in groups:
        free = g
        for _ in range(CAP):
            if len(free) < MIN_LAYER:
                break
            ds_o = free[np.argsort(d[free])]
            ds = d[ds_o]
            j = np.searchsorted(ds, ds + 2 * tol, "right")
            i = int(np.argmax(j - np.arange(len(ds))))
            best, bp = ds_o[i:j[i]], np.array([0, 0, ds[i] + tol])
            for _ in range(80):
                s = rng.choice(free, 3, replace=False)
                try:
                    p = np.linalg.solve(A[s], d[s])
                except np.linalg.LinAlgError:
                    continue
                inl = free[np.abs(A[free] @ p - d[free]) < tol]
                if len(inl) > len(best):
                    best, bp = inl, p
            if len(best) < MIN_LAYER:
                break
            lid[best] = len(planes)
            planes.append(bp)
            tiles.append((int(np.floor(u[g[0]] / TILE)), int(np.floor(w[g[0]] / TILE))))
            free = np.setdiff1d(free, best, assume_unique=True)
    return lid, np.array(planes), np.array(tiles)


# ---------------------------------------------------------------- rendering
def K_of(W, H, f):
    return torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=dev, dtype=torch.float32)[None]


def raster(parts, eye, W, H, f, margin=0):
    """parts: list of splat dicts; returns premultiplied rgb (H,W,3), alpha (H,W) (+margin px each side)."""
    cat = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    view = torch.eye(4, device=dev)
    view[:3, 3] = -torch.tensor(eye, device=dev, dtype=torch.float32)
    K = K_of(W, H, f)
    K[0, 0, 2] += margin; K[0, 1, 2] += margin
    out, alpha, _ = gs.rendering.rasterization(**cat, viewmats=view[None], Ks=K,
                                               width=W + 2 * margin, height=H + 2 * margin, eps2d=EPS)
    return out[0], alpha[0, ..., 0]


def recolor(p, c):
    q = dict(p)
    q["colors"] = torch.full_like(p["colors"], c)
    return q


def bilinear(tex, idx_base, x, y, S):
    """tex (N*S*S, C) flat; sample at local float coords (x,y) inside [0,S-1]."""
    x0, y0 = x.floor().long().clamp(0, S - 2), y.floor().long().clamp(0, S - 2)
    fx, fy = (x - x0).clamp(0, 1)[:, None], (y - y0).clamp(0, 1)[:, None]
    i = idx_base + y0 * S + x0
    return ((tex[i] * (1 - fx) + tex[i + 1] * fx) * (1 - fy)
            + (tex[i + S] * (1 - fx) + tex[i + S + 1] * fx) * fy)


def cast(n, corners, inside, eye, W, H, f, max_chunk=40_000_000):
    """Ray-cast planar primitives (n·p = 1) from eye. corners (P,C,3) bound each primitive.
    inside(prim, u_in, v_in) -> mask in input-view pixel coords. Yields (prim, pix_idx, point, u_in, v_in)."""
    e = torch.tensor(eye, device=dev, dtype=torch.float32)
    rel = corners - e
    z = rel[..., 2].clamp(min=1e-3)
    sx, sy = f * rel[..., 0] / z + W / 2, f * rel[..., 1] / z + H / 2
    ok = (rel[..., 2] > 0.02).all(1)
    x0 = sx.min(1).values.floor().clamp(0, W).long(); x1 = sx.max(1).values.ceil().clamp(0, W).long()
    y0 = sy.min(1).values.floor().clamp(0, H).long(); y1 = sy.max(1).values.ceil().clamp(0, H).long()
    bw, bh = (x1 - x0) * ok, (y1 - y0) * ok
    area = bw * bh
    cum = torch.cumsum(area, 0)
    start = 0
    while start < len(area):
        base = cum[start] - area[start]
        end = int(torch.searchsorted(cum, base + max_chunk, right=True).clamp(min=start + 1))
        sel = torch.arange(start, end, device=dev)
        a = area[sel]
        prim = torch.repeat_interleave(sel, a)
        if len(prim):
            off = torch.cumsum(a, 0) - a
            local = torch.arange(len(prim), device=dev) - torch.repeat_interleave(off, a)
            px = x0[prim] + local % bw[prim]
            py = y0[prim] + local // bw[prim]
            r = torch.stack([(px + 0.5 - W / 2) / f, (py + 0.5 - H / 2) / f, torch.ones_like(px, dtype=torch.float32)], 1)
            nn = n[prim]
            t = (1 - (nn * e).sum(1)) / (nn * r).sum(1)
            p = e + t[:, None] * r
            good = (t > 0.02) & (p[:, 2] > 0.02)
            u_in = f * p[:, 0] / p[:, 2].clamp(min=1e-3) + W / 2
            v_in = f * p[:, 1] / p[:, 2].clamp(min=1e-3) + H / 2
            good &= inside(prim, u_in, v_in)
            yield prim[good], (py * W + px)[good], p[good], u_in[good], v_in[good], t[good]
        start = end


def sample_splats(p, depth, rgb, alpha, f):
    """Tiny screen-sized Gaussians (sigma ~0.1 px) at points p seen at `depth` from the eye."""
    s = (0.1 * depth / f)[:, None].expand(-1, 3).contiguous()
    q = torch.zeros(len(p), 4, device=dev); q[:, 0] = 1
    return dict(means=p, quats=q, scales=s, opacities=alpha.clamp(0, 1), colors=rgb.clamp(0, 1))


def psnr(a, b):
    a = torch.nn.functional.avg_pool2d(a.permute(2, 0, 1)[None], SS)
    b = torch.nn.functional.avg_pool2d(b.permute(2, 0, 1)[None], SS)
    return float(-10 * torch.log10(((a - b) ** 2).mean()))


# ---------------------------------------------------------------- main
def run(scene: Path):
    meta = json.loads(scene.with_suffix(".json").read_text())
    Hg = int(round(GW * meta["height"] / meta["width"]))
    fg = meta["focal_length_px"] * GW / meta["width"]
    W, H, f = GW * SS, Hg * SS, fg * SS
    full = splat_tensors(PlyData.read(scene)["vertex"].data, dev)
    kv = PlyData.read(scene.with_name(scene.stem + "_hybrid.ply"))["vertex"].data
    kept = splat_tensors(kv, dev)
    z = np.asarray(kv["z"], np.float64)
    u, w = fg * kv["x"] / z + GW / 2, fg * kv["y"] / z + Hg / 2
    # Only tiles fully inside the frame (textures come from the input view).
    inf = np.nonzero((u >= 0) & (u < GW // TILE * TILE) & (w >= 0) & (w < Hg // TILE * TILE) & (z > 1e-3))[0]
    lid_in, planes, tiles = fit_layers(u[inf], w[inf], 1 / z[inf], PX / (fg * MOVE))
    lid = -np.ones(len(u), int); lid[inf] = lid_in
    if os.environ.get("SAFE"):
        # Keep only layers behind every remaining splat of their tile (then "layers, then splats" is the right order).
        tol = PX / (fg * MOVE)
        tk = np.floor(u / TILE).astype(int) * 100000 + np.floor(w / TILE).astype(int)
        keep = np.ones(len(planes), bool)
        tile_of = tiles[:, 0] * 100000 + tiles[:, 1]
        for t in np.unique(tile_of):
            ls = list(np.nonzero(tile_of == t)[0])
            members = np.nonzero(tk == t)[0]
            changed = True
            while changed and ls:
                changed = False
                rest = members[~np.isin(lid[members], ls)]
                for li in list(ls):
                    dl = planes[li, 0] * u[rest] + planes[li, 1] * w[rest] + planes[li, 2]
                    if (1 / z[rest] < dl - tol).any():
                        ls.remove(li); keep[li] = False; changed = True
        lid[(lid >= 0) & ~keep[np.maximum(lid, 0)]] = -1
        remap = -np.ones(len(planes), int); remap[keep] = np.arange(keep.sum())
        lid[lid >= 0] = remap[lid[lid >= 0]]
        planes, tiles = planes[keep], tiles[keep]
    L = len(planes)
    print(f"{scene.stem}: {len(kv)} kept splats, {L} layers take {100*(lid>=0).mean():.1f}%", flush=True)

    # Layer textures (premultiplied RGBA), parity groups so padded crops never see a neighbour's spill.
    S = (TILE + 2 * PAD) * SS
    tex = torch.zeros(L, S, S, 4, device=dev, dtype=torch.float16)
    k_in_tile = np.zeros(L, int)
    seen = {}
    for i, t in enumerate(map(tuple, tiles)):
        k_in_tile[i] = seen.get(t, 0); seen[t] = k_in_tile[i] + 1
    lid_t = torch.tensor(lid, device=dev)
    zero = (0.0, 0.0, 0.0)
    for k in range(CAP):
        for px_ in range(2):
            for py_ in range(2):
                ls = np.nonzero((k_in_tile == k) & (tiles[:, 0] % 2 == px_) & (tiles[:, 1] % 2 == py_))[0]
                if not len(ls):
                    continue
                m = torch.isin(lid_t, torch.tensor(ls, device=dev))
                rgb, a = raster([{kk: v[m] for kk, v in kept.items()}], zero, W, H, f, margin=PAD * SS)
                img = torch.cat([rgb, a[..., None]], -1)
                for li in ls:
                    tx, ty = tiles[li]
                    xs, ys = tx * TILE * SS, ty * TILE * SS  # padded origin in padded image
                    ya, xa = max(ys, 0), max(xs, 0)
                    yb, xb = min(ys + S, img.shape[0]), min(xs + S, img.shape[1])
                    if yb > ya and xb > xa:
                        tex[li, ya - ys:yb - ys, xa - xs:xb - xs] = img[ya:yb, xa:xb].half()
    tex_flat = tex.reshape(-1, 4).float()
    del tex

    # Layer planes in 3D: n·p = 1 (render grid coords).
    a_, b_, c_ = (torch.tensor(planes[:, i], device=dev, dtype=torch.float32) for i in range(3))
    nL = torch.stack([a_ * fg, b_ * fg, a_ * GW / 2 + b_ * Hg / 2 + c_], 1)
    org = torch.tensor(tiles * TILE - PAD, device=dev, dtype=torch.float32)  # grid px
    cu = torch.stack([org[:, 0], org[:, 0] + TILE + 2 * PAD, org[:, 0], org[:, 0] + TILE + 2 * PAD], 1)
    cv = torch.stack([org[:, 1], org[:, 1], org[:, 1] + TILE + 2 * PAD, org[:, 1] + TILE + 2 * PAD], 1)
    dc = (a_[:, None] * cu + b_[:, None] * cv + c_[:, None]).clamp(min=0.02)
    cornL = torch.stack([(cu - GW / 2) / fg, (cv - Hg / 2) / fg, torch.ones_like(cu)], -1) / dc[..., None]

    def in_layer(prim, ui, vi):
        lx = ui - org[prim, 0] * SS - 0.5
        ly = vi - org[prim, 1] * SS - 0.5
        return (lx >= 0) & (lx <= S - 1) & (ly >= 0) & (ly <= S - 1)

    # Mesh: triangles in camera space, texture = full splat render from the input view.
    mesh = trimesh.load(scene.with_name(scene.stem + "_hybrid.glb")).to_geometry()
    V = torch.tensor(mesh.vertices, device=dev, dtype=torch.float32)
    T = V[torch.tensor(mesh.faces, device=dev)]
    nM = torch.linalg.solve(T, torch.ones(len(T), 3, device=dev))
    okM = torch.isfinite(nM).all(1)
    T, nM = T[okM], nM[okM]
    tu, tv = f * T[..., 0] / T[..., 2] + W / 2, f * T[..., 1] / T[..., 2] + H / 2
    rgbM, aM = raster([full], zero, W, H, f)
    texM = torch.cat([rgbM, aM[..., None]], -1).reshape(-1, 4)

    def in_tri(prim, ui, vi):
        x0, y0 = tu[prim], tv[prim]
        e1x, e1y = tu[prim, 1] - x0[:, 0], tv[prim, 1] - y0[:, 0]
        e2x, e2y = tu[prim, 2] - x0[:, 0], tv[prim, 2] - y0[:, 0]
        px, py = ui - x0[:, 0], vi - y0[:, 0]
        det = e1x * e2y - e1y * e2x
        l1 = (px * e2y - py * e2x) / det
        l2 = (e1x * py - e1y * px) / det
        eps = 1e-4
        return (l1 >= -eps) & (l2 >= -eps) & (l1 + l2 <= 1 + eps) & (det.abs() > 1e-9)

    eyes = {"0": zero}
    for m in (0.15, 0.45, 1.0):
        eyes |= {f"x+{m}": (m, 0, 0), f"x-{m}": (-m, 0, 0), f"y+{m/2}": (0, m / 2, 0), f"z+{0.4*m:.2f}": (0, 0, 0.4 * m)}
    res = {}
    kept_rest = {k: v[lid_t < 0] for k, v in kept.items()}
    # E: per-tile split of the remaining splats: in front of every layer of their tile -> front group.
    rest = np.nonzero(lid < 0)[0]
    tkey = lambda tx, ty: tx * 100000 + ty
    rk = tkey(np.floor(u[rest] / TILE).astype(int), np.floor(w[rest] / TILE).astype(int))
    lk = tkey(tiles[:, 0], tiles[:, 1])
    front = np.ones(len(rest), bool)
    order = np.argsort(lk)
    for li in order:
        sel = rk == lk[li]
        if not sel.any():
            continue
        dl = planes[li, 0] * u[rest][sel] + planes[li, 1] * w[rest][sel] + planes[li, 2]
        front[np.nonzero(sel)[0][(1 / z[rest][sel]) <= dl + PX / (fg * MOVE)]] = False
    ft = torch.tensor(front, device=dev)
    kept_front = {k: v[ft] for k, v in kept_rest.items()}
    kept_back = {k: v[~ft] for k, v in kept_rest.items()}
    print(f"  E split: {front.sum()} front / {(~front).sum()} back of {len(rest)} remaining", flush=True)
    for name, eye in eyes.items():
        # mesh samples
        mp, mrgb, mt = [], [], []
        for prim, pix, p, ui, vi, t in cast(nM, T, in_tri, eye, W, H, f):
            x, y = (ui - 0.5).clamp(0, W - 1.001), (vi - 0.5).clamp(0, H - 1.001)
            x0, y0 = x.floor().long(), y.floor().long()
            fx, fy = (x - x0)[:, None], (y - y0)[:, None]
            i = y0 * W + x0
            c = (texM[i] * (1 - fx) + texM[(i + 1).clamp(max=len(texM) - 1)] * fx) * (1 - fy) + \
                (texM[(i + W).clamp(max=len(texM) - 1)] * (1 - fx) + texM[(i + W + 1).clamp(max=len(texM) - 1)] * fx) * fy
            mp.append(p); mrgb.append(c[:, :3] / c[:, 3:].clamp(min=1e-3)); mt.append(t)
        meshS = sample_splats(torch.cat(mp), torch.cat(mt), torch.cat(mrgb), torch.ones(sum(len(x) for x in mp), device=dev), f)
        # layer samples
        lp, lrgb, la, lt = [], [], [], []
        for prim, pix, p, ui, vi, t in cast(nL, cornL, in_layer, eye, W, H, f):
            lx = ui - org[prim, 0] * SS - 0.5
            ly = vi - org[prim, 1] * SS - 0.5
            c = bilinear(tex_flat, prim * S * S, lx, ly, S)
            keep = c[:, 3] > 1 / 255
            lp.append(p[keep]); la.append(c[keep, 3]); lt.append(t[keep])
            lrgb.append(c[keep, :3] / c[keep, 3:].clamp(min=1e-3))
        layS = sample_splats(torch.cat(lp), torch.cat(lt), torch.cat(lrgb), torch.cat(la), f)

        R, _ = raster([full], eye, W, H, f)
        Hy, _ = raster([kept, meshS], eye, W, H, f)
        O, _ = raster([kept_rest, meshS, layS], eye, W, H, f)
        # A: mesh -> layers -> splats on top (splats depth-tested vs mesh only)
        C1, _ = raster([kept_rest, recolor(meshS, 0)], eye, W, H, f)
        ak, _ = raster([recolor(kept_rest, 1), recolor(meshS, 0)], eye, W, H, f)
        LM, _ = raster([layS, meshS], eye, W, H, f)
        A = C1 + (1 - ak[..., :1]) * LM
        # B: mesh -> splats -> layers on top (layers depth-tested vs mesh only)
        CL, _ = raster([layS, recolor(meshS, 0)], eye, W, H, f)
        al, _ = raster([recolor(layS, 1), recolor(meshS, 0)], eye, W, H, f)
        SM, _ = raster([kept_rest, meshS], eye, W, H, f)
        B = CL + (1 - al[..., :1]) * SM
        # C<th>: like A, but layers write depth where alpha > th, so splats behind those parts are rejected.
        # CE<th>: mesh -> back splats -> layers (depth write alpha > th) -> front splats.
        SMb, _ = raster([kept_back, meshS], eye, W, H, f)
        X = CL + (1 - al[..., :1]) * SMb
        out = {}
        for th in (0.1, 0.3, 0.5, 0.8):
            op = layS["opacities"] > th
            occ = recolor({k: v[op] for k, v in layS.items()}, 0)
            occ["opacities"] = torch.ones_like(occ["opacities"])
            C1c, _ = raster([kept_rest, recolor(meshS, 0), occ], eye, W, H, f)
            akc, _ = raster([recolor(kept_rest, 1), recolor(meshS, 0), occ], eye, W, H, f)
            out[f"C{th}"] = C1c + (1 - akc[..., :1]) * LM
            C1f, _ = raster([kept_front, recolor(meshS, 0), occ], eye, W, H, f)
            akf, _ = raster([recolor(kept_front, 1), recolor(meshS, 0), occ], eye, W, H, f)
            out[f"CE{th}"] = C1f + (1 - akf[..., :1]) * X
        C, E = out["C0.5"], out["CE0.5"]
        res[name] = {k: psnr(R, X_) for k, X_ in (("H", Hy), ("O", O), ("A", A), ("B", B), *out.items())}
        res[name]["O_vs_H"] = psnr(Hy, O)
        print(f"  {name:8s} " + " ".join(f"{k} {v:5.2f}" for k, v in res[name].items()), flush=True)
        if name in ("0", "x+0.45", "x+1.0"):
            OUT.mkdir(exist_ok=True)
            row = torch.cat([R, Hy, O, C, E], 1).clamp(0, 1)
            Image.fromarray((row.cpu().numpy() * 255).astype(np.uint8)).resize((5 * GW, Hg)).save(
                OUT / f"{scene.stem}_{name}.jpg", quality=92)
        del meshS, layS
        torch.cuda.empty_cache()
    (OUT / f"{scene.stem}.json").write_text(json.dumps({"layers": L, "taken": float((lid >= 0).mean()),
                                                        "kept": len(kv), "psnr": res}, indent=1))


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(Path(s))
