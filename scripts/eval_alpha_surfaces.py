"""Offline quality of turning ALL remaining hybrid splats into alpha-textured surfaces.

The hybrid mesh stays. The splats it doesn't replace are split into K
surfaces by depth peeling from the input view (surface k = splats at or in
front of the expected depth of what's left after peeling 0..k-1; CLUSTER=1
uses per-TILE depth-gap clusters instead, which loses content hidden within
a cluster). Each surface is a
heightfield mesh in input-view image space (vertex every STEP render px):
depth and premultiplied RGBA texture are a render of that surface's splats
alone. Viewer order: mesh (opaque) -> surfaces back to front (blended, depth
tested against the mesh). No splats are left, so this order is exact up to
overlaps within one surface.

Usage: python scripts/eval_alpha_surfaces.py output2/<scene>.ply [...]
  env: SUFFIX (hybrid bake, default _hybrid), K, STEP, PEEL_TOL, MOVE, TILE, DIAG
  Results: docs/quest-xr-findings.md §5.
Prints PSNR vs the original splats per eye for H (hybrid) and S (surfaces),
writes renderings/alpha_surfaces/<scene>.json and comparison strips R | H | S.
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
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_alpha_layers as ev  # noqa: E402
from eval_alpha_layers import cast, dev, psnr, raster, recolor, sample_splats, splat_tensors  # noqa: E402

GW, SS = ev.GW, ev.SS
TILE = int(os.environ.get("TILE", 8))
K = int(os.environ.get("K", 4))
STEP = int(os.environ.get("STEP", 4))         # vertex spacing in render px
MOVE = float(os.environ.get("MOVE", 0.3))
SUFFIX = os.environ.get("SUFFIX", "_hybrid")
MARGIN = 384                                   # render px around the frame for surface textures
PEEL_TOL = float(os.environ.get("PEEL_TOL", 0.03))  # relative depth beyond the visible surface still peeled with it
OUT = Path("renderings") / "alpha_surfaces"
DIAG = bool(os.environ.get("DIAG"))            # also P / G / GP diagnostics (real splats, no textures)


def cluster_layers(u, w, d, gap):
    """Per-tile 1D clustering of disparity (split at gaps > gap); layer = cluster ordinal front to back."""
    key = np.floor(u / TILE).astype(np.int64) * 100000 + np.floor(w / TILE).astype(np.int64)
    o = np.lexsort((-d, key))                  # by tile, then front (large disparity) first
    ks, ds = key[o], d[o]
    new_tile = np.r_[True, ks[1:] != ks[:-1]]
    new_cl = new_tile | np.r_[True, (ds[:-1] - ds[1:]) > gap]
    cl = np.cumsum(new_cl)
    tile_start_cl = np.maximum.accumulate(np.where(new_tile, cl, 0))
    layer = np.empty(len(d), int)
    layer[o] = np.minimum(cl - tile_start_cl, K - 1)
    return layer


def peel_layers(splats, z, us, ws, W, H, f):
    """Depth peeling from the input view: surface k = splats at or in front of the expected depth of
    what's left after peeling surfaces 0..k-1 (at the splat's centre pixel); the last takes the rest."""
    layer = np.full(len(z), K - 1)
    left = np.ones(len(z), bool)
    ci = np.clip(np.round(us - 0.5).astype(int) + MARGIN, 0, W + 2 * MARGIN - 1)
    ri = np.clip(np.round(ws - 0.5).astype(int) + MARGIN, 0, H + 2 * MARGIN - 1)
    Kc = ev.K_of(W, H, f); Kc[0, 0, 2] += MARGIN; Kc[0, 1, 2] += MARGIN
    for k in range(K - 1):
        m = torch.tensor(left, device=dev)
        sub = {kk: v[m] for kk, v in splats.items()}
        out, alpha, _ = ev.gs.rendering.rasterization(**sub, viewmats=torch.eye(4, device=dev)[None], Ks=Kc,
                                                      width=W + 2 * MARGIN, height=H + 2 * MARGIN, eps2d=ev.EPS,
                                                      render_mode="ED")
        ed = out[0, ..., 0].cpu().numpy()
        a = alpha[0, ..., 0].cpu().numpy()
        idx = np.nonzero(left)[0]
        e = ed[ri[idx], ci[idx]]
        take = (z[idx] <= e * (1 + PEEL_TOL)) | (a[ri[idx], ci[idx]] < 0.05)
        layer[idx[take]] = k
        left[idx[take]] = False
    return layer


class TriMesh:
    """Camera-space triangles textured by an input-view image (premultiplied RGBA, with margin)."""

    def __init__(self, T, tex, Wt, Ht, margin, f, W, H):
        n = torch.linalg.solve(T, torch.ones(len(T), 3, device=dev))
        ok = torch.isfinite(n).all(1)
        self.T, self.n = T[ok], n[ok]
        self.tu = f * self.T[..., 0] / self.T[..., 2] + W / 2
        self.tv = f * self.T[..., 1] / self.T[..., 2] + H / 2
        self.tex, self.Wt, self.Ht, self.m = tex.reshape(-1, 4), Wt, Ht, margin

    def inside(self, prim, ui, vi):
        x0, y0 = self.tu[prim], self.tv[prim]
        e1x, e1y = x0[:, 1] - x0[:, 0], y0[:, 1] - y0[:, 0]
        e2x, e2y = x0[:, 2] - x0[:, 0], y0[:, 2] - y0[:, 0]
        px, py = ui - x0[:, 0], vi - y0[:, 0]
        det = e1x * e2y - e1y * e2x
        l1 = (px * e2y - py * e2x) / det
        l2 = (e1x * py - e1y * px) / det
        return (l1 >= -1e-4) & (l2 >= -1e-4) & (l1 + l2 <= 1 + 1e-4) & (det.abs() > 1e-9)

    def samples(self, eye, W, H, f, opaque=False):
        P, C, A, Tt = [], [], [], []
        Wt, tex = self.Wt, self.tex
        for prim, pix, p, ui, vi, t in cast(self.n, self.T, self.inside, eye, W, H, f):
            x = (ui - 0.5 + self.m).clamp(0, Wt - 1.001)
            y = (vi - 0.5 + self.m).clamp(0, self.Ht - 1.001)
            x0, y0 = x.floor().long(), y.floor().long()
            fx, fy = (x - x0)[:, None], (y - y0)[:, None]
            i = y0 * Wt + x0
            c = (tex[i] * (1 - fx) + tex[i + 1] * fx) * (1 - fy) + (tex[i + Wt] * (1 - fx) + tex[i + Wt + 1] * fx) * fy
            keep = c[:, 3] > (0.5 if opaque else 1 / 255)
            P.append(p[keep]); Tt.append(t[keep]); A.append(c[keep, 3])
            C.append(c[keep, :3] / c[keep, 3:].clamp(min=1e-3))
        if not P:
            return None
        a = torch.ones(sum(len(x) for x in P), device=dev) if opaque else torch.cat(A)
        return sample_splats(torch.cat(P), torch.cat(Tt), torch.cat(C), a, f)


def surface_mesh(rgb, depth, alpha, f, W, H, gap_disp):
    """Heightfield triangles over the padded canvas where the surface has alpha; cut at depth jumps."""
    Hc, Wc = alpha.shape
    a = alpha.cpu().numpy()
    dep = depth.cpu().numpy() / np.maximum(a, 1e-6)          # expected depth of this surface's splats
    valid = a > 0.05
    if not valid.any():
        return None, 0
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    dep = dep[idx[0], idx[1]]                                 # fill depth outside the surface from nearest
    ys, xs = np.arange(0, Hc, STEP), np.arange(0, Wc, STEP)
    if ys[-1] != Hc - 1: ys = np.r_[ys, Hc - 1]
    if xs[-1] != Wc - 1: xs = np.r_[xs, Wc - 1]
    amax = ndimage.maximum_filter(a, size=STEP + 1)[ys][:, xs]   # cell has any alpha near its corner
    D = dep[ys][:, xs]
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    X = (gx + 0.5 - MARGIN - W / 2) / f * D
    Y = (gy + 0.5 - MARGIN - H / 2) / f * D
    V = np.stack([X, Y, D], -1)
    disp = 1 / D
    tris = []
    for (i0, j0), (i1, j1), (i2, j2) in (((0, 0), (0, 1), (1, 0)), ((0, 1), (1, 1), (1, 0))):
        sl = lambda i, j: (slice(i, len(ys) - 1 + i), slice(j, len(xs) - 1 + j))
        d3 = np.stack([disp[sl(i0, j0)], disp[sl(i1, j1)], disp[sl(i2, j2)]])
        am = np.stack([amax[sl(i0, j0)], amax[sl(i1, j1)], amax[sl(i2, j2)]]).max(0)
        has = am > 1 / 255
        r, c = np.nonzero(has)
        tri = np.stack([V[r + i0, c + j0], V[r + i1, c + j1], V[r + i2, c + j2]], 1)
        # Across a depth jump, don't stretch a rubber sheet: flatten the triangle to its front depth.
        jump = (d3.max(0) - d3.min(0) >= gap_disp)[r, c]
        zf = tri[jump, :, 2].min(1)
        tri[jump] *= (zf[:, None] / tri[jump, :, 2])[..., None]
        tris.append(tri)
    T = np.concatenate(tris)
    return torch.tensor(T, device=dev, dtype=torch.float32), len(T)


def run(scene: Path):
    meta = json.loads(scene.with_suffix(".json").read_text())
    Hg = int(round(GW * meta["height"] / meta["width"]))
    fg = meta["focal_length_px"] * GW / meta["width"]
    W, H, f = GW * SS, Hg * SS, fg * SS
    full = splat_tensors(PlyData.read(scene)["vertex"].data, dev)
    kv = PlyData.read(scene.with_name(scene.stem + SUFFIX + ".ply"))["vertex"].data
    kept = splat_tensors(kv, dev)
    z = np.asarray(kv["z"], np.float64)
    u, w = fg * kv["x"] / np.maximum(z, 1e-6) + GW / 2, fg * kv["y"] / np.maximum(z, 1e-6) + Hg / 2
    gap = 1.0 / (fg * MOVE)                    # 1 grid px parallax at MOVE
    if os.environ.get("CLUSTER"):
        layer = cluster_layers(u, w, 1 / np.maximum(z, 1e-6), gap)
    else:
        layer = peel_layers(kept, z, u * SS, w * SS, W, H, f)
    zero = (0.0, 0.0, 0.0)

    # Hybrid mesh.
    mesh = trimesh.load(scene.with_name(scene.stem + SUFFIX + ".glb")).to_geometry()
    Tm = torch.tensor(mesh.vertices, device=dev, dtype=torch.float32)[torch.tensor(mesh.faces, device=dev)]
    rgbM, aM = raster([full], zero, W, H, f)
    M = TriMesh(Tm, torch.cat([rgbM, aM[..., None]], -1), W, H, 0, f, W, H)

    # Surfaces.
    lt = torch.tensor(layer, device=dev)
    surfs, ntri, cover, diag = [], [], [], []
    for k in range(K):
        m = lt == k
        if not m.any():
            continue
        sub = {kk: v[m] for kk, v in kept.items()}
        view = torch.eye(4, device=dev)
        Kc = ev.K_of(W, H, f); Kc[0, 0, 2] += MARGIN; Kc[0, 1, 2] += MARGIN
        out, alpha, _ = ev.gs.rendering.rasterization(**sub, viewmats=view[None], Ks=Kc, width=W + 2 * MARGIN,
                                                      height=H + 2 * MARGIN, eps2d=ev.EPS, render_mode="RGB+D")
        rgb, dacc, a = out[0, ..., :3], out[0, ..., 3], alpha[0, ..., 0]
        T, n = surface_mesh(rgb, dacc, a, f, W, H, gap_disp=3 * gap)
        if DIAG:
            # Diagnostic: the same splats, snapped along their rays onto this surface's depth map.
            an = a.cpu().numpy()
            dmap = dacc.cpu().numpy() / np.maximum(an, 1e-6)
            valid = an > 0.05
            ii = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
            dmap = dmap[ii[0], ii[1]]
            sel = np.nonzero(layer == k)[0]
            ci = np.clip(np.round(u[sel] * SS - 0.5).astype(int) + MARGIN, 0, W + 2 * MARGIN - 1)
            ri = np.clip(np.round(w[sel] * SS - 0.5).astype(int) + MARGIN, 0, H + 2 * MARGIN - 1)
            sc = torch.tensor((dmap[ri, ci] / z[sel]) ** 0.99, device=dev, dtype=torch.float32)  # keeps per-pixel order
            snap = dict(sub); snap["means"] = sub["means"] * sc[:, None]; snap["scales"] = sub["scales"] * sc[:, None]
            diag.append((sub, snap))
        if T is None:
            continue
        surfs.append(TriMesh(T, torch.cat([rgb, a[..., None]], -1), W + 2 * MARGIN, H + 2 * MARGIN, MARGIN, f, W, H))
        ntri.append(n)
        cover.append(float((a > 1 / 255).float().mean()) * (W + 2 * MARGIN) * (H + 2 * MARGIN) / (W * H))
    counts = np.bincount(layer, minlength=K)
    print(f"{scene.stem}{SUFFIX}: {len(kv)} splats -> {K} surfaces ({', '.join(map(str, counts))} splats), "
          f"{sum(ntri)} tris ({', '.join(map(str, ntri))}), mesh {len(M.T)} tris, "
          f"surface overdraw {sum(cover):.2f}x", flush=True)

    eyes = {"0": zero}
    for mv in (0.1, 0.2, 0.3):
        eyes |= {f"x+{mv}": (mv, 0, 0), f"x-{mv}": (-mv, 0, 0), f"y+{mv/2}": (0, mv / 2, 0),
                 f"z+{0.4*mv:.2f}": (0, 0, 0.4 * mv)}
    res = {}
    for name, eye in eyes.items():
        ms = M.samples(eye, W, H, f, opaque=True)
        R, _ = raster([full], eye, W, H, f)
        Hy, _ = raster([kept, ms], eye, W, H, f)
        img, _ = raster([ms], eye, W, H, f)
        allS = []
        for s in reversed(surfs):              # back to front, each depth-tested against the mesh
            ss = s.samples(eye, W, H, f)
            if ss is None:
                continue
            allS.append(ss)
            Ck, _ = raster([ss, recolor(ms, 0)], eye, W, H, f)
            ak, _ = raster([recolor(ss, 1), recolor(ms, 0)], eye, W, H, f)
            img = Ck + (1 - ak[..., :1]) * img
        O, _ = raster([ms] + allS, eye, W, H, f)   # exact per-pixel order, for reference
        res[name] = {"H": psnr(R, Hy), "S": psnr(R, img), "S_exact": psnr(R, O), "S_vs_H": psnr(Hy, img)}
        if DIAG:
            def in_order(groups):
                im, _ = raster([ms], eye, W, H, f)
                for g in reversed(groups):
                    Ck, _ = raster([g, recolor(ms, 0)], eye, W, H, f)
                    ak, _ = raster([recolor(g, 1), recolor(ms, 0)], eye, W, H, f)
                    im = Ck + (1 - ak[..., :1]) * im
                return im
            Gx, _ = raster([ms] + [sn for _, sn in diag], eye, W, H, f)
            res[name] |= {"P": psnr(R, in_order([o for o, _ in diag])), "G": psnr(R, Gx),
                          "GP": psnr(R, in_order([sn for _, sn in diag]))}
        print(f"  {name:8s} " + "  ".join(f"{k} {v:5.2f}" for k, v in res[name].items()), flush=True)
        if name in ("0", "x+0.2", "x+0.3"):
            OUT.mkdir(parents=True, exist_ok=True)
            row = torch.cat([R, Hy, img], 1).clamp(0, 1)
            Image.fromarray((row.cpu().numpy() * 255).astype(np.uint8)).resize((3 * GW, Hg)).save(
                OUT / f"{scene.stem}{SUFFIX}_{name}.jpg", quality=92)
        torch.cuda.empty_cache()
    (OUT / f"{scene.stem}{SUFFIX}.json").write_text(json.dumps(
        {"splats": len(kv), "per_surface": counts.tolist(), "tris": ntri, "overdraw": cover, "psnr": res}, indent=1))


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(Path(s))
