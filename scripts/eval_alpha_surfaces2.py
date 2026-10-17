"""v2 of eval_alpha_surfaces: discontinuity-aware surfaces + multi-view texture optimisation.

Surfaces come from depth peeling as in v1 (`peel_layers`). Changes:
- Depth map: 3x3 median of the surface's disparity (where it has alpha).
- Depth edges: a triangle whose vertices span a depth jump is emitted twice,
  flat at its front and at its back depth, and each copy shows only the
  texels whose stored disparity is closer to its own (a viewer would do this
  with a texel-depth texture and a fragment discard). Background texels then
  move with the background instead of being dragged along with the
  foreground.
- OPT>0: the RGBA textures are optimised (Adam, logits) so that the viewer
  composite (opaque hybrid mesh, then surfaces back to front, each
  depth-tested against the mesh) matches renders of the original splats from
  NTRAIN random eyes within +-MOVE (x), +-MOVE/2 (y), 0..0.4 MOVE (z).
  Geometry stays fixed, so each view's samples are precomputed once.
Evaluation uses the fixed eyes of v1 (not in the training set).

Compositing is done in torch per pixel (samples sit exactly on pixel
centres); with OPT=0 it matches v1's gsplat-based composite.

Usage: python scripts/eval_alpha_surfaces2.py <scene>.ply [...]
  env: SUFFIX, K, STEP, PEEL_TOL, MOVE, OPT (iterations, default 0), NTRAIN, LR, SMOOTH, NOSPLIT, EXPORT, EXPORT_DIR, EVAL, SHARPLAYERS, NOMESH
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from plyfile import PlyData
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_alpha_layers as ev  # noqa: E402
import eval_alpha_surfaces as s1  # noqa: E402
from eval_alpha_layers import cast, dev, psnr, raster, splat_tensors  # noqa: E402

GW, SS = ev.GW, ev.SS
SHARPLAYERS = bool(os.environ.get("SHARPLAYERS"))   # surfaces = SHARP's own two layers (not depth peeling)
NOMESH = bool(os.environ.get("NOMESH"))             # all splats as surfaces, no hybrid mesh
K = s1.K = 2 if SHARPLAYERS else int(os.environ.get("K", 4))
STEP = int(os.environ.get("STEP", 2))
MOVE = float(os.environ.get("MOVE", 0.3))
OPT = int(os.environ.get("OPT", 0))
SMOOTH = float(os.environ.get("SMOOTH", 0))
EXPORT = os.environ.get("EXPORT", "")          # e.g. _surfaces: write <scene>_surfaces.glb for ?lod=surfaces
EXPORT_DIR = os.environ.get("EXPORT_DIR", "")  # default: next to the scene
EVAL = int(os.environ.get("EVAL", 1))     # sigma (render px) of depth smoothing
NTRAIN = int(os.environ.get("NTRAIN", 12))
LR = float(os.environ.get("LR", 0.05))
SUFFIX = os.environ.get("SUFFIX", "_hybrid")
MARGIN = s1.MARGIN
OUT = Path("renderings") / "alpha_surfaces2"


def build_surface(rgb, dacc, a, f, W, H, gap_disp):
    """Triangles (N,3,3) + per-triangle (d_this, d_other) disparities (d_other NaN = not on a jump), disparity map."""
    an = a.cpu().numpy()
    valid = an > 0.05
    if not valid.any():
        return None
    disp = an / np.maximum(dacc.cpu().numpy(), 1e-6)                # 1 / (dacc / a)
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    disp = disp[idx[0], idx[1]]                                     # fill from the nearest texel with alpha
    disp = np.where(valid, ndimage.median_filter(disp, 3), disp)
    if SMOOTH:                                  # alpha-weighted Gaussian smoothing of disparity
        wgt = ndimage.gaussian_filter(np.maximum(an, 1e-3), SMOOTH)
        disp = ndimage.gaussian_filter(disp * np.maximum(an, 1e-3), SMOOTH) / wgt
    Hc, Wc = an.shape
    ys, xs = np.arange(0, Hc, STEP), np.arange(0, Wc, STEP)
    if ys[-1] != Hc - 1: ys = np.r_[ys, Hc - 1]
    if xs[-1] != Wc - 1: xs = np.r_[xs, Wc - 1]
    amax = ndimage.maximum_filter(an, size=STEP + 1)[ys][:, xs]
    Dv = disp[ys][:, xs]
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    # Vertices sit on texel centres = eval pixel centres; nudge them so no pixel centre lies exactly on an
    # edge (with the strict inside test below, every pixel is then in exactly one triangle, as on a GPU).
    rays = np.stack([(gx + 0.5 + 0.0137 - MARGIN - W / 2) / f, (gy + 0.5 + 0.0071 - MARGIN - H / 2) / f,
                     np.ones_like(Dv)], -1)
    tris, dth, dot = [], [], []
    for (i0, j0), (i1, j1), (i2, j2) in (((0, 0), (0, 1), (1, 0)), ((0, 1), (1, 1), (1, 0))):
        sl = lambda i, j: (slice(i, len(ys) - 1 + i), slice(j, len(xs) - 1 + j))
        d3 = np.stack([Dv[sl(i0, j0)], Dv[sl(i1, j1)], Dv[sl(i2, j2)]], -1)
        r3 = np.stack([rays[sl(i0, j0)], rays[sl(i1, j1)], rays[sl(i2, j2)]], -2)
        am = np.maximum.reduce([amax[sl(i0, j0)], amax[sl(i1, j1)], amax[sl(i2, j2)]])
        r, c = np.nonzero(am > 1 / 255)
        d3, r3 = d3[r, c], r3[r, c]
        jump = d3.max(1) - d3.min(1) >= gap_disp
        sm = ~jump
        tris.append(r3[sm] / d3[sm, :, None]); dth.append(np.full(sm.sum(), np.nan)); dot.append(np.full(sm.sum(), np.nan))
        dfr, dbk = d3[jump].max(1), d3[jump].min(1)
        if os.environ.get("NOSPLIT"):           # v1 behaviour: flatten to the front depth, no gating
            tris.append(r3[jump] / dfr[:, None, None]); dth.append(np.full(jump.sum(), np.nan)); dot.append(np.full(jump.sum(), np.nan))
            continue
        # Each copy keeps the true depth of the vertices on its side (so it stays attached to that side's
        # neighbours) and moves the other side's vertices to its side's mean depth.
        dj, rj = d3[jump], r3[jump]
        fore = np.abs(dj - dfr[:, None]) < np.abs(dj - dbk[:, None])
        mf = (dj * fore).sum(1) / fore.sum(1)
        mb = (dj * ~fore).sum(1) / np.maximum((~fore).sum(1), 1)
        for side, mine, other in ((fore, mf, mb), (~fore, mb, mf)):
            dd = np.where(side, dj, mine[:, None])
            tris.append(rj / dd[..., None]); dth.append(mine); dot.append(other)
    T = torch.tensor(np.concatenate(tris), device=dev, dtype=torch.float32)
    dth = torch.tensor(np.concatenate(dth), device=dev, dtype=torch.float32)
    dot = torch.tensor(np.concatenate(dot), device=dev, dtype=torch.float32)
    return T, dth, dot, torch.tensor(disp, device=dev, dtype=torch.float32)


class Surface:
    def __init__(self, T, dth, dot, disp, tex, f, W, H):
        n = torch.linalg.solve(T, torch.ones(len(T), 3, device=dev))
        ok = torch.isfinite(n).all(1)
        self.T, self.n, self.dth, self.dot = T[ok], n[ok], dth[ok], dot[ok]
        self.tu = f * self.T[..., 0] / self.T[..., 2] + W / 2
        self.tv = f * self.T[..., 1] / self.T[..., 2] + H / 2
        self.Ht, self.Wt = disp.shape
        self.disp = disp.reshape(-1)
        self.tex = tex.reshape(-1, 4)          # premultiplied RGBA (initial)
        self.ntri = len(self.T)

    def inside(self, prim, ui, vi):
        x0, y0 = self.tu[prim], self.tv[prim]
        e1x, e1y = x0[:, 1] - x0[:, 0], y0[:, 1] - y0[:, 0]
        e2x, e2y = x0[:, 2] - x0[:, 0], y0[:, 2] - y0[:, 0]
        px, py = ui - x0[:, 0], vi - y0[:, 0]
        det = e1x * e2y - e1y * e2x
        l1 = (px * e2y - py * e2x) / det
        l2 = (e1x * py - e1y * px) / det
        return (l1 >= 0) & (l2 >= 0) & (l1 + l2 < 1) & (det.abs() > 1e-9)

    def samples(self, eye, W, H, f):
        """Per-sample (pix, t, texel idx (N,4), weights (N,4)); gated bilinear on jump copies."""
        out = {k: [] for k in ("pix", "t", "idx", "w")}
        Wt = self.Wt
        for prim, pix, p, ui, vi, t in cast(self.n, self.T, self.inside, eye, W, H, f):
            x = (ui - 0.5 + MARGIN).clamp(0, Wt - 1.001)
            y = (vi - 0.5 + MARGIN).clamp(0, self.Ht - 1.001)
            x0, y0 = x.floor().long(), y.floor().long()
            fx, fy = x - x0, y - y0
            b = y0 * Wt + x0
            idx = torch.stack([b, b + 1, b + Wt, b + Wt + 1], 1)
            w = torch.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1)
            dth, dot = self.dth[prim, None], self.dot[prim, None]
            dt = self.disp[idx]
            gate = torch.isnan(dot) | ((dt - dth).abs() < (dt - dot).abs())
            w = w * gate
            s = w.sum(1)
            keep = s > 1e-6
            out["pix"].append(pix[keep]); out["t"].append(t[keep])
            out["idx"].append(idx[keep].int()); out["w"].append((w[keep] / s[keep, None]).half())
        return {k: torch.cat(v) for k, v in out.items()} if out["pix"] else None


def mesh_layer(M, eye, W, H, f):
    """Nearest opaque mesh hit per pixel: colour (W*H,3), depth (W*H,) (inf = none)."""
    rgb = torch.zeros(W * H, 3, device=dev)
    depth = torch.full((W * H,), float("inf"), device=dev)
    Wt, tex = M.Wt, M.tex
    for prim, pix, p, ui, vi, t in cast(M.n, M.T, M.inside, eye, W, H, f):
        x, y = (ui - 0.5).clamp(0, Wt - 1.001), (vi - 0.5).clamp(0, M.Ht - 1.001)
        x0, y0 = x.floor().long(), y.floor().long()
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        i = y0 * Wt + x0
        c = (tex[i] * (1 - fx) + tex[i + 1] * fx) * (1 - fy) + (tex[i + Wt] * (1 - fx) + tex[i + Wt + 1] * fx) * fy
        k = c[:, 3] > 0.5
        pix, t, c = pix[k], t[k], c[k, :3] / c[k, 3:]
        cur = depth.clone()
        depth.scatter_reduce_(0, pix, t, "amin")
        win = (t <= depth[pix]) & (t < cur[pix])
        rgb[pix[win]] = c[win]
    return rgb, depth


def prep_view(surfs, M, eye, W, H, f):
    """Mesh colour/depth and, per surface, depth-tested samples sorted front to back per pixel."""
    mrgb, mdep = mesh_layer(M, eye, W, H, f)
    per = []
    for s in surfs:
        sm = s.samples(eye, W, H, f)
        if sm is None:
            per.append(None); continue
        k = sm["t"] < mdep[sm["pix"]]
        sm = {kk: v[k] for kk, v in sm.items()}
        o = torch.argsort(sm["pix"].double() * 1e4 + sm["t"].double())
        sm = {kk: v[o] for kk, v in sm.items()}
        first = torch.ones(len(o), dtype=torch.bool, device=dev)
        first[1:] = sm["pix"][1:] != sm["pix"][:-1]
        sm["seg"] = torch.cummax(torch.where(first, torch.arange(len(o), device=dev), 0), 0).values.int()
        sm["pix"] = sm["pix"].int()
        del sm["t"]
        per.append(sm)
    return mrgb, per


def composite(texs, mrgb, per, npix):
    img = mrgb
    for tex, sm in zip(reversed(texs), reversed(per)):
        if sm is None:
            continue
        v = (tex[sm["idx"].long()] * sm["w"].float()[..., None]).sum(1)
        a = v[:, 3].clamp(0, 0.999)
        la = torch.log1p(-a)
        cs = torch.cumsum(la.double(), 0)       # float32 loses the per-pixel differences over ~1e7 samples
        seg = sm["seg"].long()
        excl = (cs - la.double() - (cs[seg] - la[seg].double())).float()
        C = torch.zeros(npix, 3, device=dev).index_add_(0, sm["pix"], torch.exp(excl)[:, None] * v[:, :3])
        L = torch.zeros(npix, device=dev).index_add_(0, sm["pix"], la)
        img = C + torch.exp(L)[:, None] * img
    return img


def export_glb(surfs, f, W, H, path):
    """One textured mesh per surface ("surface_<k>", k = 0 front): camera-space vertices, premultiplied RGBA
    PNG texture (canvas incl. margin). Viewer: draw after the opaque hybrid mesh, back to front, blending
    ONE / ONE_MINUS_SRC_ALPHA, depth test on, depth write off."""
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals
    meshes = {}
    for k, srf in enumerate(surfs):
        V = srf.T.reshape(-1, 3).cpu().numpy()
        V, inv = np.unique(V, axis=0, return_inverse=True)      # share grid vertices
        faces = inv.reshape(-1, 3)
        uvx = (f * V[:, 0] / V[:, 2] + W / 2 + MARGIN) / srf.Wt
        uvy = (f * V[:, 1] / V[:, 2] + H / 2 + MARGIN) / srf.Ht
        tex = (srf.tex.reshape(srf.Ht, srf.Wt, 4).clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
        mat = PBRMaterial(name=f"surface_{k}", baseColorTexture=Image.fromarray(tex, "RGBA"), metallicFactor=0.0,
                          roughnessFactor=1.0, doubleSided=True, alphaMode="BLEND")
        # trimesh's glTF exporter flips v (it expects OpenGL-style uv), so pre-flip.
        vis = TextureVisuals(uv=np.stack([uvx, 1 - uvy], 1), material=mat)
        meshes[f"surface_{k}"] = trimesh.Trimesh(vertices=V.astype(np.float32), faces=faces, process=False, visual=vis)
    trimesh.Scene(meshes).export(path)
    print(f"  wrote {path} ({path.stat().st_size / 2**20:.0f} MB, "
          f"{sum(len(m.faces) for m in meshes.values())} tris)", flush=True)


def run(scene: Path):
    t0 = time.time()
    meta = json.loads(scene.with_suffix(".json").read_text())
    Hg = int(round(GW * meta["height"] / meta["width"]))
    fg = meta["focal_length_px"] * GW / meta["width"]
    W, H, f = GW * SS, Hg * SS, fg * SS
    fv = PlyData.read(scene)["vertex"].data
    full = splat_tensors(fv, dev)
    kv = fv if NOMESH else PlyData.read(scene.with_name(scene.stem + SUFFIX + ".ply"))["vertex"].data
    kept = splat_tensors(kv, dev)
    z = np.asarray(kv["z"], np.float64)
    u, w = fg * kv["x"] / np.maximum(z, 1e-6) + GW / 2, fg * kv["y"] / np.maximum(z, 1e-6) + Hg / 2
    gap = 1.0 / (fg * MOVE)
    if SHARPLAYERS:
        # SHARP's own two layers: the .ply holds layer 0 for every output pixel, then layer 1. For a hybrid's
        # subset, recover each splat's index by matching its position in the full file.
        N = len(fv) // 2
        if NOMESH:
            idx = np.arange(len(fv))
        else:
            key = lambda d: np.ascontiguousarray(np.stack([d["x"], d["y"], d["z"]], -1).astype(np.float32)).view("V12").ravel()
            lut = {k.tobytes(): i for i, k in enumerate(key(fv))}
            idx = np.array([lut[k.tobytes()] for k in key(kv)])
        layer = (idx >= N).astype(int)
    else:
        layer = s1.peel_layers(kept, z, u * SS, w * SS, W, H, f)
    zero = (0.0, 0.0, 0.0)

    if NOMESH:                                  # no hybrid mesh: one triangle behind the camera (never hit)
        Tm = torch.tensor([[[0.0, 0.0, -10.0], [1.0, 0.0, -10.0], [0.0, 1.0, -10.0]]], device=dev)
    else:
        mesh = trimesh.load(scene.with_name(scene.stem + SUFFIX + ".glb")).to_geometry()
        Tm = torch.tensor(mesh.vertices, device=dev, dtype=torch.float32)[torch.tensor(mesh.faces, device=dev)]
    rgbM, aM = raster([full], zero, W, H, f)
    M = s1.TriMesh(Tm, torch.cat([rgbM, aM[..., None]], -1), W, H, 0, f, W, H)

    Kc = ev.K_of(W, H, f); Kc[0, 0, 2] += MARGIN; Kc[0, 1, 2] += MARGIN
    lt = torch.tensor(layer, device=dev)
    surfs = []
    for k in range(K):
        m = lt == k
        if not m.any():
            continue
        sub = {kk: v[m] for kk, v in kept.items()}
        out, alpha, _ = ev.gs.rendering.rasterization(**sub, viewmats=torch.eye(4, device=dev)[None], Ks=Kc,
                                                      width=W + 2 * MARGIN, height=H + 2 * MARGIN, eps2d=ev.EPS,
                                                      render_mode="RGB+D")
        rgb, dacc, a = out[0, ..., :3], out[0, ..., 3], alpha[0, ..., 0]
        b = build_surface(rgb, dacc, a, f, W, H, 3 * gap)
        if b is not None:
            surfs.append(Surface(*b, torch.cat([rgb, a[..., None]], -1), f, W, H))
    ntri = [s.ntri for s in surfs]
    print(f"{scene.stem}{SUFFIX}: {len(kv)} splats -> {len(surfs)} surfaces "
          f"({', '.join(map(str, np.bincount(layer, minlength=K)))}), {sum(ntri)} tris ({', '.join(map(str, ntri))}), "
          f"STEP {STEP}, built in {time.time() - t0:.0f}s", flush=True)

    if EXPORT:
        export_glb(surfs, f, W, H, scene.with_name(scene.stem + EXPORT + ".glb") if not EXPORT_DIR
                   else Path(EXPORT_DIR) / (scene.stem + EXPORT + ".glb"))
        if not EVAL:
            return

    eyes = {"0": zero}
    for mv in (0.1, 0.2, 0.3):
        eyes |= {f"x+{mv}": (mv, 0, 0), f"x-{mv}": (-mv, 0, 0), f"y+{mv/2}": (0, mv / 2, 0),
                 f"z+{0.4*mv:.2f}": (0, 0, 0.4 * mv)}
    texs0 = [s.tex for s in surfs]

    def evaluate(texs, tag):
        res = {}
        for name, eye in eyes.items():
            R, _ = raster([full], eye, W, H, f)
            ms = M.samples(eye, W, H, f, opaque=True)
            Hy = R if NOMESH else raster([kept, ms], eye, W, H, f)[0]
            mrgb, per = prep_view(surfs, M, eye, W, H, f)
            with torch.no_grad():
                S = composite(texs, mrgb, per, W * H).reshape(H, W, 3)
            res[name] = {"H": psnr(R, Hy), tag: psnr(R, S)}
            if name in ("0", "x+0.2", "x+0.3"):
                OUT.mkdir(parents=True, exist_ok=True)
                row = torch.cat([R, Hy, S], 1).clamp(0, 1)
                Image.fromarray((row.cpu().numpy() * 255).astype(np.uint8)).resize((3 * GW, Hg)).save(
                    OUT / f"{scene.stem}{SUFFIX}{'_sharp2' if SHARPLAYERS else ''}{'_nomesh' if NOMESH else ''}_{tag}_{name}.jpg", quality=92)
            del per, ms
            torch.cuda.empty_cache()
        return res

    res = evaluate(texs0, "S")

    if OPT:
        rng = np.random.default_rng(1)
        train = [zero] + [(rng.uniform(-MOVE, MOVE), rng.uniform(-MOVE / 2, MOVE / 2), rng.uniform(0, 0.4 * MOVE))
                          for _ in range(NTRAIN - 1)]
        views = []
        for eye in train:
            R, _ = raster([full], eye, W, H, f)
            views.append((R.reshape(-1, 3).half(), *prep_view(surfs, M, eye, W, H, f)))
        torch.cuda.empty_cache()
        print(f"  prepared {len(views)} training views in {time.time() - t0:.0f}s; "
              f"GPU {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)
        eps = 1e-4
        params = []
        for tex in texs0:
            a = tex[:, 3:].clamp(eps, 1 - eps)
            c = (tex[:, :3] / tex[:, 3:].clamp(min=eps)).clamp(eps, 1 - eps)
            params.append(torch.cat([torch.logit(c), torch.logit(a)], 1).requires_grad_())
        opt = torch.optim.Adam(params, lr=LR)
        for it in range(OPT):
            R, mrgb, per = views[rng.integers(len(views))]
            texs = [torch.cat([torch.sigmoid(p[:, :3]) * torch.sigmoid(p[:, 3:]), torch.sigmoid(p[:, 3:])], 1)
                    for p in params]
            img = composite(texs, mrgb, per, W * H)
            loss = ((img - R.float()) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 100 == 0 or it == OPT - 1:
                print(f"  it {it:4d} loss {loss.item():.5f} (train PSNR {-10 * np.log10(loss.item()):.2f})", flush=True)
        del views
        torch.cuda.empty_cache()
        with torch.no_grad():
            texs = [torch.cat([torch.sigmoid(p[:, :3]) * torch.sigmoid(p[:, 3:]), torch.sigmoid(p[:, 3:])], 1)
                    for p in params]
        r2 = evaluate(texs, "S_opt")
        for k in res:
            res[k]["S_opt"] = r2[k]["S_opt"]
    for name, r in res.items():
        print(f"  {name:8s} " + "  ".join(f"{k} {v:5.2f}" for k, v in r.items()), flush=True)
    (OUT / f"{scene.stem}{SUFFIX}_K{K}_s{STEP}_opt{OPT}.json").write_text(
        json.dumps({"tris": ntri, "psnr": res}, indent=1))


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(Path(s))
