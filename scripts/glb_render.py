"""Render an exported MPI-only GLB (PNG pages) from a moved eye, as the viewer draws it:
meshes surface_<k> with larger k first, triangles in index order, premultiplied "over" blending, no depth test,
bilinear texels, perspective-correct UVs; KTX2 pages (_ktx2.glb) are sampled trilinearly from their mip levels with the
viewer's LOD bias (NOMIPS=1: level 0 only; LODBIAS=-0.5). Plain, shared-texture (ADAPT_SHARE / LCT) and LCOL quads;
meshes named surface_<k>_u (bake UNPREM) hold un-premultiplied colour.

All fragments are generated at once (per-triangle bounding boxes, batched), then blended in draw order with one sort
and a segmented suffix sum of log(1 - alpha): out = sum_i c_i * prod_{j drawn after i} (1 - a_j).

python scripts/glb_render.py scene.json layers.glb out.png x,y,z[@tx,ty,tz] [scale]   (eye in metres, OpenCV axes; photo
       camera at 0; @: look at that point instead of along the photo's axis)
   or: python scripts/glb_render.py scene.json layers.glb OUTDIR x,y,z[;x,y,z...] [scale]   (one GLB, several eyes:
       OUTDIR/<i>.png; the GLB is read once)
Env: OLD_DIAG=1 blends a quad's diagonal pixels twice, as the old per-triangle renderer did (to check against it).
     ONLY=surface_<k> draws only that mesh.
"""
import json, os, sys, time
import numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glb_read import read_glb

dev = "cuda" if torch.cuda.is_available() else "cpu"
OLD_DIAG = bool(os.environ.get("OLD_DIAG"))
CHUNK = 60_000_000                                   # candidate fragments per batch
MIPS = not os.environ.get("NOMIPS")                  # KTX2 pages: trilinear mipmapping as in the viewer
LOD_BIAS = float(os.environ.get("LODBIAS", -0.5))     # the viewer's ?lodbias (layerLodBias)
PAR_ITERS = int(os.environ.get("PAR_ITERS", 1))       # PARALLAX bakes: offset lookups per fragment (viewer: 1)
PAR_CLAMP = float(os.environ.get("PAR_CLAMP", 2.0))   # PARALLAX: max shift in texels (the tiles' border; viewer: 2)


def sample(tex, uvpx):
    """Bilinear sample tex (h, w, C) at texel coords (N, 2) (texel centres at integers), zeros outside."""
    th, tw = tex.shape[:2]
    g = torch.stack([uvpx[:, 0] / (tw - 1) * 2 - 1, uvpx[:, 1] / (th - 1) * 2 - 1], -1)
    return torch.nn.functional.grid_sample(tex.permute(2, 0, 1)[None], g[None, None], align_corners=True)[0, :, 0].T


def sample_mip(mips, uvpx, lod):
    """Trilinear sample, as WebGL with LinearMipmapLinearFilter: uvpx in level-0 texel coords, lod (N,) after bias
    (<= 0: level 0, bilinear)."""
    if len(mips) == 1:
        return sample(mips[0], uvpx)
    lod = lod.clamp(0, len(mips) - 1)
    l0 = lod.floor().long(); fr = (lod - l0)[:, None]
    out = torch.zeros(len(uvpx), mips[0].shape[2], device=uvpx.device)
    for L in range(len(mips)):
        for sel, wgt in ((l0 == L, 1 - fr), (l0 + 1 == L, fr)):
            if sel.any():
                out[sel] += wgt[sel] * sample(mips[L], (uvpx[sel] + 0.5) / 2 ** L - 0.5)
    return out


def load(path):
    scene = read_glb(path)
    key = lambda n: int(n.split("_")[1].split(".")[0]) if n.split("_")[1].split(".")[0].isdigit() else 0
    texc = {}
    T = lambda a: texc.setdefault(id(a), torch.tensor(a, dtype=torch.float32, device=dev) / 255)
    cpage = None
    for n, g in scene.items():
        if n.startswith("ldicolor_"):
            cw_, ch_ = (int(v) for v in n.split("_")[1:3])
            cpage = (T(g["tex"]), cw_, ch_)
    meshes = []
    for n in sorted((n for n in scene if not n.startswith("ldicolor_") and (not os.environ.get("ONLY") or n == os.environ["ONLY"])),
                    key=lambda n: -key(n)):
        g = scene[n]
        tex = T(g["tex"]); th, tw = tex.shape[:2]
        uv = torch.tensor(g["uv"].astype(np.float32), device=dev) * torch.tensor([tw, th], device=dev) - 0.5
        hi = uvhi = None
        if g.get("hi") is not None:                   # HI_PEOPLE: quads with u >= 2 read the uncompressed people texture
            hi = T(g["hi"]); hh, hw = hi.shape[:2]
            raw = torch.tensor(g["uv"].astype(np.float32), device=dev)
            uvhi = torch.cat([(raw[:, :1] - 2) * hw, raw[:, 1:] * hh], 1) - 0.5
            uvhi = torch.cat([uvhi, (raw[:, :1] > 1.5).float()], 1)        # + the flag
        uv1 = (torch.tensor(g["uv1"].astype(np.float32), device=dev) * torch.tensor([tw, th], device=dev) - 0.5
               if g["uv1"] is not None else None)
        uv2 = torch.tensor(g["uv2"].astype(np.float32), device=dev) if g["uv2"] is not None else None
        uv3 = torch.tensor(g["uv3"].astype(np.float32), device=dev) if g.get("uv3") is not None else None
        mips = [tex] + [T(x) for x in g["mips"][1:]] if MIPS and g.get("mips") else [tex]
        meshes.append(dict(unprem=n.endswith("_u"), tex=tex, mips=mips, hi=hi, uvhi=uvhi,
                           V=torch.tensor(g["V"].astype(np.float32), device=dev),
                           F=torch.tensor(g["F"].astype(np.int64), device=dev), uv=uv, uv1=uv1, uv2=uv2, uv3=uv3))
    return meshes, cpage


def parse_eye(e):
    """'x,y,z' or 'x,y,z@tx,ty,tz' (look at the target) -> eye (3,), camera rotation R (3, 3) (OpenCV axes:
    rows = camera x right, y down, z forward in photo-camera coordinates)."""
    p, _, t = e.partition("@")
    eye = torch.tensor([float(v) for v in p.split(",")], device=dev)
    if not t:
        return eye, torch.eye(3, device=dev)
    z = torch.tensor([float(v) for v in t.split(",")], device=dev) - eye; z = z / z.norm()
    x = torch.linalg.cross(torch.tensor([0.0, 1.0, 0.0], device=dev), z); x = x / x.norm()
    return eye, torch.stack([x, torch.linalg.cross(z, x), z])


def render(meshes, cpage, meta, eye, sc=1.0, R=None):
    GW = 1024; Hg = round(GW * meta["height"] / meta["width"]); fg = meta["focal_length_px"] * GW / meta["width"]
    W, H, f = int(GW * sc), int(Hg * sc), fg * sc
    pix_l, ord_l, col_l = [], [], []
    order0 = 0
    for m in meshes:
        V = m["V"] - eye; F = m["F"]; nT = len(F)
        if R is not None:
            V = V @ R.T
        z = V[:, 2]
        P = torch.stack([f * V[:, 0] / z.clamp(min=1e-4) + W / 2, f * V[:, 1] / z.clamp(min=1e-4) + H / 2], 1)
        tz = z[F]                                                         # (T, 3)
        tp = P[F]                                                         # (T, 3, 2)
        x0 = torch.floor(tp[..., 0].amin(1)).clamp(0, W - 1).long(); x1 = torch.ceil(tp[..., 0].amax(1)).clamp(0, W - 1).long()
        y0 = torch.floor(tp[..., 1].amin(1)).clamp(0, H - 1).long(); y1 = torch.ceil(tp[..., 1].amax(1)).clamp(0, H - 1).long()
        p0, p1, p2 = tp[:, 0], tp[:, 1], tp[:, 2]
        den = (p1[:, 1] - p2[:, 1]) * (p0[:, 0] - p2[:, 0]) + (p2[:, 0] - p1[:, 0]) * (p0[:, 1] - p2[:, 1])
        ok = (tz > 1e-3).all(1) & (x1 >= x0) & (y1 >= y0) & (den.abs() >= 1e-9)
        bw = (x1 - x0 + 1); n = torch.where(ok, bw * (y1 - y0 + 1), 0)
        cs = torch.cumsum(n, 0)
        start = 0
        while start < nT:                                                 # chunks of triangles
            base = cs[start - 1] if start else torch.zeros((), dtype=cs.dtype, device=dev)
            end = int(torch.searchsorted(cs, base + CHUNK, right=True)); end = max(end, start + 1); end = min(end, nT)
            tri = torch.repeat_interleave(torch.arange(start, end, device=dev), n[start:end])
            if len(tri) == 0:
                start = end; continue
            loc = torch.arange(len(tri), device=dev) - (cs[tri] - n[tri] - base)
            xs = x0[tri] + loc % bw[tri]; ys = y0[tri] + loc // bw[tri]
            xc, yc = xs.float() + 0.5, ys.float() + 0.5
            a0, a1, a2 = p0[tri], p1[tri], p2[tri]; d = den[tri]
            a = ((a1[:, 1] - a2[:, 1]) * (xc - a2[:, 0]) + (a2[:, 0] - a1[:, 0]) * (yc - a2[:, 1])) / d
            b = ((a2[:, 1] - a0[:, 1]) * (xc - a2[:, 0]) + (a0[:, 0] - a2[:, 0]) * (yc - a2[:, 1])) / d
            c = 1 - a - b
            # second triangle of a quad (v0, v2, v3): its edge v0-v2 (c = 0) belongs to the first one
            cin = (c >= 0) if OLD_DIAG else torch.where(tri % 2 == 1, c > 0, c >= 0)
            sel = (a >= 0) & (b >= 0) & cin
            tri, xs, ys, a, b, c = tri[sel], xs[sel], ys[sel], a[sel], b[sel], c[sel]
            Ft = F[tri]
            wz = torch.stack([a, b, c], 1) / tz[tri]                       # perspective-correct weights
            wz = wz / wz.sum(1, keepdim=True)
            interp = lambda A: (wz[..., None] * A[Ft]).sum(1)
            uvp = interp(m["uv"])
            lod = None
            if len(m["mips"]) > 1:                                        # mip level from the uv footprint of a pixel
                d_ = d[sel]; q0, q1, q2 = a0[sel], a1[sel], a2[sel]
                def uv_at(da, db):
                    w_ = torch.stack([a + da, b + db, c - da - db], 1) / tz[tri]
                    return ((w_ / w_.sum(1, keepdim=True))[..., None] * m["uv"][Ft]).sum(1)
                ux = uv_at((q1[:, 1] - q2[:, 1]) / d_, (q2[:, 1] - q0[:, 1]) / d_) - uvp
                uy = uv_at((q2[:, 0] - q1[:, 0]) / d_, (q0[:, 0] - q2[:, 0]) / d_) - uvp
                lod = torch.log2(torch.maximum(ux.norm(dim=1), uy.norm(dim=1)).clamp(min=1e-6)) + LOD_BIAS
            smp = (lambda u: sample_mip(m["mips"], u, lod)) if lod is not None else (lambda u: sample(m["tex"], u))
            col = smp(uvp)
            if m["hi"] is not None:                                       # HI_PEOPLE: no mips, bilinear
                uh = interp(m["uvhi"])
                col = torch.where((uh[:, 2] > 0.5)[:, None], sample(m["hi"], uh[:, :2]), col)
            plain = torch.cat([col[:, :3] * col[:, 3:], col[:, 3:]], 1) if m["unprem"] else col   # UNPREM: colour / alpha
            if m["uv1"] is not None:
                ch = m["uv2"][Ft[:, 0], 0]
                pa = smp(interp(m["uv1"]))
                al = pa.gather(1, ch.round().long().clamp(0, 3)[:, None])
                if m["uv3"] is not None:                                  # LCW: whole alpha, colour mixed from 2 layers
                    pr = ch; iA = m["uv2"][Ft[:, 0], 1]; iB = m["uv3"][Ft[:, 0], 0]
                    stp = m["uv3"][Ft[:, 0], 1] * m["tex"].shape[1]                # tile stride, texels
                    a_ = torch.where(pr < 0.5, pa[:, 0], pa[:, 2])[:, None]; w_ = torch.where(pr < 0.5, pa[:, 1], pa[:, 3])[:, None]
                    cA = smp(uvp + torch.stack([iA * stp, torch.zeros_like(stp)], 1))[:, :3]
                    cB = smp(uvp + torch.stack([iB * stp, torch.zeros_like(stp)], 1))[:, :3]
                    col = torch.where((pr > -0.5)[:, None], torch.cat([(cA * (1 - w_) + cB * w_) * a_, a_], 1), plain)
                elif cpage is not None:                                   # LCOL: colour from the layer of this bin
                    cp, cw_, chh_ = cpage; CPH, CPW = cp.shape[:2]; th, tw = m["tex"].shape[:2]
                    bin_ = m["uv2"][Ft[:, 0], 1]
                    uvn = (uvp + 0.5) / torch.tensor([tw, th], device=dev)
                    ix = (uvn[:, 0] * CPW).long().clamp(0, CPW - 1); iy = (uvn[:, 1] * CPH).long().clamp(0, CPH - 1)
                    s1 = cp[iy, ix, 3] * 255; s2 = cp[iy, (ix + cw_).clamp(max=CPW - 1), 3] * 255
                    ox = torch.where(bin_ > s1 - 0.5, 0, torch.where(bin_ > s2 - 0.5, cw_, 0))
                    oy = torch.where(bin_ > s1 - 0.5, 0, torch.where(bin_ > s2 - 0.5, 0, chh_))
                    cc = sample(cp, torch.stack([uvn[:, 0] * CPW - 0.5 + ox, uvn[:, 1] * CPH - 0.5 + oy], 1))
                    col = torch.cat([cc[:, :3] * al, al], 1)
                else:                                                     # shared: colour x its alpha channel
                    dk = m["uv2"][Ft[:, 0], 1]                            # PARALLAX: plane thickness (0: none)
                    if bool((dk > 0).any()):
                        # shift the lookups so each texel shows at its own depth: offset (colour tile alpha, in plane
                        # thicknesses) x (eye_xy - x eye_z) / (1 - d eye_z), photo-camera coordinates
                        P = interp(m["V"]); xn = P[:, :2] / P[:, 2:3]; dn = 1 / P[:, 2]
                        V0, V1 = m["V"][Ft[:, 0]], m["V"][Ft[:, 1]]
                        sc_ = (m["uv"][Ft[:, 1]] - m["uv"][Ft[:, 0]]).norm(dim=1) / (
                            V1[:, :2] / V1[:, 2:3] - V0[:, :2] / V0[:, 2:3]).norm(dim=1).clamp(min=1e-9)   # texels per unit
                        dir_ = (eye[None, :2] - xn * eye[2]) / (1 - dn * eye[2])[:, None]
                        uv1p = interp(m["uv1"]); sh = torch.zeros_like(uvp)
                        for _ in range(PAR_ITERS):
                            off = (smp(uvp + sh)[:, 3] * 2 - 1) * dk
                            sh = torch.where((dk > 0)[:, None], off[:, None] * dir_ * sc_[:, None], 0)
                            sh = sh * (PAR_CLAMP / sh.norm(dim=1, keepdim=True).clamp(min=PAR_CLAMP))   # <= the tiles' border
                        col = smp(uvp + sh); pa = smp(uv1p + sh)
                        al = pa.gather(1, ch.round().long().clamp(0, 3)[:, None])
                    col = torch.where((ch > -0.5)[:, None], torch.cat([col[:, :3] * al, al], 1), plain)
            else:
                col = plain
            pix_l.append(ys * W + xs); ord_l.append(order0 + tri); col_l.append(col)
            start = end
        order0 += nT
    pix = torch.cat(pix_l); od = torch.cat(ord_l); col = torch.cat(col_l)
    del pix_l, ord_l, col_l
    o = torch.argsort(pix * order0 + od)                                  # by pixel, then draw order
    pix, col = pix[o], col[o]
    L = torch.log1p(-col[:, 3].double().clamp(0, 1 - 1e-7))
    R = torch.flip(torch.cumsum(torch.flip(L, [0]), 0), [0])             # R_i = sum_{j >= i} L_j
    last = torch.ones_like(pix, dtype=torch.bool); last[:-1] = pix[1:] != pix[:-1]
    # sum over the fragments after i within its pixel: R_i - L_i - (R at the start of the next pixel)
    nxt = torch.zeros_like(R); idx = torch.nonzero(last)[:, 0]
    seg_end_R = torch.where(idx + 1 < len(R), R[(idx + 1).clamp(max=len(R) - 1)], torch.zeros_like(R[idx]))
    seg_id = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long, device=dev), last[:-1].long()]), 0)
    S = R - L - seg_end_R[seg_id]
    out = torch.zeros(H * W, 4, device=dev, dtype=torch.float64)
    out.index_add_(0, pix, col.double() * torch.exp(S)[:, None])
    return out.view(H, W, 4).float()


if __name__ == "__main__":
    meta = json.load(open(sys.argv[1]))
    sc = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    t0 = time.time()
    meshes, cpage = load(sys.argv[2])
    eyes = sys.argv[4].split(";")
    if len(eyes) > 1:
        os.makedirs(sys.argv[3], exist_ok=True)
    for i, e in enumerate(eyes):
        eye, R = parse_eye(e)
        out = render(meshes, cpage, meta, eye, sc, R)
        dst = sys.argv[3] if len(eyes) == 1 else os.path.join(sys.argv[3], f"{i}.png")
        Image.fromarray((out[..., :3].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(dst)
        print(dst, out.shape[1], out.shape[0], f"{time.time() - t0:.1f}s", flush=True)
