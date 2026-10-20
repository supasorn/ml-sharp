"""Fill of an MPI GLB: summed screen area of all drawn triangles at the photo's viewpoint / frame area; texels."""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glb_read import read_glb
meta = json.load(open(sys.argv[1]))
f, W, H = meta["focal_length_px"], meta["width"], meta["height"]
for p in sys.argv[2:]:
    sc = read_glb(p); area = 0; nq = 0; tex = {}
    for n, g in sc.items():
        if n.startswith("ldicolor_"):
            tex[id(g["tex"])] = g["tex"].shape[0] * g["tex"].shape[1]; continue
        V = g["V"].astype(np.float64); P = f * V[:, :2] / V[:, 2:3]
        T = P[g["F"]]
        a_, b_ = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]; area += np.abs(a_[:, 0] * b_[:, 1] - a_[:, 1] * b_[:, 0]).sum() / 2
        nq += len(g["F"]) // 2; tex[id(g["tex"])] = g["tex"].shape[0] * g["tex"].shape[1]
    print(f"{os.path.basename(p):40s} fill {area / (W * H):5.2f}x  quads {nq / 1e3:6.0f}k  pages {sum(tex.values()) / 1e6:5.0f}M texels  "
          f"ktx2 {os.path.getsize(p.replace('.glb', '_ktx2.glb')) / 2**20:4.0f} MB")
