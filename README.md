# Sharp Monocular View Synthesis in Less Than a Second

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://apple.github.io/ml-sharp/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.10685-b31b1b.svg)](https://arxiv.org/abs/2512.10685)

This software project accompanies the research paper: _Sharp Monocular View Synthesis in Less Than a Second_
by _Lars Mescheder, Wei Dong, Shiwei Li, Xuyang Bai, Marcel Santos, Peiyun Hu, Bruno Lecouat, Mingmin Zhen, Amaël Delaunoy,
Tian Fang, Yanghai Tsin, Stephan Richter and Vladlen Koltun_.

![](data/teaser.jpg)

We present SHARP, an approach to photorealistic view synthesis from a single image. Given a single photograph, SHARP regresses the parameters of a 3D Gaussian representation of the depicted scene. This is done in less than a second on a standard GPU via a single feedforward pass through a neural network. The 3D Gaussian representation produced by SHARP can then be rendered in real time, yielding high-resolution photorealistic images for nearby views. The representation is metric, with absolute scale, supporting metric camera movements. Experimental results demonstrate that SHARP delivers robust zero-shot generalization across datasets. It sets a new state of the art on multiple datasets, reducing LPIPS by 25–34% and DISTS by 21–43% versus the best prior model, while lowering the synthesis time by three orders of magnitude.

## Quest 3 viewer and gallery (this fork)

This fork adds a web gallery and a WebXR viewer (Meta Quest 3, AR/VR), and a
hybrid mesh + Gaussian representation that renders much faster on the headset.

### Run everything

```bash
cd /projects/ml-sharp
scripts/start_server.sh
```

This starts three processes and stops all of them on Ctrl+C:

| Process | What it does | Log |
|---|---|---|
| `node web.js` | Gallery and viewer: HTTP on port 3000, HTTPS on port 3443 | `logs/web.log` |
| `sharp predict-with-info ... --loop` (GPU 0) | Turns photos in `images/` into scenes in `output2/` | `logs/predict.log` |
| `sharp bake-hybrid -i output2 --watch` (GPU 1) | Bakes each new scene into `<scene>_hybrid.glb` + `.ply` | `logs/bake.log` |

Run it inside `tmux` or `screen` so it keeps running after you log out:

```bash
tmux new -s sharp 'scripts/start_server.sh'   # detach with Ctrl+B then D; reattach: tmux attach -t sharp
```

Options (environment variables): `SHARP_ENV` (conda env, default
`/conda_envs/sharp`), `PREDICT_GPU` (default 0), `BAKE_GPU` (default 1), e.g.
`PREDICT_GPU=2 BAKE_GPU=3 scripts/start_server.sh`.

Requirements: Node.js with this folder's `npm install` done, the conda env with
this project installed, two CUDA GPUs (or set both variables to the same GPU),
and an HTTPS certificate in `certs/key.pem` + `certs/cert.pem` (WebXR needs
HTTPS; without them only HTTP on port 3000 runs).

### Conda environment

Use **`/conda_envs/sharp`**. It has this project installed in editable mode
(so it always runs the current code), plus PyTorch, gsplat, trimesh and
pyvista. On this machine gsplat needs the CUDA 13 runtime; the baking code
loads it by itself (`load_gsplat()` in `src/sharp/cli/bake_layers.py`).

- `scripts/start_server.sh` calls the env's programs directly
  (`/conda_envs/sharp/bin/sharp`), so you don't need to activate anything. Use
  another env with `SHARP_ENV=/path/to/env scripts/start_server.sh`.
- To run `sharp` commands yourself (baking, prediction), activate it first, or
  call the binary directly:

  ```bash
  conda activate /conda_envs/sharp
  sharp bake-hybrid -i output2
  # or, without activating:
  /conda_envs/sharp/bin/sharp bake-hybrid -i output2
  ```

The web server (`node web.js`) doesn't use conda: it runs on the system
Node.js with the packages from `npm install`.

### Use it

1. Open `https://<server-ip>:3443/` (it redirects to the gallery). On the Quest,
   accept the self-signed certificate warning once.
2. Drag photos onto the gallery (or use **Select Files**). Each appears in the
   gallery once predicted (a few seconds; refresh the page) and switches to
   its hybrid once baked (10–20 s later).
3. At the top of the gallery choose **Original splats** or **Hybrid** and the XR
   options, then open a scene and tap **START AR** (or **ENTER VR**).

Wait until `logs/predict.log` shows `Started watching images` before dropping
photos: files that arrive while the watcher is still starting (~10 s) are
missed. To process a missed photo, re-save it without changing it, e.g.
`python3 -c "open('images/<photo>.jpg','ab').close()"`.

Headset controls (right controller): thumbstick moves the scene, trigger
re-centres it, grip drags it, A / B go to the next / previous scene, **hold A
for 1 s to exit AR/VR**.

More: baking guide in [`docs/hybrid-baking.md`](docs/hybrid-baking.md);
design, measurements and lessons learned in
[`docs/quest-xr-findings.md`](docs/quest-xr-findings.md).

## Getting started

We recommend to first create a python environment:

```
conda create -n sharp python=3.13
```

Afterwards, you can install the project using

```
pip install -r requirements.txt
```

To test the installation, run

```
sharp --help
```

## Using the CLI

To run prediction:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians
```

The model checkpoint will be downloaded automatically on first run and cached locally at `~/.cache/torch/hub/checkpoints/`.

Alternatively, you can download the model directly:

```
wget https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt
```

To use a manually downloaded checkpoint, specify it with the `-c` flag:

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians -c sharp_2572gikvuh.pt
```

The results will be 3D gaussian splats (3DGS) in the output folder. The 3DGS `.ply` files are compatible to various public 3DGS renderers. We follow the OpenCV coordinate convention (x right, y down, z forward). The 3DGS scene center is roughly at (0, 0, +z). When dealing with 3rdparty renderers, please scale and rotate to re-center the scene accordingly.

### Rendering trajectories (CUDA GPU only)

Additionally you can render videos with a camera trajectory. While the gaussians prediction works for all CPU, CUDA, and MPS, rendering videos via the `--render` option currently requires a CUDA GPU. The gsplat renderer takes a while to initialize at the first launch.

```
sharp predict -i /path/to/input/images -o /path/to/output/gaussians --render

# Or from the intermediate gaussians:
sharp render -i /path/to/output/gaussians -o /path/to/output/renderings
```

## Evaluation

Please refer to the paper for both quantitative and qualitative evaluations.
Additionally, please check out this [qualitative examples page](https://apple.github.io/ml-sharp/) containing several video comparisons against related work.

## Citation

If you find our work useful, please cite the following paper:

```bibtex
@inproceedings{Sharp2025:arxiv,
  title      = {Sharp Monocular View Synthesis in Less Than a Second},
  author     = {Lars Mescheder and Wei Dong and Shiwei Li and Xuyang Bai and Marcel Santos and Peiyun Hu and Bruno Lecouat and Mingmin Zhen and Ama\"{e}l Delaunoy and Tian Fang and Yanghai Tsin and Stephan R. Richter and Vladlen Koltun},
  journal    = {arXiv preprint arXiv:2512.10685},
  year       = {2025},
  url        = {https://arxiv.org/abs/2512.10685},
}
```

## Acknowledgements

Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details.

## License

Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.
