"""Contains `sharp predict` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import json
from sharp.utils import vis

from sharp.models import (
    PredictorParams,
    RGBGaussianPredictor,
    create_predictor,
)
from sharp.utils import io, gsplat
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import (
    Gaussians3D,
    SceneMetaData,
    save_ply,
    load_ply,
    unproject_gaussians,
)

from .render import render_gaussians

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    help="Path to an image or containing a list of images.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the predicted Gaussians and renderings.",
    required=True,
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default="sharp_2572gikvuh.pt",
    help="Path to the .pt checkpoint. If not provided, downloads the default model automatically.",
    required=False,
)
@click.option(
    "--render/--no-render",
    "with_rendering",
    is_flag=True,
    default=False,
    help="Whether to render trajectory for checkpoint.",
)
@click.option(
    "--device",
    type=str,
    default="default",
    help="Device to run on. ['cpu', 'mps', 'cuda']",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def predict_cli(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    with_rendering: bool,
    device: str,
    verbose: bool,
):
    """Predict Gaussians from input images."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    extensions = io.get_supported_image_extensions()

    image_paths = []
    if input_path.is_file():
        if input_path.suffix in extensions:
            image_paths = [input_path]
    else:
        for ext in extensions:
            image_paths.extend(list(input_path.glob(f"**/*{ext}")))

    if len(image_paths) == 0:
        LOGGER.info("No valid images found. Input was %s.", input_path)
        return

    LOGGER.info("Processing %d valid image files.", len(image_paths))

    if device == "default":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    LOGGER.info("Using device %s", device)

    # Load or download checkpoint
    if checkpoint_path is None:
        LOGGER.info("No checkpoint provided. Downloading default model from %s", DEFAULT_MODEL_URL)
        state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    else:
        LOGGER.info("Loading checkpoint from %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, weights_only=True)

    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)

    output_path.mkdir(exist_ok=True, parents=True)

    for image_path in image_paths:
        if "441902" not in str(image_path):
            continue
        output_ply_path = output_path / f"{image_path.stem}.ply"
        output_json_path = output_path / f"{image_path.stem}.json"
        output_depth_path = output_path / f"{image_path.stem}.npy"

        if output_ply_path.exists() and output_json_path.exists() and output_depth_path.exists():
            LOGGER.info("Skipping %s (already processed)", image_path)
            continue

        LOGGER.info("Processing %s", image_path)
        image, _, f_px = io.load_rgb(image_path)
        height, width = image.shape[:2]

        gaussians = None
        intrinsics = torch.tensor(
            [
                [f_px, 0, (width - 1) / 2.0, 0],
                [0, f_px, (height - 1) / 2.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=device,
            dtype=torch.float32,
        )

        if not output_ply_path.exists():
            gaussians = predict_image(gaussian_predictor, image, f_px, torch.device(device))

            LOGGER.info("Saving 3DGS to %s", output_path)
            save_ply(gaussians, f_px, (height, width), output_ply_path)
        else:
            LOGGER.info("PLY file %s already exists, skipping.", output_ply_path)

        if not output_json_path.exists():
            # Compute vertical FOV in degrees
            vertical_fov_radians = 2 * np.arctan((height / 2) / f_px)
            vertical_fov_degrees = np.degrees(vertical_fov_radians)

            # Load the saved ply and compute depth statistics
            ply_path = output_ply_path
            if ply_path.exists():
                if gaussians is None:
                  gaussians, metadata, _, _ = load_ply(ply_path)
                depths = gaussians.mean_vectors[..., 2].cpu().numpy().flatten()
                min_depth = float(np.min(depths))
                max_depth = float(np.max(depths))
                mean_depth = float(np.mean(depths))
                median_depth = float(np.median(depths))
                percentile_1_depth = float(np.percentile(depths, 1))
                percentile_99_depth = float(np.percentile(depths, 99))
            else:
                min_depth = max_depth = mean_depth = median_depth = percentile_1_depth = percentile_99_depth = None

            info = {
                "image": str(image_path.name),
                "width": width,
                "height": height,
                "vertical_fov_degrees": float(vertical_fov_degrees),
                "focal_length_px": float(f_px),
                "min_depth": min_depth,
                "max_depth": max_depth,
                "mean_depth": mean_depth,
                "median_depth": median_depth,
                "percentile_1_depth": percentile_1_depth,
                "percentile_99_depth": percentile_99_depth,
            }
            # Save info as JSON
            with open(output_json_path, "w") as f:
                json.dump(info, f, indent=2)
        else:
            LOGGER.info("Info file %s already exists, skipping.", output_json_path)

        if not output_depth_path.exists():
            if gaussians is None:
                gaussians, metadata, _, _ = load_ply(output_ply_path)
            else:
                metadata = SceneMetaData(intrinsics[0, 0].item(), (width, height), "linearRGB")

            renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)
            rendering_output = renderer(
                gaussians.to(device),
                extrinsics=torch.eye(4, device=device).unsqueeze(0),
                intrinsics=intrinsics.unsqueeze(0),
                image_width=width,
                image_height=height,
            )
            depth = rendering_output.depth[0]
            depth_np = depth.cpu().numpy()
            np.save(output_depth_path, depth_np)

            depth_min = np.nanmin(depth_np)
            depth_max = np.nanmax(depth_np)
            if depth_max > depth_min:
                depth_norm = (depth_np - depth_min) / (depth_max - depth_min)
            else:
                depth_norm = np.zeros_like(depth_np)
            depth_gray = (depth_norm * 255).astype(np.uint8)
            if depth_gray.ndim == 3:
                depth_gray = np.squeeze(depth_gray)

            # Save as grayscale PNG
            io.save_image(depth_gray, output_path / f"{image_path.stem}_gray.png")

            colored_depth_pt = vis.colorize_depth(
                depth,
                min(depth_np.max(), vis.METRIC_DEPTH_MAX_CLAMP_METER),  # type: ignore[call-overload]
            )
            colored_depth_np = colored_depth_pt.squeeze(0).permute(1, 2, 0).cpu().numpy()
            io.save_image(colored_depth_np, output_path / f"{image_path.stem}_color.png")


        else:
            LOGGER.info("Depth file %s already exists, skipping.", output_depth_path)


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
) -> Gaussians3D:
    """Predict Gaussians from an image."""
    internal_shape = (1536, 1536)

    LOGGER.info("Running preprocessing.")
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # Predict Gaussians in the NDC space.
    LOGGER.info("Running inference.")
    gaussians_ndc = predictor(image_resized_pt, disparity_factor)

    LOGGER.info("Running postprocessing.")
    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Convert Gaussians to metrics space.
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians
