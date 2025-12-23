"""Contains `sharp render-mask` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import torch
import numpy as np

from sharp.utils import camera, gsplat, io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import Gaussians3D, SceneMetaData, load_ply

LOGGER = logging.getLogger(__name__)


def compute_gaussian_input_visibility(
    gaussians: Gaussians3D,
    input_intrinsics: torch.Tensor,
    input_extrinsics: torch.Tensor,
    image_width: int,
    image_height: int,
    renderer: gsplat.GSplatRenderer,
    device: torch.device,
    depth_epsilon: float = 0.01,
) -> torch.BoolTensor:
    """Computes a boolean mask indicating which Gaussians are visible in the input view."""
    input_intrinsics = input_intrinsics.to(device)
    input_extrinsics = input_extrinsics.to(device)
    with torch.no_grad():
        input_rendering_output = renderer(
            gaussians.to(device),
            extrinsics=input_extrinsics[None],
            intrinsics=input_intrinsics[None],
            image_width=image_width,
            image_height=image_height,
        )
    input_depth_map = input_rendering_output.depth[0].squeeze().cpu().numpy() # (H, W)
    mean_vectors_world = gaussians.mean_vectors.squeeze(0) # (N, 3)
    mean_vectors_world_hom = torch.cat(
        [mean_vectors_world, torch.ones_like(mean_vectors_world[..., :1])], dim=-1
    ) # (N, 4)
    mean_vectors_camera_hom = (input_extrinsics @ mean_vectors_world_hom.to(device).T).T # (N, 4)
    mean_vectors_camera = mean_vectors_camera_hom[..., :3] # (N, 3)

    projected_homogeneous = (input_intrinsics @ mean_vectors_camera_hom.T).T
    projected_depths = projected_homogeneous[..., 2] # (N,)
    valid_depth_mask = projected_depths > 1e-6
    projected_x = projected_homogeneous[..., 0] / projected_homogeneous[..., 2]
    projected_y = projected_homogeneous[..., 1] / projected_homogeneous[..., 2]
    projected_x_int = torch.round(projected_x).long()
    projected_y_int = torch.round(projected_y).long()
    num_gaussians = len(mean_vectors_world)
    visible_in_input_view = torch.full((num_gaussians,), False, dtype=torch.bool, device=device)
    clamped_x = torch.clamp(projected_x_int, 0, image_width - 1)
    clamped_y = torch.clamp(projected_y_int, 0, image_height - 1)
    input_depths_at_pixels = torch.from_numpy(
        input_depth_map[clamped_y.cpu().numpy(), clamped_x.cpu().numpy()]
    ).to(device)

    is_projected_on_image = (
        (projected_x_int >= 0) & (projected_x_int < image_width) &
        (projected_y_int >= 0) & (projected_y_int < image_height)
    )
    depth_match = torch.abs(projected_depths - input_depths_at_pixels) < depth_epsilon
    visible_in_input_view = is_projected_on_image & valid_depth_mask & depth_match
    return visible_in_input_view.cpu()


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the ply or a list of plys.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the rendered mask videos.",
    required=True,
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def render_mask_cli(input_path: Path, output_path: Path, verbose: bool):
    """Render mask video based on Gaussian visibility in the input view."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)
    if not torch.cuda.is_available():
        LOGGER.error("Rendering a checkpoint requires CUDA.")
        exit(1)
    output_path.mkdir(exist_ok=True, parents=True)
    params = camera.TrajectoryParams()
    if input_path.suffix == ".ply":
        scene_paths = [input_path]
    elif input_path.is_dir():
        scene_paths = list(input_path.glob("*.ply"))
    else:
        LOGGER.error("Input path must be either directory or single PLY file.")
        exit(1)
    device = torch.device("cuda")
    for scene_path in scene_paths:
        # if "175229" not in scene_path.stem: continue
        LOGGER.info("Rendering mask for %s", scene_path)
        gaussians_original, metadata, input_intrinsics, input_extrinsics = load_ply(scene_path)
        (width, height) = metadata.resolution_px
        renderer = gsplat.GSplatRenderer(color_space=metadata.color_space, background_color="black")
        visible_in_input_view = compute_gaussian_input_visibility(
            gaussians=gaussians_original,
            input_intrinsics=input_intrinsics,
            input_extrinsics=input_extrinsics,
            image_width=width,
            image_height=height,
            renderer=renderer,
            device=device,
            depth_epsilon=0.1,
        )
        masked_opacities = gaussians_original.opacities * visible_in_input_view.float().unsqueeze(0)
        white_colors = torch.ones_like(gaussians_original.colors) * 0.99
        masked_gaussians = Gaussians3D(
            mean_vectors=gaussians_original.mean_vectors,
            singular_values=gaussians_original.singular_values,
            quaternions=gaussians_original.quaternions,
            colors=white_colors,
            opacities=masked_opacities,
        ).to(device)
        intrinsics = torch.tensor(
            [
                [metadata.focal_length_px, 0, (width - 1) / 2.0, 0],
                [0, metadata.focal_length_px, (height - 1) / 2.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            device=device,
            dtype=torch.float32,
        )
        camera_model = camera.create_camera_model(
            gaussians_original, intrinsics, resolution_px=metadata.resolution_px
        )
        trajectory = camera.create_eye_trajectory(
            gaussians_original, params, resolution_px=metadata.resolution_px, f_px=metadata.focal_length_px
        )
        mask_output_path = (output_path / scene_path.stem).with_suffix(".mask.mp4")
        mask_video_writer = io.VideoWriter(mask_output_path, render_depth=False)
        for _, eye_position in enumerate(trajectory):
            camera_info = camera_model.compute(eye_position)
            rendering_output = renderer(
                masked_gaussians,
                extrinsics=camera_info.extrinsics[None].to(device),
                intrinsics=camera_info.intrinsics[None].to(device),
                image_width=camera_info.width,
                image_height=camera_info.height,
            )
            mask_alpha = (rendering_output.alpha[0].permute(1, 2, 0) * 255.0).to(dtype=torch.uint8)
            if mask_alpha.shape[-1] == 1:
                mask_rgb = mask_alpha.repeat(1, 1, 3)
            else:
                mask_rgb = mask_alpha
            mask_video_writer.add_frame(mask_rgb)

        mask_video_writer.close()
