# Rule: tmux send -t 1 '!python' Enter Enter
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
    # RGBGaussianPredictor,
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

from sharp.cli.render import render_gaussians
import cv2 as cv

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
    device = torch.device("cuda")

    extensions = io.get_supported_image_extensions()
    image_paths = []
    if input_path.is_file():
        if input_path.suffix in extensions:
            image_paths = [input_path]
    else:
        for ext in extensions:
            image_paths.extend(list(input_path.glob(f"**/*{ext}")))

    for image_path in image_paths:
        if "441902" not in str(image_path):
            continue
        output_depth_path = output_path / f"{image_path.stem}.npy"
        depth = np.load(output_depth_path)
        print(depth.shape, depth.dtype, depth.min(), depth.max())

        # find the seed from the nearest point
        seed_point = np.unravel_index(np.nanargmin(depth[0]), depth[0].shape)
        seed_point = (seed_point[1], seed_point[0])  # (col, row) for OpenCV

        print("seed_point:", seed_point)

        flooded_depth, mask = flood_fill(depth[0], seed_point=seed_point, max_diff=0.01)
        # output_mask_path = output_path / f"{image_path.stem}_mask.png"
        # save_depth(flooded_depth, output_mask_path)
        #
        # output_mask_path = output_path / f"{image_path.stem}_depth0.png"
        # save_depth(depth[0], output_mask_path)

        # dilate mask by 5
        kernel = np.ones((3, 3), np.uint8)
        # mask = cv.dilate(mask, kernel, iterations=5)

        # erode mask by 10
        mask = cv.erode(mask, kernel, iterations=5)
        mask = mask[1:-1, 1:-1]  # remove the border

        cv.imwrite(output_path / f"{image_path.stem}_mask2.png", mask * 255)  # remove the border

        gaussians, metadata, _, _ = load_ply(output_path / f"{image_path.stem}.ply")
        width, height = metadata.resolution_px
        f_px = metadata.focal_length_px
        
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

        # identity exrtrinsics
        ext_id = torch.eye(4).unsqueeze(0)

        renderer = gsplat.GSplatRenderer(color_space=metadata.color_space)

        rendering_output = renderer(
            gaussians.to(device),
            extrinsics=ext_id.to(device),
            intrinsics=intrinsics[None],
            image_width=width,
            image_height=height,
        )
        color = (rendering_output.color[0].permute(1, 2, 0) * 255.0).to(dtype=torch.uint8)
        color = color.cpu().numpy()

        # save color image
        io.save_image(color, output_path / f"{image_path.stem}_rendered.png")


        # 1. Extract masked depth and color and compute vertices
        h, w = depth.shape[1], depth.shape[2]

        # apply gaussian blur to the depth
        depth_blurred = cv.GaussianBlur(depth[0], (5, 5), 0)
        depth_blurred = cv.GaussianBlur(depth_blurred, (5, 5), 0)
        depth_blurred = cv.GaussianBlur(depth_blurred, (5, 5), 0)

        effective_mask = mask
        rows, cols = np.where(effective_mask == 1)

        if len(rows) == 0:
            LOGGER.warning(
                f"No masked pixels found for {image_path.stem}. Skipping mesh creation."
            )
            continue

        # Extract depth and color values for the masked pixels
        # masked_depth_values = depth[0][rows, cols]
        masked_depth_values = depth_blurred[rows, cols]
        masked_color_values = color[rows, cols]  # color is HxWx3, so direct indexing works

        # Get intrinsics parameters from the torch tensor
        f_x = intrinsics[0, 0].item()
        f_y = intrinsics[1, 1].item()
        c_x = intrinsics[0, 2].item()
        c_y = intrinsics[1, 2].item()

        vertices = []
        vertex_colors = []
        # Map (row, col) from original image to its index in the `vertices` list
        vertex_map = {}

        vertex_idx_counter = 0
        for i in range(len(rows)):
            r, c = rows[i], cols[i]  # Current pixel (row, col)
            d = masked_depth_values[i]  # Depth at this pixel

            # Skip invalid or zero depth values
            if np.isnan(d) or d <= 0:
                continue

            # Unproject 2D pixel (c, r) with depth d to 3D point (x, y, z) in camera coordinates
            x = (c - c_x) * d / f_x
            y = (r - c_y) * d / f_y
            z = d  # Z-coordinate is the depth value

            vertices.append([x, y, z])
            vertex_colors.append(masked_color_values[i].tolist())
            vertex_map[(r, c)] = vertex_idx_counter  # Store the mapping
            vertex_idx_counter += 1

        if not vertices:
            LOGGER.warning(
                f"No valid 3D points could be generated for {image_path.stem}. Skipping mesh creation."
            )
            continue

        print(len(vertices), "vertices generated.")
        # Convert lists to numpy arrays for efficiency if needed later, or direct use for PLY writing
        vertices_np = np.array(vertices, dtype=np.float32)
        vertex_colors_np = np.array(vertex_colors, dtype=np.uint8)

        # 2. Triangulation: Create faces by connecting 2x2 blocks of masked pixels
        faces = []
        # Iterate over the effective mask, considering 2x2 blocks
        for r in range(h - 1):
            for c in range(w - 1): 
                # Check if all four corners of the 2x2 block are within the masked region
                if (
                    effective_mask[r, c]
                    and effective_mask[r + 1, c]
                    and effective_mask[r, c + 1]
                    and effective_mask[r + 1, c + 1]
                ):
                    # Get the vertex indices for the four corners from the vertex_map
                    idx_rc = vertex_map.get((r, c))
                    idx_r1c = vertex_map.get((r + 1, c))
                    idx_rc1 = vertex_map.get((r, c + 1))
                    idx_r1c1 = vertex_map.get((r + 1, c + 1))

                    # Ensure all four corner vertices were valid (i.e., had valid depth and were processed)
                    if all(idx is not None for idx in [idx_rc, idx_r1c, idx_rc1, idx_r1c1]):
                        # Create two triangles from the quadrilateral (r,c), (r+1,c), (r,c+1), (r+1,c+1)
                        # Triangle 1: Top-left, Bottom-left, Top-right
                        faces.append([idx_rc, idx_r1c, idx_rc1])
                        # Triangle 2: Bottom-left, Bottom-right, Top-right
                        faces.append([idx_r1c, idx_r1c1, idx_rc1])

        if not faces:
            LOGGER.warning(
                f"No faces could be generated for {image_path.stem}. Skipping mesh saving."
            )
            continue

        # 3. Save the mesh as a PLY file
        mesh_output_path = output_path / f"{image_path.stem}_mesh.ply"

        with open(mesh_output_path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices_np)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # Write vertex data
            for i in range(len(vertices_np)):
                v = vertices_np[i]
                c = vertex_colors_np[i]
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

            # Write face data
            for face in faces:
                # PLY format for faces: <num_vertices_in_face> <idx1> <idx2> ...
                f.write(f"3 {' '.join(map(str, face))}\n")

        LOGGER.info(f"Mesh with {len(vertices_np)} vertices and {len(faces)} faces saved to {mesh_output_path}")



        # constructing a new_gaussians by taking the gaussians and removeing any guassians that lie near the mesh.
        # i.e., if its depth value is without a threshold of the mesh depth value at that pixel, we remove it.

        # Define a depth threshold for proximity to the mesh
        depth_threshold = 0.05  # meters, adjust as needed

        # Get Gaussian 3D means (x, y, z)
        gaussian_means_3d = gaussians.mean_vectors[0].cpu().numpy()  # Nx3 tensor, convert to numpy

        # Store whether each gaussian should be kept or removed
        keep_gaussian = np.ones(len(gaussian_means_3d), dtype=bool)

        effective_mask = cv.erode(effective_mask, kernel, iterations=5)

        for i in range(len(gaussian_means_3d)):
            X, Y, Z_gaussian = gaussian_means_3d[i]
            # Skip invalid or negative Z values (behind camera)
            if Z_gaussian <= 0:
                keep_gaussian[i] = False
                continue

            # Project to 2D pixel coordinates (u, v) using intrinsics
            u = (f_x * X / Z_gaussian) + c_x
            v = (f_y * Y / Z_gaussian) + c_y

            int_u, int_v = int(round(u)), int(round(v))

            # Check if projected point is within image bounds
            if 0 <= int_v < h and 0 <= int_u < w:
                # Check if this pixel is part of the masked region (mesh region)
                if effective_mask[int_v, int_u] == 1:
                    mesh_depth = depth_blurred[int_v, int_u]

                    # Compare Gaussian depth with mesh depth if mesh depth is valid
                    if not np.isnan(mesh_depth) and mesh_depth > 0:
                        if abs(Z_gaussian - mesh_depth) < depth_threshold:
                            keep_gaussian[i] = False  # Gaussian is too close to the mesh
            # If not within mask or bounds, we keep it by default (as it's not "near the mesh")

        # Create new Gaussians3D object with filtered Gaussians
        # We assume Gaussians3D attributes are torch tensors and need to be indexed with a torch boolean tensor
        flag = torch.from_numpy(keep_gaussian).to("cpu")
        new_gaussians = Gaussians3D(
            mean_vectors=gaussians.mean_vectors[0][flag][None],
            singular_values=gaussians.singular_values[0][flag][None],
            quaternions=gaussians.quaternions[0][flag][None],
            colors=gaussians.colors[0][flag][None],
            opacities=gaussians.opacities[0][flag][None],
        )
        mesh_pruned_output_path = output_path / f"{image_path.stem}_pruned.ply"
        save_ply(new_gaussians, f_px, (height, width), mesh_pruned_output_path)

        LOGGER.info(f"Filtered Gaussians: {len(gaussian_means_3d) - len(new_gaussians)} removed. {len(new_gaussians)} remaining.")

        # TODO: Decide what to do with new_gaussians, e.g., save them or use for further processing


def save_depth(depth_np, name):
    depth_min = np.nanmin(depth_np)
    depth_max = np.nanmax(depth_np)
    print(depth_min, depth_max)
    if depth_max > depth_min:
        depth_norm = (depth_np - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth_np)
    depth_gray = (depth_norm * 255).astype(np.uint8)
    if depth_gray.ndim == 3:
        depth_gray = np.squeeze(depth_gray)

    io.save_image(depth_gray, name)

def flood_fill(depth, seed_point=(0, 0), max_diff=0.1):
    h, w = depth.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)  # +2 for the border required by OpenCV
    flooded_depth = depth.copy()
    cv.floodFill(flooded_depth, mask, seedPoint=seed_point, newVal=0, loDiff=max_diff, upDiff=max_diff)
    return flooded_depth, mask

if __name__ == "__main__":
    predict_cli()
