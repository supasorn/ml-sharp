"""Command-line-interface to run SHARP model.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

import click

from . import predict, render, predict_with_info, dof_edit, quest_lod, bake_layers, bake_hybrid
from .render_mask import render_mask_cli



@click.group()
def main_cli():
    """Run inference for SHARP model."""
    pass


main_cli.add_command(predict.predict_cli, "predict")
main_cli.add_command(render.render_cli, "render")
main_cli.add_command(render_mask_cli, "render-mask")
main_cli.add_command(predict_with_info.predict_cli, "predict-with-info")
main_cli.add_command(dof_edit.dof_edit_cli, "dof-edit")
main_cli.add_command(quest_lod.quest_lod_cli, "quest-lod")
main_cli.add_command(bake_layers.bake_layers_cli, "bake-layers")
main_cli.add_command(bake_hybrid.bake_hybrid_cli, "bake-hybrid")

