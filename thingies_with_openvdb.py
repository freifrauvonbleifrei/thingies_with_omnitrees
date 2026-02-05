#!/usr/bin/env python
# install openvdb with spack install openvdb@13.0.0+python
# (needs some adaptations to openvdb spack package)

import argparse as arg
from icecream import ic
import numpy as np
import math
import openvdb as vdb
import trimesh

from thingies_utils import mesh_to_unit_cube
from thingies_with_omnitrees_evaluate import check_inside_or_outside_mesh, ErrorL1File
from special_thingies import get_special_thingies


def get_regular_grid_occupancy(
    mesh: trimesh.Trimesh,
    voxel_size: float,
    num_samples: int,
):
    if not mesh.is_watertight:
        raise ValueError("Mesh must be watertight")

    random_points_3d = np.random.rand(num_samples, 3)

    # Compute grid resolution
    dims = np.ceil(1.0 / voxel_size).astype(int)

    # Generate voxels' lower bounds
    xs = np.arange(dims) * voxel_size
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    lower_corners = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    is_inside = np.zeros(len(lower_corners), dtype=bool)

    for i, lower_corner in enumerate(lower_corners):
        upper_corner = lower_corner + voxel_size
        # move random points to the interval
        random_points_in_interval = (
            random_points_3d * (upper_corner - lower_corner) + lower_corner
        )
        is_inside[i] = (
            check_inside_or_outside_mesh(mesh, random_points_in_interval).mean() > 0.5
        )
    return is_inside, dims


def mesh_to_boolgrid(
    mesh: trimesh.Trimesh,
    voxel_size: float = 0.002,
    num_samples: int = 10000,
    grid_name: str = "inside",
) -> vdb.BoolGrid:
    """
    Convert a watertight triangle mesh into a BoolGrid where
    True = inside mesh, False = outside mesh.
    """
    is_inside, resolution = get_regular_grid_occupancy(mesh, voxel_size, num_samples)

    # Create BoolGrid
    grid = vdb.BoolGrid(False)
    grid.name = grid_name

    # Set transform (voxel -> world)
    transform = vdb.createLinearTransform(voxel_size)
    grid.transform = transform

    # Write inside voxels
    accessor = grid.getAccessor()
    idx = np.argwhere(is_inside).flatten()
    for linear_idx in idx:
        i = int(linear_idx // (resolution * resolution))
        j = int((linear_idx // resolution) % resolution)
        k = int(linear_idx % resolution)
        accessor.setValueOn((i, j, k), True)

    return grid


def get_monte_carlo_l1_error_openvdb(mesh, bool_grid, num_samples=10000):
    num_dimensions = 3
    points = np.random.rand(num_samples, num_dimensions)

    is_inside_mesh = check_inside_or_outside_mesh(mesh, points)
    is_inside_boolgrid = np.zeros(num_samples, dtype=bool)
    acc = bool_grid.getAccessor()

    def query(grid, point: tuple[float]):
        ijk = grid.transform.worldToIndex(point)
        ijk = tuple(int(math.floor(c)) for c in ijk)
        if acc.isValueOn(ijk):
            return acc.getValue(ijk)
        else:
            return False

    for i, point in enumerate(points):
        is_inside_boolgrid[i] = query(bool_grid, point)

    # calculate the L1 error
    return (is_inside_mesh ^ is_inside_boolgrid).mean()


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument(
        "number_tree_boxes",
        type=str,
        help="number of boxes allowed in tree descriptors, or a range of them when powers of 2 (upper-inclusive)",
    )
    parser.add_argument(
        "--sobol_samples",
        type=int,
        help="number of samples for the Sobol criterion, needs to be a power of 2 (and will be multiplied by 8!)",
        default=512,
    )
    parser.add_argument(
        "--plane",
        action="store_true",
        help="if present, run only the F25 model, assumes stl file in parent directory",
    )
    parser.add_argument(
        "--thingi_index",
        type=int,
        help="index of the thingi to use, if not specified, all thingies will be used",
        default=None,
    )
    args = parser.parse_args()
    parsed_number_tree_boxes = args.number_tree_boxes.split("-")
    if len(parsed_number_tree_boxes) == 1:
        number_tree_boxes = [int(parsed_number_tree_boxes[0])]
    elif len(parsed_number_tree_boxes) == 2:
        number_tree_boxes = []
        number_boxes = int(parsed_number_tree_boxes[0])
        while number_boxes <= int(parsed_number_tree_boxes[1]):
            number_tree_boxes.append(number_boxes)
            number_boxes *= 2
    else:
        raise ValueError("wrong formatting for number_tree_boxes")

    special_thingies = get_special_thingies(args.plane)
    # if thingi_index is given, use only that one
    if args.thingi_index is not None:
        special_thingies = [
            special_thingies[args.thingi_index],
        ]

    error_file = ErrorL1File(args.sobol_samples)
    for special_thingy in special_thingies:
        mesh = special_thingy["mesh"]
        fake_file_id = special_thingy["fake_file_id"]
        number_error_samples = 262144
        number_occupancy_samples = args.sobol_samples * 8
        voxel_size = 0.5
        current_num_leaves = 0
        while current_num_leaves < max(number_tree_boxes):
            voxel_size *= 0.5
            # downsample to the respective number of allowed boxes
            boolgrid = mesh_to_boolgrid(mesh, voxel_size, number_occupancy_samples)
            boolgrid.pruneInactive()
            monte_carlo_l1_error = get_monte_carlo_l1_error_openvdb(
                mesh,
                boolgrid,
                number_error_samples,
            )
            current_num_leaves = boolgrid.leafCount()
            ic(boolgrid.memUsage(), boolgrid.leafCount(), monte_carlo_l1_error)

            error_file.append_row(
                {
                    "thingi_file_id": fake_file_id,
                    "tree": "openvdb",
                    "allowed_tree_boxes": 0,
                    "num_sobol_samples": args.sobol_samples,
                    "num_occupancy_samples": number_occupancy_samples,
                    "num_error_samples": number_error_samples,
                    "num_boxes": boolgrid.leafCount(),
                    "num_boxes_occupied": 0,
                    "num_tree_nodes": boolgrid.memUsage(),
                    "tree_number_of_1s": boolgrid.activeVoxelCount(),
                    "l1error": monte_carlo_l1_error,
                }
            )
            vdb.write(
                f"{fake_file_id}_openvdb_{boolgrid.leafCount()}.vdb",
                boolgrid,
            )
