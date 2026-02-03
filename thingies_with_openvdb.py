#!/usr/bin/env python
# install openvdb with spack install openvdb@13.0.0+python
# (needs some adaptations to openvdb spack package)

import argparse as arg
from icecream import ic
import numpy as np
import math
import openvdb as vdb

# import thingi10k
import trimesh

from thingies_utils import mesh_to_unit_cube
from thingies_with_omnitrees_evaluate import check_inside_or_outside_mesh


def mesh_to_boolgrid(
    mesh: trimesh.Trimesh, voxel_size: float = 0.004, grid_name: str = "inside"
) -> vdb.BoolGrid:
    """
    Convert a watertight triangle mesh into a BoolGrid where
    True = inside mesh, False = outside mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Watertight triangle mesh
    voxel_size : float
        Size of a voxel in world units
    padding : int
        Number of voxels to pad around mesh bounds
    grid_name : str
        Name of the OpenVDB grid

    Returns
    -------
    vdb.BoolGrid
    """

    if not mesh.is_watertight:
        raise ValueError("Mesh must be watertight")

    # Compute mesh bounds
    bounds_min = (0.0, 0.0, 0.0)
    bounds_max = (1.0, 1.0, 1.0)

    # Compute grid resolution
    dims = np.ceil(1.0 / voxel_size).astype(int)

    # Generate voxel center coordinates
    xs = np.arange(dims) * voxel_size + bounds_min[0] + voxel_size * 0.5
    ys = np.arange(dims) * voxel_size + bounds_min[1] + voxel_size * 0.5
    zs = np.arange(dims) * voxel_size + bounds_min[2] + voxel_size * 0.5

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    # Point-in-mesh test
    inside = mesh.contains(points)
    ic(sum(inside), len(inside))

    # Create BoolGrid
    grid = vdb.BoolGrid(False)
    grid.name = grid_name

    # Set transform (voxel -> world)
    transform = vdb.createLinearTransform(voxel_size)
    grid.transform = transform

    accessor = grid.getAccessor()

    # Write inside voxels
    idx = np.argwhere(inside).flatten()
    for linear_idx in idx:
        i = int(linear_idx // (dims * dims))
        j = int((linear_idx // dims) % dims)
        k = int(linear_idx % dims)
        accessor.setValueOn((i, j, k), True)
        # accessor.setValue((i, j, k), True)

    return grid


def get_monte_carlo_l1_error_openvdb(mesh, bool_grid, num_samples=10000):
    # Sample random points in and around the mesh bounding box
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


def downsample_boolgrid(
    grid: vdb.BoolGrid, scale_factor: int, threshold: float = 0.5
) -> vdb.BoolGrid:
    """
    Downsample a BoolGrid by a scale factor (>1 = fewer voxels).
    """
    # Extract active voxel bounding box
    bbox = grid.evalActiveVoxelBoundingBox()
    ijk_min, ijk_max = bbox

    dims = (
        ijk_max[0] - ijk_min[0] + 1,
        ijk_max[1] - ijk_min[1] + 1,
        ijk_max[2] - ijk_min[2] + 1,
    )

    arr = np.zeros(dims, dtype=np.bool_)
    grid.copyToArray(arr, ijk=ijk_min)

    # Trim to multiple of factor
    new_dims = (
        dims[0] // scale_factor,
        dims[1] // scale_factor,
        dims[2] // scale_factor,
    )
    arr = arr[
        : new_dims[0] * scale_factor,
        : new_dims[1] * scale_factor,
        : new_dims[2] * scale_factor,
    ]

    # Downsample
    arr_ds = (
        arr.reshape(
            new_dims[0],
            scale_factor,
            new_dims[1],
            scale_factor,
            new_dims[2],
            scale_factor,
        ).mean(axis=(1, 3, 5), dtype=np.float32)
        > threshold
    )

    # Resample transform
    out_grid = vdb.BoolGrid(False)
    out_grid.transform = vdb.createLinearTransform(
        grid.transform.voxelSize()[0] * scale_factor
    )

    out_grid.copyFromArray(arr_ds, ijk=(0, 0, 0))
    return out_grid


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

    special_thingies = [
        {
            "mesh": mesh_to_unit_cube(
                trimesh.load_mesh("../f25_no_wheels.stl", file_type="stl")
            ),
            "fake_file_id": 25,  # F25 model
        }
    ]

    for special_thingy in special_thingies:
        mesh = special_thingy["mesh"]
        fake_file_id = special_thingy["fake_file_id"]

        fine_boolgrid = mesh_to_boolgrid(mesh)
        ic(fine_boolgrid.memUsage(), fine_boolgrid.leafCount())
        number_error_samples = 262144
        number_occupancy_samples = args.sobol_samples * 8
        monte_carlo_l1_error = get_monte_carlo_l1_error_openvdb(
            mesh,
            fine_boolgrid,
            number_error_samples,
        )
        vdb.write(
            f"{special_thingy['fake_file_id']}_openvdb_{fine_boolgrid.leafCount()}.vdb",
            fine_boolgrid,
        )
        scale_factor = 1
        for allowed_tree_boxes in number_tree_boxes:
            scale_factor *= 2
            # downsample to the respective number of allowed boxes
            downsampled_boolgrid = downsample_boolgrid(fine_boolgrid, scale_factor)
            ic(downsampled_boolgrid.memUsage(), downsampled_boolgrid.leafCount())
            monte_carlo_l1_error = get_monte_carlo_l1_error_openvdb(
                mesh,
                downsampled_boolgrid,
                number_error_samples,
            )
            ic(monte_carlo_l1_error)
            vdb.write(
                f"{special_thingy['fake_file_id']}_openvdb_{downsampled_boolgrid.leafCount()}.vdb",
                downsampled_boolgrid,
            )
