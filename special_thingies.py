#!/usr/bin/env python
import argparse as arg
import bitarray as ba
import numpy as np
import os
import functools
from icecream import ic
import subprocess
from sympy.utilities.iterables import multiset_permutations
import thingi10k
import trimesh

from thingies_utils import mesh_to_unit_cube

from thingies_with_omnitrees_plot import plot_binary_3d_omnitree_with_pyplot

from thingies_with_omnitrees import (
    get_initial_tree_and_queue,
    get_sobol_importances,
    skip_function_no_importance,
    tree_voxel_thingi,
)

from thingies_with_omnitrees_evaluate import (
    ErrorL1File,
    get_binary_discretization_occupancy,
    get_monte_carlo_l1_error,
    get_shannon_information,
    plot_mesh_with_pyplot,
    plot_mesh_with_opengl,
)


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
        "--thingi_index",
        type=int,
        help="index of the thingi to use, if not specified, all thingies will be used",
        default=None,
    )
    parser.add_argument(
        "--tree_index",
        type=int,
        help="index of the tree to use, if not specified, all trees will be used",
        default=None,
    )
    parser.add_argument(
        "--two-tier-criterion",
        action="store_true",
        help="use a two-tier criterion for the importance, first the variance, then the Sobol indices",
        default=False,
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="if present, use the temporal 4d version of the thingies",
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

    special_thingies: list[dict] = []

    # tetrahedron mesh
    mesh = trimesh.Trimesh(
        vertices=[[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]],
        faces=[[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        process=True,
    )

    mesh = mesh_to_unit_cube(mesh)

    special_thingies.append(
        {
            "mesh": mesh,
            "fake_file_id": 0,
        }
    )

    # icosphere mesh
    mesh = trimesh.creation.icosphere(subdivisions=4)
    mesh = mesh_to_unit_cube(mesh)
    special_thingies.append(
        {
            "mesh": mesh,
            "fake_file_id": 1,
        }
    )

    # construct a diagonal rod
    mesh = trimesh.creation.cylinder(
        radius=0.05,
        height=1,
        transform=trimesh.transformations.rotation_matrix(
            angle=np.pi / 4,
            direction=[1, 1, 0],
            point=[0.5, 0.5, 0.5],
        ),
    )
    mesh = mesh_to_unit_cube(mesh)
    special_thingies.append(
        {
            "mesh": mesh,
            "fake_file_id": 2,
        }
    )

    use_thingies = True
    if use_thingies:
        thingi10k.init()
        # use special thingies from thingi10k
        # 53750: Hilbert cube
        # 100349: kitty
        # 187279: cube
        for id in [53750, 100349, 187279, 99905]:
            thingi = thingi10k.dataset(file_id=id)[0]
            mesh_data = np.load(thingi["file_path"])
            mesh_vertices = mesh_data["vertices"]
            mesh_faces = mesh_data["facets"]
            mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
            mesh = mesh_to_unit_cube(mesh)
            special_thingies.append(
                {
                    "mesh": mesh,
                    "fake_file_id": id,
                }
            )

    for special_thingy in special_thingies:
        if not special_thingy["mesh"].is_watertight:
            raise ValueError(f"Mesh is not watertight")
        # plot the original mesh
        if args.temporal:
            for rotation in range(64):
                rotation_angle = (rotation + 0.5) / 64.0 * 2 * np.pi
                mesh_copy = special_thingy["mesh"].copy()
                mesh_copy.apply_transform(
                    trimesh.transformations.rotation_matrix(
                        rotation_angle, [1, 1, 1], (0.5, 0.5, 0.5)
                    )
                )
                mesh_copy = mesh_to_unit_cube(mesh_copy)
                plot_mesh_with_opengl(
                    mesh_copy,
                    filename=f"{special_thingy["fake_file_id"]}_original_t{rotation:03d}",
                )
        else:
            plot_mesh_with_opengl(
                special_thingy["mesh"],
                filename=str(special_thingy["fake_file_id"]) + "_original",
            )

    importance_function = functools.partial(
        get_sobol_importances,
        num_sobol_samples=args.sobol_samples,
        variance_as_first_criterion=args.two_tier_criterion,
    )
    skip_function = skip_function_no_importance

    if args.temporal:
        num_dimensions = 4
    else:
        num_dimensions = 3
    allowed_refinements_octree = {ba.frozenbitarray("1" * num_dimensions)}
    allowed_refinements_omnitree_1 = {
        ba.frozenbitarray(permutation)
        for permutation in multiset_permutations([1, 0, 0, 0], num_dimensions)
    }
    allowed_refinements_omnitree_1.discard(ba.frozenbitarray("0" * num_dimensions))

    error_file = ErrorL1File(args.sobol_samples)

    # if thingi_index is given, use only that one
    if args.thingi_index is not None:
        special_thingies = [
            special_thingies[args.thingi_index],
        ]

    for special_thingy in special_thingies:
        mesh = special_thingy["mesh"]
        fake_file_id = special_thingy["fake_file_id"]

        discretization_octree, queue_octree = get_initial_tree_and_queue(
            mesh, importance_function, list(allowed_refinements_octree)
        )
        discretization_omnitree_1, queue_omnitree_1 = get_initial_tree_and_queue(
            mesh, importance_function, list(allowed_refinements_omnitree_1)
        )
        tree_tuples = [
            (
                list(allowed_refinements_octree),
                discretization_octree,
                queue_octree,
                "octree",
            ),
            (
                list(allowed_refinements_omnitree_1),
                discretization_omnitree_1,
                queue_omnitree_1,
                "omnitree_1",
            ),
        ]
        if args.tree_index is not None:
            # if tree_index is given, use only that one
            tree_tuples = [tree_tuples[args.tree_index]]

        for allowed_refinements, discretization, queue, tree_name in tree_tuples:
            for allowed_tree_boxes in number_tree_boxes:
                discretization, queue = tree_voxel_thingi(
                    discretization,
                    queue,
                    mesh,
                    allowed_tree_boxes,
                    importance_function,
                    skip_function,
                    allowed_refinements,
                )
                filename_tree = (
                    str(fake_file_id)
                    + "_"
                    + tree_name
                    + "_"
                    + str(allowed_tree_boxes)
                    + "_s"
                    + str(args.sobol_samples)
                )
                discretization.descriptor.to_file(filename_tree)

                # and evaluate the error immediately
                number_error_samples = 262144
                number_occupancy_samples = args.sobol_samples * 8
                if num_dimensions == 4:
                    number_error_samples = 16777216
                    number_occupancy_samples = args.sobol_samples * 16

                binary_discretization_occupancy = get_binary_discretization_occupancy(
                    discretization, mesh, number_occupancy_samples
                )
                with open(filename_tree + "_occupancy.bin", "wb") as f:
                    ba.bitarray(binary_discretization_occupancy.tolist()).tofile(f)
                assert len(binary_discretization_occupancy) == len(discretization)
                # get the shannon information both in the tree descriptor and in the function
                tree_information = get_shannon_information(
                    discretization.descriptor.get_data()
                )
                function_information = get_shannon_information(
                    binary_discretization_occupancy
                )
                ic(tree_information, function_information)

                monte_carlo_l1_error = get_monte_carlo_l1_error(
                    mesh,
                    discretization,
                    binary_discretization_occupancy,
                    number_error_samples,
                )
                ic(monte_carlo_l1_error)

                num_boxes_occupied = np.sum(binary_discretization_occupancy)
                ic(num_boxes_occupied)

                error_file.append_row(
                    {
                        "thingi_file_id": fake_file_id,
                        "tree": tree_name,
                        "allowed_tree_boxes": allowed_tree_boxes,
                        "num_sobol_samples": args.sobol_samples,
                        "num_occupancy_samples": number_occupancy_samples,
                        "num_error_samples": number_error_samples,
                        "num_boxes": len(discretization),
                        "num_boxes_occupied": num_boxes_occupied,
                        "num_tree_nodes": len(discretization.descriptor),
                        "tree_number_of_1s": discretization.descriptor.get_data().count(),
                        "l1error": monte_carlo_l1_error,
                    }
                )

    # TODO plot and merge to svg separately

    # # # call the thingies_merge_svgs.py script to merge the SVGs
    # # subprocess.run(
    # #     ["python3", os.path.dirname(__file__) + "/thingies_merge_svgs.py"], check=True
    # # )
