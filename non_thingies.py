#!/usr/bin/env python
import argparse as arg
import bitarray as ba
import numpy as np
import os
import functools
from icecream import ic
import subprocess
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
        default=64,
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

    # tetrahedron mesh
    mesh = trimesh.Trimesh(
        vertices=[[1, 0, 0], [0, 0, 0], [1, 1, 0], [1, 0, 1]],
        faces=[[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        process=True,
    )

    if not mesh.is_watertight:
        raise ValueError(f"Mesh is not watertight")
    mesh = mesh_to_unit_cube(mesh)
    fake_file_id = 0
    azim = 105
    plot_mesh_with_pyplot(mesh, azim=azim, filename=str(fake_file_id) + "_original")

    importance_function = functools.partial(
        get_sobol_importances, num_sobol_samples=args.sobol_samples
    )
    skip_function = skip_function_no_importance

    allowed_refinements_octree = [ba.bitarray("111")]
    allowed_refinements_omnitree_1 = [
        ba.bitarray("100"),
        ba.bitarray("010"),
        ba.bitarray("001"),
    ]
    allowed_refinements_omnitree_2 = [
        ba.bitarray("100"),
        ba.bitarray("010"),
        ba.bitarray("001"),
        ba.bitarray("110"),
        ba.bitarray("011"),
        ba.bitarray("101"),
    ]
    allowed_refinements_omnitree_3 = [
        ba.bitarray("100"),
        ba.bitarray("010"),
        ba.bitarray("001"),
        ba.bitarray("110"),
        ba.bitarray("011"),
        ba.bitarray("101"),
        ba.bitarray("111"),
    ]

    discretization_octree, queue_octree = get_initial_tree_and_queue(
        mesh, importance_function, allowed_refinements_octree
    )
    discretization_omnitree_1, queue_omnitree_1 = get_initial_tree_and_queue(
        mesh, importance_function, allowed_refinements_omnitree_1
    )
    discretization_omnitree_2, queue_omnitree_2 = get_initial_tree_and_queue(
        mesh, importance_function, allowed_refinements_omnitree_2
    )
    discretization_omnitree_3, queue_omnitree_3 = get_initial_tree_and_queue(
        mesh, importance_function, allowed_refinements_omnitree_3
    )

    tree_tuples = [
        (allowed_refinements_octree, discretization_octree, queue_octree, "octree"),
        (
            allowed_refinements_omnitree_1,
            discretization_omnitree_1,
            queue_omnitree_1,
            "omnitree_1",
        ),
        (
            allowed_refinements_omnitree_2,
            discretization_omnitree_2,
            queue_omnitree_2,
            "omnitree_2",
        ),
        (
            allowed_refinements_omnitree_3,
            discretization_omnitree_3,
            queue_omnitree_3,
            "omnitree_3",
        ),
    ]

    error_file = ErrorL1File(str(args.sobol_samples))

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
            number_error_samples = 131072
            number_occupancy_samples = 2048
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
                    "num_boxes_occupied": num_boxes_occupied,
                    "num_error_samples": number_error_samples,
                    "num_boxes": len(discretization),
                    "tree_information": get_shannon_information(
                        discretization.descriptor.get_data()
                    ),
                    "occupancy_information": get_shannon_information(
                        binary_discretization_occupancy
                    ),
                    "l1error": monte_carlo_l1_error,
                }
            )

            filename_svg = filename_tree + "_eval"
            if not num_boxes_occupied == 0:
                plot_binary_3d_omnitree_with_pyplot(
                    discretization,
                    binary_discretization_occupancy,
                    azim=azim,
                    filename=filename_svg,
                )

    # call the thingies_merge_svgs.py script to merge the SVGs
    subprocess.run(
        ["python3", os.path.dirname(__file__) + "/thingies_merge_svgs.py"], check=True
    )
