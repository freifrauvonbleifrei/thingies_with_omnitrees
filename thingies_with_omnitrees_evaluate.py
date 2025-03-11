#!/usr/bin/env python
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
import os.path
import thingi10k
import trimesh

import dyada.drawing
import dyada.linearization
import dyada.refinement


from thingies_utils import mesh_to_unit_cube, check_inside_or_outside_mesh


def check_inside_or_outside_tree(
    discretization: dyada.refinement.Discretization,
    binary_discretization_occupancy: np.ndarray,
    points: np.ndarray,
) -> np.array:
    is_inside = np.zeros(len(points), dtype=bool)
    for p_i, point in enumerate(points):
        box_index = discretization.get_containing_box(point)
        if binary_discretization_occupancy[box_index]:
            is_inside[p_i] = True
    return is_inside


def get_monte_carlo_l1_error(
    mesh: trimesh.Geometry,
    discretization: dyada.refinement.Discretization,
    binary_discretization_occupancy: np.ndarray,
    num_samples: int,
) -> float:
    # generate random points in the unit cube
    points = np.random.rand(num_samples, 3)

    is_inside_mesh = check_inside_or_outside_mesh(mesh, points)
    is_inside_tree = check_inside_or_outside_tree(
        discretization, binary_discretization_occupancy, points
    )

    # calculate the L1 error
    return (is_inside_mesh ^ is_inside_tree).mean()


def get_binary_discretization_occupancy(
    discretization: dyada.refinement.Discretization,
    mesh: trimesh.Geometry,
    num_samples: int,
):
    binary_discretization_occupancy = np.zeros(len(discretization), dtype=bool)
    for box_index in range(len(discretization)):
        interval = dyada.refinement.coordinates_from_box_index(
            discretization, box_index
        )
        # get random points in the interval
        random_points_in_interval = (
            np.random.rand(num_samples, 3)
            * (interval.upper_bound - interval.lower_bound)
            + interval.lower_bound
        )
        if np.mean(check_inside_or_outside_mesh(mesh, random_points_in_interval)) > 0.5:
            binary_discretization_occupancy[box_index] = True
    return binary_discretization_occupancy


def get_mesh_from_discretization(discretization, binary_discretization_occupancy):
    # construct a mesh from the discretization and binary values
    list_of_boxes = []
    for box_index in range(len(discretization)):
        if binary_discretization_occupancy[box_index]:
            interval = dyada.refinement.coordinates_from_box_index(
                discretization, box_index
            )
            # box = trimesh.creation.box(bounds=interval)
            box = trimesh.creation.box(
                extents=interval.upper_bound - interval.lower_bound,
                transform=trimesh.transformations.translation_matrix(
                    interval.lower_bound
                    + (interval.upper_bound - interval.lower_bound) / 2
                ),
            )
            list_of_boxes.append(box)
    return trimesh.util.concatenate(list_of_boxes)


def plot_with_pyplot(mesh: trimesh.Geometry, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.view_init(azim=220, share=True)
    ax.plot([0, 1], [0, 1], [0, 1])
    ax.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
    )
    if filename is not None:
        fig.savefig(filename + ".svg", dpi=300)
        plt.close()
    else:
        plt.show()


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
    )  # TODO replace by which files are found
    parser.add_argument(
        "--slice",
        type=str,
        help="which slice of the data set this should work on, zero-indexed",
        default="0/2048",
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

    parsed_slice = args.slice.split("/")
    assert len(parsed_slice) == 2
    my_slice = int(parsed_slice[0])
    num_slices = int(parsed_slice[1])
    assert my_slice < num_slices

    thingi10k.init(variant="raw")

    # select thingi meshes by closedness, having at most 10000 vertices, etc.
    subset = thingi10k.dataset(
        num_vertices=(None, 10000),
        closed=True,
        self_intersecting=False,
        solid=True,
    )
    chunk_size = len(subset) / num_slices
    ic(len(subset["file_id"]))
    subset = thingi10k.dataset(
        file_id=subset["file_id"][
            round(my_slice * chunk_size) : round((my_slice + 1) * chunk_size)
        ]
    )
    # or, select thingies by interest
    # interesting_ids = [100349]
    # subset = thingi10k.dataset(file_id=interesting_ids)

    num_my_thingies = len(subset["file_id"])
    ic(num_my_thingies, subset)

    for thingi in subset:
        print(thingi)
        mesh = trimesh.load_mesh(thingi["file_path"], file_type="stl")
        if not mesh.is_watertight:
            continue
        mesh = mesh_to_unit_cube(mesh)
        plot_with_pyplot(mesh, str(thingi["file_id"]) + "_original")

        tree_names = ["octree", "omnitree_1", "omnitree_2", "omnitree_3"]

        for allowed_tree_boxes in number_tree_boxes:
            for tree_name in tree_names:
                ic(allowed_tree_boxes, tree_name)
                filename_tree = (
                    str(thingi["file_id"])
                    + "_"
                    + tree_name
                    + "_"
                    + str(allowed_tree_boxes)
                    + "_s"
                    + str(args.sobol_samples)
                )
                filename_binary = filename_tree + "_3d.bin"
                if not os.path.isfile(filename_binary):
                    print(filename_binary + " does not exist, skipping")
                    continue

                discretization = dyada.refinement.Discretization(
                    dyada.linearization.MortonOrderLinearization(),
                    dyada.refinement.RefinementDescriptor.from_file(filename_binary),
                )

                number_error_samples = 131072
                number_occupancy_samples = 2048
                binary_discretization_occupancy = get_binary_discretization_occupancy(
                    discretization, mesh, number_occupancy_samples
                )
                assert len(binary_discretization_occupancy) == len(discretization)
                monte_carlo_l1_error = get_monte_carlo_l1_error(
                    mesh,
                    discretization,
                    binary_discretization_occupancy,
                    number_error_samples,
                )
                ic(monte_carlo_l1_error)

                # construct a mesh from the discretization
                mesh_from_tree = get_mesh_from_discretization(
                    discretization, binary_discretization_occupancy
                )
                ic(np.sum(binary_discretization_occupancy))
                # dyada.drawing.plot_all_boxes_3d(discretization, labels=None, wireframe=True, filename=filename)
                ic(mesh_from_tree)
                filename = filename_tree + "_eval"
                if not mesh_from_tree.is_empty:
                    plot_with_pyplot(mesh_from_tree, filename)
