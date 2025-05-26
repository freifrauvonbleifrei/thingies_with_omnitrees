#!/usr/bin/env python
import argparse as arg
import bitarray as ba
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from itertools import pairwise
import os.path

import dyada.coordinates
import dyada.discretization
import dyada.drawing
import dyada.linearization

from thingies_with_omnitrees_evaluate import get_all_time_slices


def plot_binary_3d_omnitree_with_pyplot(
    discretization: dyada.discretization.Discretization,
    binary_occupancy,
    azim=200,
    filename=None,
):
    level_indices = list(discretization.get_all_boxes_level_indices())
    coordinates = [
        dyada.coordinates.get_coordinates_from_level_index(box_li)
        for box_li in level_indices
    ]
    fig, ax = dyada.drawing.get_figure_3d_matplotlib(
        coordinates,
        labels=None,
        wireframe=True,
        alpha=0.3,
        colors=["gray"],
        linewidth=0.09,
    )
    ax.set_title("")
    ax.text2D(
        0.6,
        0.93,
        "filled: " + f"{sum(binary_occupancy):>4}" + " / " + f"{len(coordinates):>4}",
        transform=ax.transAxes,
    )
    ax.view_init(azim=azim, share=True)  # cf. evaluate script
    ax.plot([0, 1], [0, 1], [0, 1])

    # filter coordinates by binary occupancy
    coordinates = [
        coordinates[i] for i in range(len(coordinates)) if binary_occupancy[i] == 1
    ]
    for coordinate in coordinates:
        ax = dyada.drawing.draw_cuboid_on_axis(
            ax,
            coordinate,
            color="orange",
            edgecolors="black",
            linewidth=0.3,
        )

    if filename is not None:
        fig.savefig(filename + ".svg", dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_binary_3d_omnitree_with_opengl(
    discretization: dyada.discretization.Discretization, binary_occupancy, filename
):
    level_indices = list(discretization.get_all_boxes_level_indices())
    coordinates = [
        dyada.coordinates.get_coordinates_from_level_index(box_li)
        for box_li in level_indices
    ]
    projection = [0, 2, 1]
    dyada.drawing.plot_boxes_3d_pyopengl(
        coordinates,
        wireframe=True,
        alpha=0.3,
        colors="gray",
        filename=None,
        projection=projection,
    )

    # filter coordinates by binary occupancy
    coordinates = [
        coordinates[i] for i in range(len(coordinates)) if binary_occupancy[i] == 1
    ]
    for coordinate in coordinates:
        dyada.drawing.draw_cuboid_opengl(
            coordinate,
            wireframe=False,
            alpha=0.15,
            color="orange",
            projection=projection,
        )
    dyada.drawing.gl_save_file(filename)


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument(
        "occupancy_file",
        type=str,
        help="occupancy file ending with '_occupancy.bin', needs the corrensponding tree file '_3d.bin' in the same folder too",
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="either 'opengl' or 'matplotlib'",
        default="matplotlib",
    )
    args = parser.parse_args()

    filename_tree_3d = args.occupancy_file[:-14] + "_3d.bin"
    filename_tree_4d = args.occupancy_file[:-14] + "_4d.bin"
    filename_tree = filename_tree_3d

    if not os.path.isfile(filename_tree_3d):
        if os.path.isfile(filename_tree_4d):
            filename_tree = filename_tree_4d
        else:
            print(
                filename_tree_3d
                + " or "
                + filename_tree_4d
                + " does not exist, returning"
            )
        exit(1)
    if not os.path.isfile(args.occupancy_file):
        print(args.occupancy_file + " does not exist, returning")
        exit(1)

    discretization = dyada.discretization.Discretization(
        dyada.linearization.MortonOrderLinearization(),
        dyada.discretization.RefinementDescriptor.from_file(filename_tree),
    )

    with open(args.occupancy_file, "rb") as f:
        binary_discretization_occupancy_ba = ba.bitarray()
        ba.bitarray.fromfile(binary_discretization_occupancy_ba, f)

    # crop bitarray to length of discretization
    binary_discretization_occupancy_ba = binary_discretization_occupancy_ba[
        : len(discretization)
    ]
    # and convert to numpy array
    binary_discretization_occupancy = np.array(
        binary_discretization_occupancy_ba.tolist(), dtype=bool
    )

    num_boxes_occupied = np.sum(binary_discretization_occupancy)
    ic(filename_tree[:-7], num_boxes_occupied)
    filename_img = filename_tree[:-7] + "_eval"
    assert len(discretization) == len(binary_discretization_occupancy)
    ic(len(discretization), len(binary_discretization_occupancy))
    if filename_tree == filename_tree_3d:
    if args.backend == "opengl":
        plot_binary_3d_omnitree_with_opengl(
            discretization,
            binary_discretization_occupancy,
            filename=filename_img,
        )
    else:
        plot_binary_3d_omnitree_with_pyplot(
            discretization,
            binary_discretization_occupancy,
            azim=220,
            filename=filename_img,
        )
    else:
        # get all the time sliced discretizations
        time_slices = get_all_time_slices(
            discretization, binary_discretization_occupancy
        )
        ic(len(time_slices), time_slices.keys())

        for time_i in range(100):
            time = time_i * 0.01
            time_found = False
            for t_slice_time_lower, t_slice_time_upper in pairwise(time_slices.keys()):
                if time >= t_slice_time_lower and time < t_slice_time_upper:
                    time_found = True
                    ic(f"Plotting time slice {time_i} at time {t_slice_time_lower}")
                    discretization_at_time, binary_discretization_occupancy_at_time = (
                        time_slices[t_slice_time_lower]
                    )
                    assert args.backend == "opengl"
                    filename_img_at_time = filename_img + f"_t{time_i:03d}"
                    plot_binary_3d_omnitree_with_opengl(
                        discretization_at_time,
                        binary_discretization_occupancy_at_time,
                        filename=filename_img_at_time,
                    )
                    break
            assert time_found, (
                "Time slice not found for time",
                time,
                t_slice_time_lower,
                t_slice_time_upper,
            )
