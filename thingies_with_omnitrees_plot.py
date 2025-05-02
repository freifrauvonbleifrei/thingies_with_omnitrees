#!/usr/bin/env python
import argparse as arg
import bitarray as ba
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
import os.path

import dyada.coordinates
import dyada.drawing
import dyada.linearization
import dyada.refinement


def plot_binary_3d_omnitree_with_pyplot(
    discretization: dyada.refinement.Discretization,
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
            ax, coordinate, color="orange", edgecolors="black", linewidth=0.3
        )

    if filename is not None:
        fig.savefig(filename + ".svg", dpi=600, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument(
        "occupancy_file",
        type=str,
        help="occupancy file ending with '_occupancy.bin', needs the corrensponding tree file '_3d.bin' in the same folder too",
    )
    args = parser.parse_args()

    filename_tree = args.occupancy_file[:-14] + "_3d.bin"

    if not os.path.isfile(filename_tree):
        print(filename_tree + " does not exist, returning")
        exit(1)
    if not os.path.isfile(args.occupancy_file):
        print(args.occupancy_file + " does not exist, returning")
        exit(1)

    discretization = dyada.refinement.Discretization(
        dyada.linearization.MortonOrderLinearization(),
        dyada.refinement.RefinementDescriptor.from_file(filename_tree),
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
    filename_svg = filename_tree[:-7] + "_eval"
    if not num_boxes_occupied == 0:
        plot_binary_3d_omnitree_with_pyplot(
            discretization,
            binary_discretization_occupancy,
            azim=220,
            filename=filename_svg,
        )
