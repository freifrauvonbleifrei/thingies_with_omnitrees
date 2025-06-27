#!/usr/bin/env python
import argparse as arg
import bitarray as ba
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
import io
import os.path
import pandas as pd

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
        height=512,
        width=512,
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
    dyada.drawing.gl_save_file(filename, width=512, height=512)


def export_binary_3d_omnitree_to_obj(
    discretization: dyada.discretization.Discretization, binary_occupancy, filename
):
    level_indices = list(discretization.get_all_boxes_level_indices())
    coordinates = [
        dyada.coordinates.get_coordinates_from_level_index(box_li)
        for box_li in level_indices
    ]
    projection = [0, 2, 1]
    buffer_obj, vertex_offset = dyada.drawing.export_boxes_3d_to_obj(
        coordinates,
        projection=projection,
        wireframe=True,
        filename=None,
    )

    # filter coordinates by binary occupancy
    coordinates = [
        coordinates[i] for i in range(len(coordinates)) if binary_occupancy[i] == 1
    ]
    for coordinate in coordinates:
        buffer_obj, vertex_offset = dyada.drawing.add_cuboid_to_buffer(
            buffer_obj,
            vertex_offset,
            coordinate,
            projection=projection,
            wireframe=False,
        )
    dyada.drawing.write_obj_file(
        buffer_obj,
        filename,
    )


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
        help="either 'opengl', or 'matplotlib', or 'obj'",
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
        elif args.backend == "obj":
            export_binary_3d_omnitree_to_obj(
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
        ic(len(time_slices), list(time_slices.keys()))

        assert args.backend == "opengl"
        slice_image_map: dict = {}
        timeline_string = ""
        for time_i in range(64):
            time = (time_i + 0.5) * (1.0 / 64.0)
            discretization_at_time, binary_discretization_occupancy_at_time = (
                time_slices[time]
            )
            filename_img_at_time = filename_img + f"_t{time_i:03d}"
            key = (
                discretization_at_time.descriptor,
                binary_discretization_occupancy_at_time.tobytes(),
            )
            if key in slice_image_map:
                # # copy existing image #useful for naive gif, but not otherwise
                # filename_img_at_previous_time = slice_image_map[key]
                # ic("using existing image", filename_img_at_previous_time)
                # subprocess.run(
                #     [
                #         "cp",
                #         filename_img_at_previous_time + ".png",
                #         filename_img_at_time + ".png",
                #     ]
                # )
                pass
            else:
                ic("creating", filename_img_at_time + ".png")
                plot_binary_3d_omnitree_with_opengl(
                    discretization_at_time,
                    binary_discretization_occupancy_at_time,
                    filename=filename_img_at_time,
                )
                slice_image_map[key] = filename_img_at_time
                # add comment line containing the file name
                timeline_string += (
                    "% \\includegraphics[width=\\linewidth]{"
                    + str(filename_img_at_time)
                    + ".png}"
                    + f"\n"
                )
            # add timeline file line containing the index of the image to use
            timeline_string += f"::{len(slice_image_map)-1} \n"
            # at the end of timeline, write all images in one comment again
            timeline_string += (
                "% \\def\\filelist{{ "
                + "}, {".join([f"{img}.png" for img in slice_image_map.values()])
                + "}}"
                + f"\n"
            )
        # save the timeline file
        timeline_filename = filename_img + "_timeline.txt"
        # with open(timeline_filename, "w") as f:
        #     f.write(timeline_string)

        possible_trees = ["octree", "omnitree_1"]
        for tree in possible_trees:
            if tree in filename_img:
                # see if other timeline file is already there
                octree_timeline_filename = timeline_filename.replace(tree, "octree")
                omnitree_timeline_filename = timeline_filename.replace(
                    tree, "omnitree_1"
                )
                if os.path.isfile(octree_timeline_filename):
                    if os.path.isfile(omnitree_timeline_filename):
                        # first dataframe is the iota frame from 0 to 63
                        df_original_timeline = pd.Series(range(64))
                        # read octree one as df
                        df_octree_timeline = pd.read_csv(
                            octree_timeline_filename,
                            sep=":",
                            engine="python",
                            header=None,
                            comment="%",
                        )
                        # use only the last column
                        df_octree_timeline = df_octree_timeline.iloc[:, -1]
                        # and offset all values by previous ones
                        df_octree_timeline += len(df_original_timeline)
                        octree_unique_values = df_octree_timeline.unique()
                        # same for omnitree
                        df_omni_timeline = pd.read_csv(
                            omnitree_timeline_filename,
                            sep=":",
                            engine="python",
                            header=None,
                            comment="%",
                        )
                        df_omni_timeline = df_omni_timeline.iloc[:, -1]
                        df_omni_timeline += len(octree_unique_values) + len(
                            df_original_timeline
                        )
                        # try to merge the three timelines
                        ic(
                            df_original_timeline,
                            df_octree_timeline,
                            df_omni_timeline,
                        )
                        df_combined = pd.concat(
                            [
                                df_original_timeline,
                                df_octree_timeline,
                                df_omni_timeline,
                            ],
                            axis=1,
                            ignore_index=True,
                        )
                        ic(df_combined)
                        # and save it, format per line should be :: <original> ; <octree> ; <omnitree>

                        buffer = io.StringIO()
                        df_combined.to_csv(
                            buffer,
                            sep=";",
                            header=False,
                            index=False,
                        )

                        buffer.seek(0)
                        lines = buffer.readlines()
                        prefixed_lines = [":: " + line for line in lines]
                        with open(
                            filename_img.replace(tree + "_", "")
                            + "_combined_timeline.txt",
                            "w",
                        ) as f:
                            f.writelines(prefixed_lines)
                break
