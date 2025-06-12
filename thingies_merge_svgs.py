#!/usr/bin/env python
import argparse as arg
import cairosvg
from icecream import ic
from itertools import chain
import os.path
from svgutils.compose import Figure, SVG, Text
from svgutils.transform import fromstring, ImageElement
import subprocess


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument(
        "--img_extension",
        type=str,
        help="input image extension, either 'png' or 'svg'",
        choices=["png", "svg"],
        default="svg",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="if present, use the temporal 4d version of the thingies",
    )
    args = parser.parse_args()

    # find all svg files in the current directory and group them by numerical prefix
    input_paths = [
        f
        for f in os.listdir(".")
        if os.path.isfile(f) and f.endswith(args.img_extension)
    ]
    ic(sorted(input_paths))
    img_files: dict[str, list[str]] = {}
    for path in input_paths:
        prefix = path.split("_")[0]
        try:
            int(prefix)
        except ValueError:
            # skip if prefix is not a number
            continue
        if prefix not in img_files:
            img_files[prefix] = []
        img_files[prefix].append(path)

    thingi_names = {
        "0": "Tetrahedron",
        "1": "Sphere",
        "2": "Rod",
        "53750": "Hilbert Cube",
        "96453": "Car",
        "99905": "Gear",
        "100349": "Cat",
        "187279": "Cube",
    }

    for thingi_id, img_file_list in img_files.items():
        thingi_img_files = []
        # extract "_orginal." path
        original = [f for f in img_file_list if "_original" in f]
        if args.temporal:
            assert (
                len(original) == 64
            ), f"expected 64 '_original.{args.img_extension}', got {len(original)}"
        else:
            assert (
                len(original) == 1
            ), f"expected 1 '_original.{args.img_extension}', got {len(original)}"
            # remove from the list
            for o in original:
                img_file_list.remove(o)

        if args.temporal:
            common_allowed_boxes = 65536
            filename_temporal_skeleton_original = "%d_original_t%03d.png"
            filename_temporal_skeleton_octree = "%d_octree_%d_s512_eval_t%03d.png"
            filename_temporal_skeleton_omnitree = "%d_omnitree_1_%d_s512_eval_t%03d.png"
            common_boxes_found = False
            while common_allowed_boxes > 4 and not common_boxes_found:
                common_boxes_found = True
                timeslices: dict[int, list[str]] = {}
                for time in range(64):
                    if time not in timeslices:
                        timeslices[time] = []
                    path_original = filename_temporal_skeleton_original % (
                        int(thingi_id),
                        time,
                    )
                    path_octree = filename_temporal_skeleton_octree % (
                        int(thingi_id),
                        common_allowed_boxes,
                        time,
                    )
                    path_omnitree = filename_temporal_skeleton_omnitree % (
                        int(thingi_id),
                        common_allowed_boxes,
                        time,
                    )
                    if (
                        not os.path.isfile(path_octree)
                        or not os.path.isfile(path_omnitree)
                        or not os.path.isfile(path_original)
                    ):
                        # try again at smaller resolution
                        common_allowed_boxes //= 2
                        common_boxes_found = False
                        break
                    else:
                        timeslices[time].append(path_original)
                        timeslices[time].append(path_octree)
                        timeslices[time].append(path_omnitree)
            ic(common_allowed_boxes, path_octree)
            if common_allowed_boxes < 8:
                print(
                    "no common images found for original and "
                    "omnitree and octree for thingi_id %d" % int(thingi_id)
                )
                continue
            # sort timeslices by key
            timeslices = dict(sorted(timeslices.items()))
            iterable_for_gif = timeslices

        else:
            # example filename: 100349_omnitree_3_64_s256_eval.svg
            nums_boxes: dict[int, list[str]] = {}
            for path in img_file_list:
                ic(path.split("_"))
                num_boxes = int(path.split("_")[-3])
                if num_boxes not in nums_boxes:
                    nums_boxes[num_boxes] = []
                nums_boxes[num_boxes].append(path)
            # sort nums_boxes by key
            nums_boxes = dict(sorted(nums_boxes.items()))
            iterable_for_gif = nums_boxes

        for i_index, paths in iterable_for_gif.items():
            ic(paths)
            if not (len(paths) == 2 and not args.temporal) and not (
                len(paths) == 3 and args.temporal
            ):
                print(
                    f"Warning: expected 2 files for id {thingi_id}, {i_index} "
                    "boxes or time, got {len(paths)}"
                )
                continue
            if args.temporal:
                paths = [
                    [f for f in paths if "_octree_" in f][0],
                    [f for f in paths if "_omnitree_1_" in f][0],
                    [f for f in paths if "original" in f][0],
                ]
            else:
                paths = [
                    [f for f in paths if "_octree_" in f][0],
                    [f for f in paths if "_omnitree_1_" in f][0],
                ]
            if args.img_extension == "svg":
                if args.temporal:
                    img_original = SVG(paths[2])
                else:
                    img_original = SVG(original[0])
                img_octree = SVG(paths[0])
                img_omnitree_1 = SVG(paths[1])

                original_width, original_height = (
                    img_original.width,
                    img_original.height,
                )
                octree_width, octree_height = img_octree.width, img_octree.height
            else:
                width = 1024
                height = 1024
                # create a new SVG figure and embed the PNG images
                original_path = paths[2] if args.temporal else original[0]
                with open(original_path, "rb") as original_img_file:
                    img_original = ImageElement(original_img_file, width, height)
                with open(paths[0], "rb") as octree_img_file:
                    img_octree = ImageElement(octree_img_file, width, height)
                with open(paths[1], "rb") as omni_1_img_file:
                    img_omnitree_1 = ImageElement(omni_1_img_file, width, height)
                original_width, original_height = width, height
                octree_width, octree_height = width, height

            ic(original_width, original_height)
            ic(octree_width, octree_height)

            # move original down a bit
            original_down_shift = 0.0 * octree_height
            img_original.moveto(0, original_down_shift)

            # move all except original right and!
            right_offset = original_width * 0.9
            more_right_offset = right_offset + octree_width
            # octree and omnitree_1 down, omnitree_1 and omnitree_3 right
            img_octree.moveto(right_offset, 0)
            img_omnitree_1.moveto(more_right_offset, 0)

            combined_width = original_width + 2 * octree_width
            combined_height = 1 * octree_height

            background = fromstring(
                f"""<?xml version="1.0" encoding="UTF-8" standalone="no"?>
                    <svg version="1.1">
                        <rect
                            style="fill:#ffffff"
                            width="{combined_width}"
                            height="{combined_height}"
                            x="0"
                            y="0"
                        />
                    </svg>"""
            ).getroot()

            tree_label_style_dict = {
                "size": 30,
                "weight": "bold",
                "color": "black",
            }
            label_shift_up = 0.15 * octree_height

            thingi_label = f"Thingi {thingi_id}"
            if thingi_id in thingi_names:
                if int(thingi_id) < 20:
                    thingi_label = f"{thingi_names[thingi_id]}"
                else:
                    thingi_label = f"{thingi_names[thingi_id]} \n(Thingi {thingi_id})"

            combined = Figure(
                combined_width,
                combined_height,
                background,
                img_original,
                Text(
                    thingi_label,
                    original_width * 0.25,
                    original_down_shift + 0.8 * original_height,
                    size=36,
                    weight="bold",
                    color="black",
                ),
                img_octree,
                Text(
                    "Octree",
                    right_offset + octree_width * 0.25,
                    octree_height * 1 - label_shift_up,
                    **tree_label_style_dict,
                ),
                img_omnitree_1,
                Text(
                    "Omnitree",
                    more_right_offset + octree_width * 0.25,
                    octree_height * 1 - label_shift_up,
                    **tree_label_style_dict,
                ),
            )
            # Todo consider putting the errors here as well
            # save file
            filename = f"{thingi_id}_{i_index}_combined.svg"
            combined.save(filename)

            thingi_img_files.append(filename)

        # convert svg to png
        for file in thingi_img_files:
            ic(file)
            cairosvg.svg2png(url=file, write_to=f"{file}.png")

        input_file_arg_list = list(
            chain.from_iterable(
                [
                    [
                        "-delay",
                        "200" if i == len(thingi_img_files) - 1 else "50",
                        f"{file}.png",
                    ]
                    for i, file in enumerate(thingi_img_files)
                ]
            )
        )
        ic(input_file_arg_list)
        if len(input_file_arg_list) == 0:
            print(
                f"Warning: no images found for thingi {thingi_id}, skipping gif creation"
            )
            continue
        # create gif
        combined_gif_filename = f"{thingi_id}_{args.img_extension}_combined.gif"
        if args.temporal:
            combined_gif_filename = (
                f"{thingi_id}_{common_allowed_boxes}_{args.img_extension}_combined.gif"
            )
        subprocess.run(["convert"] + input_file_arg_list + [combined_gif_filename])

        # remove the png files
        for file in thingi_svg_files:
            subprocess.run(["rm", f"{file}.png"])

        # remove the svg files
        for file in thingi_svg_files:
            subprocess.run(["rm", file])
