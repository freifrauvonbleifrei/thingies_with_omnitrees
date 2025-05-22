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
        help="input image extension,xeither 'png' or 'svg'",
        choices=["png", "svg"],
        default="svg",
    )
    args = parser.parse_args()

    # find all svg files in the current directory and group them by numerical prefix
    input_paths = [
        f
        for f in os.listdir(".")
        if os.path.isfile(f) and f.endswith(args.img_extension)
    ]
    ic(input_paths)
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

    for thingi_id, img_file_list in img_files.items():
        thingi_img_files = []
        # extract "_orginal." path
        original = [f for f in img_file_list if "_original." + args.img_extension in f]
        assert ( #TODO
            len(original) == 1
        ), f"expected 1 '_original.{args.img_extension}', got {len(original)}"
        # remove it from the list
        img_file_list.remove(original[0])

        # example filename: 100349_omnitree_3_64_s256_eval.svg
        nums_boxes: dict[int, list[str]] = {}
        for path in img_file_list:
            num_boxes = int(path.split("_")[-3])
            if num_boxes not in nums_boxes:
                nums_boxes[num_boxes] = []
            nums_boxes[num_boxes].append(path)
        # sort nums_boxes by key
        nums_boxes = dict(sorted(nums_boxes.items()))

        for num_boxes, paths in nums_boxes.items():
            ic(paths)
            if len(paths) != 4:
                print(
                    f"Warning: expected 4 SVG files for id {thingi_id}, {num_boxes} boxes, got {len(paths)}"
                )
                continue
            paths = [
                [f for f in paths if "_octree_" in f][0],
                [f for f in paths if "_omnitree_1_" in f][0],
                [f for f in paths if "_omnitree_2_" in f][0],
                [f for f in paths if "_omnitree_3_" in f][0],
            ]
            if args.img_extension == "svg":
                img_original = SVG(original[0])
                img_octree = SVG(paths[0])
                img_omnitree_1 = SVG(paths[1])
                img_omnitree_2 = SVG(paths[2])
                img_omnitree_3 = SVG(paths[3])

                original_width, original_height = (
                    img_original.width,
                    img_original.height,
                )
                octree_width, octree_height = img_octree.width, img_octree.height
            else:
                width = 1024
                height = 1024
                # create a new SVG figure and embed the PNG images
                with open(original[0], "rb") as original_img_file:
                    img_original = ImageElement(original_img_file, width, height)
                with open(paths[0], "rb") as octree_img_file:
                    img_octree = ImageElement(octree_img_file, width, height)
                with open(paths[1], "rb") as omni_1_img_file:
                    img_omnitree_1 = ImageElement(omni_1_img_file, width, height)
                with open(paths[2], "rb") as omni_2_img_file:
                    img_omnitree_2 = ImageElement(omni_2_img_file, width, height)
                with open(paths[3], "rb") as omni_3_img_file:
                    img_omnitree_3 = ImageElement(omni_3_img_file, width, height)
                original_width, original_height = width, height
                octree_width, octree_height = width, height

            ic(original_width, original_height)
            ic(octree_width, octree_height)

            # move original down a bit
            original_down_shift = 0.25 * octree_height
            img_original.moveto(0, original_down_shift)

            # move all except original right and!
            right_offset = original_width * 0.9
            more_right_offset = right_offset + octree_width
            # octree and omnitree_1 down, omnitree_1 and omnitree_3 right
            img_octree.moveto(right_offset, octree_height)
            img_omnitree_1.moveto(more_right_offset, octree_height)
            img_omnitree_2.moveto(right_offset, 0)
            img_omnitree_3.moveto(more_right_offset, 0)

            combined_width = original_width + 2 * octree_width
            combined_height = 2 * octree_height

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
                "size": 18,
                "weight": "bold",
                "color": "black",
            }
            label_shift_up = 0.15 * octree_height

            combined = Figure(
                combined_width,
                combined_height,
                background,
                img_original,
                Text(
                    f"Thingi {thingi_id}",
                    original_width * 0.25,
                    original_down_shift + original_height,
                    size=36,
                    weight="bold",
                    color="black",
                ),
                img_octree,
                Text(
                    "Octree",
                    right_offset + octree_width * 0.25,
                    octree_height * 2 - label_shift_up,
                    **tree_label_style_dict,
                ),
                img_omnitree_1,
                Text(
                    "Omnitree 1",
                    more_right_offset + octree_width * 0.25,
                    octree_height * 2 - label_shift_up,
                    **tree_label_style_dict,
                ),
                img_omnitree_2,
                Text(
                    "Omnitree 2",
                    right_offset + octree_width * 0.25,
                    octree_height - label_shift_up,
                    **tree_label_style_dict,
                ),
                img_omnitree_3,
                Text(
                    "Omnitree 3",
                    more_right_offset + octree_width * 0.25,
                    octree_height - label_shift_up,
                    **tree_label_style_dict,
                ),
            )
            # Todo consider putting the errors here as well
            # save file
            filename = f"{thingi_id}_{num_boxes}_combined.svg"
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

        # create gif
        subprocess.run(
            ["convert"]
            + input_file_arg_list
            + [f"{thingi_id}_{args.img_extension}_combined.gif"]
        )

        # remove the png files
        for file in thingi_svg_files:
            subprocess.run(["rm", f"{file}.png"])

        # remove the svg files
        for file in thingi_svg_files:
            subprocess.run(["rm", file])
