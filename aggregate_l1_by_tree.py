#!/usr/bin/env python
import argparse as arg
import pandas as pd
from icecream import ic
from num2words import num2words
import math
import seaborn as sns
import matplotlib.pyplot as plt

from thingies_with_omnitrees_evaluate import ErrorL1File


def aggregate_l1_by_tree(num_sobol_samples: int, plot: bool = False):
    error_file = ErrorL1File(num_sobol_samples)

    df = pd.read_csv(error_file.l1fileName)
    tree_names = ["octree", "omnitree_1"]
    # create pyplot figure to fill with scatterplot
    fig, ax = plt.subplots()
    colors = {
        "octree": "blue",
        "omnitree_1": "orange",
    }

    # filter by tree name
    for tree_name in tree_names:
        df_tree = df[df["tree"] == tree_name]
        # if empty, warn and continue
        if df_tree.empty:
            print(f"Warning: no data for {tree_name}")
            continue
        ic(df_tree.describe())
        # get the unique values of allowed_tree_boxes
        num_allowed_box_variants = len(df_tree["allowed_tree_boxes"].unique())
        # print the number of duplicates for this tree
        values_per_thingi_file_id = df_tree["thingi_file_id"].value_counts()
        ic(
            tree_name,
            num_allowed_box_variants,
            values_per_thingi_file_id[
                values_per_thingi_file_id != num_allowed_box_variants
            ],
        )
        if len(values_per_thingi_file_id) != 4166:
            print(
                f"Warning: {tree_name} has only {len(values_per_thingi_file_id)} different thingi_file_ids, expected 4166"
            )
        if plot:
            df_tree["occupancy_ratio"] = (
                df_tree["num_boxes_occupied"] / df_tree["num_boxes"]
            )
            df_tree["occupancy_ratio_diff_half"] = abs(df_tree["occupancy_ratio"] - 0.5)
            df_tree["tree_ones_ratio"] = df_tree["tree_number_of_1s"] / (
                3 * df_tree["num_tree_nodes"]
            )
            df_tree["tree_ones_ratio_diff_half"] = abs(df_tree["tree_ones_ratio"] - 0.5)
            
            label_length = 1 if tree_name == "octree" else 3
            df_tree["total_storage_size"] = (
                label_length * df_tree["num_tree_nodes"] + df_tree["num_boxes"]
            )

            for interesting_allowed_tree_boxes in df_tree["allowed_tree_boxes"].unique():
                # select only a subset of allowed_tree_boxes
                df_scatter_tree = df_tree[
                    df_tree["allowed_tree_boxes"].isin([interesting_allowed_tree_boxes])
                ]

                x = "total_storage_size"
                y = "l1error"
                ax.set_xlabel(x)
                ax.set_ylabel("L1 error")
                ax.set_xscale("log")
                ax.set_yscale("log")
                sns.scatterplot(
                    x=df_scatter_tree[x],
                    y=df_scatter_tree[y],
                    color=colors[tree_name],
                    ax=ax,
                    label=tree_name + " " + str(interesting_allowed_tree_boxes),
                    # set marker size according to allowed_tree_boxes
                    marker="o",
                    s=math.sqrt(interesting_allowed_tree_boxes) / 20,
                )
                # Compute medians
                median_x = df_scatter_tree[x].median()
                median_y = df_scatter_tree[y].median()
                # Mark the median
                plt.scatter(
                    median_x,
                    median_y,
                    color=colors[tree_name],
                    marker="X",
                    s=math.sqrt(interesting_allowed_tree_boxes) / 3,
                )

        # group by allowed_tree_boxes
        df_tree_grouped = (
            df_tree.groupby("allowed_tree_boxes")["l1error"]
            .apply(list)
            .apply(pd.Series)
        )
        df_tree_grouped = df_tree_grouped.transpose()
        ic(tree_name, df_tree_grouped)

        # sns.violinplot(
        #     x="allowed_tree_boxes", y="l1error", data=df_tree, cut=0, split=True
        # )  # , hue='tree'
        # plt.show()

        # rename columns from int values to words
        df_tree_grouped.columns = [
            num2words(i).replace(" ", "").replace("-", "").replace(",", "")
            for i in df_tree_grouped.columns
        ]
        df_tree_grouped.to_csv(
            f"l1_errors_{tree_name}_s{num_sobol_samples}.csv", index=False, na_rep="nan"
        )
        # get the statistics of l1error
        df_tree_statistics = (
            df_tree.groupby("allowed_tree_boxes")["l1error"]
            .agg(
                [
                    "mean",
                    "std",
                    "count",
                    "median",
                    lambda x: x.quantile(0.25),
                    lambda x: x.quantile(0.75),
                    "min",
                    "max",
                ]
            )
            .reset_index()
        )
        df_tree_statistics.columns = [
            "allowed_tree_boxes",
            "mean",
            "std",
            "count",
            "median",
            "q25",
            "q75",
            "min",
            "max",
        ]
        ic(tree_name, df_tree_statistics)
        df_tree_statistics.to_csv(
            f"l1_error_{tree_name}_s{num_sobol_samples}.csv", index=False
        )
    if plot:
        ax.legend()
        plt.show()

    # filter by allowed_tree_boxes == 2048,  group by thingi_file_id
    df_grouped = (
        df[df["allowed_tree_boxes"] == 2048]
        .groupby("thingi_file_id")["l1error"]
        .apply(list)
        .apply(pd.Series)
    )
    df_grouped = df_grouped.rename(
        columns={
            "0": "octree",
            "1": "omnitree_1",
        }
    )
    ic(df_grouped)
    # output thingi_file_id for each entry where the octree can perfectly represent the function
    df_octree_perfect = df_grouped[(df_grouped[0] == 0)]
    ic(
        len(df_octree_perfect), df_octree_perfect
    )  # 56094, 59767, 69058, 187279, 233199, 249519


def aggregate_l1_by_tree_and_id(num_sobol_samples):
    # from multiple subfolders:
    # (head -n 1 folder1/l1_errors_s512.csv && tail -n +2 -q folder*/l1_errors_s512.csv) > l1_errors_s512.csv
    error_file = ErrorL1File(num_sobol_samples)
    df = pd.read_csv(error_file.l1fileName)
    # add a column with the occupancy ratio
    df["occupancy_ratio"] = df["num_boxes_occupied"] / df["num_boxes"]
    # and the number of ones in the descriptor
    df["tree_ones_ratio"] = df["tree_number_of_1s"] / df["num_tree_nodes"]
    # group by tree, thingi_file_id and allowed_tree_boxes
    # for each group, get mean, min, max, and median
    df_grouped = df.groupby(["tree", "thingi_file_id", "allowed_tree_boxes"]).agg(
        {
            "l1error": ["mean", "min", "max", "count", "median"],
            "occupancy_ratio": ["mean", "min", "max", "median"],
        }
    )
    # flatten and save to csv
    df_grouped.columns = [
        "_".join(map(str, col)).strip() for col in df_grouped.columns.values
    ]
    df_grouped = df_grouped.reset_index()
    ic(df_grouped)
    # save to csv
    df_grouped.to_csv(f"l1_errors_s{num_sobol_samples}_by_id.csv", index=False)


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument(
        "--sobol_samples",
        type=int,
        help="number of samples used for the Sobol criterion",
        default=512,
    )
    # optional bool argument for special thingies
    parser.add_argument(
        "--by_id",
        action="store_true",
        help="don't aggregate by tree and allowed_tree_boxes only, but also by thingi_file_id",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="sns scatterplots of the data",
    )
    args = parser.parse_args()
    if args.by_id:
        aggregate_l1_by_tree_and_id(args.sobol_samples)
    else:
        aggregate_l1_by_tree(args.sobol_samples, plot=args.plot)
