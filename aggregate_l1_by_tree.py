#!/usr/bin/env python
import argparse as arg
import pandas as pd
from icecream import ic
from num2words import num2words
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

from thingies_with_omnitrees_evaluate import ErrorL1File, shannon_information


def aggregate_l1_by_tree(num_sobol_samples: int, plot: bool = False):
    error_file = ErrorL1File(num_sobol_samples)

    df = pd.read_csv(error_file.l1fileName)
    num_dimensions = 3
    df["occupancy_ratio"] = df["num_boxes_occupied"] / df["num_boxes"]
    df["occupancy_ratio_diff_half"] = abs(df["occupancy_ratio"] - 0.5)
    df["tree_ones_ratio"] = df["tree_number_of_1s"] / (
        num_dimensions * df["num_tree_nodes"]
    )
    df["tree_ones_ratio_diff_half"] = abs(df["tree_ones_ratio"] - 0.5)
    df["shannon_information_function"] = df["occupancy_ratio"].apply(
        shannon_information
    )
    df["shannon_information_tree"] = df["tree_ones_ratio"].apply(shannon_information)
    label_length = df["tree"].apply(lambda n: 1 if n == "octree" else num_dimensions)
    df["total_storage_size"] = label_length * df["num_tree_nodes"] + df["num_boxes"]

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

            for interesting_allowed_tree_boxes in df_tree[
                "allowed_tree_boxes"
            ].unique():
                # select only a subset of allowed_tree_boxes
                df_scatter_tree = df_tree[
                    df_tree["allowed_tree_boxes"].isin([interesting_allowed_tree_boxes])
                ]

                x = "occupancy_ratio_diff_half"
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
                mean_x = df_scatter_tree[x].mean()
                mean_y = df_scatter_tree[y].mean()
                # Mark the median
                plt.scatter(
                    median_x,
                    median_y,
                    # mean_x,
                    # mean_y,
                    color="red",
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
        df_tree_grouped_storage = (
            df_tree.groupby("allowed_tree_boxes")["total_storage_size"]
            .apply(list)
            .apply(pd.Series)
        )
        df_tree_grouped_storage = df_tree_grouped_storage.transpose()
        df_tree_grouped_storage.columns = [
            num2words(i).replace(" ", "").replace("-", "").replace(",", "") + "_storage"
            for i in df_tree_grouped_storage.columns
        ]
        df_tree_grouped = pd.concat([df_tree_grouped, df_tree_grouped_storage], axis=1)
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
        df_tree_info_medians = (
            df_tree.groupby("allowed_tree_boxes")["occupancy_ratio_diff_half"]
            .agg(
                [
                    "mean",
                    "std",
                    "median",
                ]
            )
            .reset_index()
        )
        df_tree_info_medians.columns = [
            "allowed_tree_boxes",
            "mean_occupancy_ratio_diff_half",
            "std_occupancy_ratio_diff_half",
            "median_occupancy_ratio_diff_half",
        ]
        # drop the first column
        df_tree_info_medians = df_tree_info_medians.drop(columns=["allowed_tree_boxes"])

        df_tree_storage = (
            df_tree.groupby("allowed_tree_boxes")["total_storage_size"]
            .agg(
                [
                    "mean",
                    "std",
                    "median",
                ]
            )
            .reset_index()
        )
        df_tree_storage.columns = [
            "allowed_tree_boxes",
            "mean_total_storage_size",
            "std_total_storage_size",
            "median_total_storage_size",
        ]
        df_tree_storage = df_tree_storage.drop(columns=["allowed_tree_boxes"])

        df_tree_statistics = pd.concat(
            [df_tree_statistics, df_tree_info_medians, df_tree_storage], axis=1
        )
        ic(tree_name, df_tree_statistics)
        df_tree_statistics.to_csv(
            f"l1_error_{tree_name}_s{num_sobol_samples}.csv", index=False
        )
    if plot:
        ax.legend()
        plt.show()

    # filter by allowed_tree_boxes == 8192,  group by thingi_file_id
    df_grouped = (
        df[df["allowed_tree_boxes"] == 8192]
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
    df_octree_better = df_grouped[(df_grouped[0] < df_grouped[1])]
    ic(len(df_octree_better), df_octree_better)


def aggregate_l1_by_tree_and_id(num_sobol_samples, plot: bool = False):
    # from multiple subfolders:
    # (head -n 1 folder1/l1_errors_s512.csv && tail -n +2 -q folder*/l1_errors_s512.csv) > l1_errors_s512.csv
    error_file = ErrorL1File(num_sobol_samples)
    df = pd.read_csv(error_file.l1fileName)
    # drop num_*_samples columns
    df = df.drop(
        columns=[
            "num_sobol_samples",
            "num_error_samples",
            "num_occupancy_samples",
        ]
    )
    fig, ax = plt.subplots()
    ax.set_xlabel("abs(occupancy_ratio - 0.5)")
    ax.set_ylabel("l1 error")
    colors = {
        "octree": "blue",
        "omnitree_1": "orange",
    }
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
    if plot:
        for tree_name in df_grouped.index.levels[0]:
            for interesting_allowed_tree_boxes in df_grouped.index.levels[2]:
                # select only a subset of allowed_tree_boxes
                df_scatter_tree = df[
                    df["allowed_tree_boxes"].isin([interesting_allowed_tree_boxes])
                ]
                df_scatter_tree = df_scatter_tree[df_scatter_tree["tree"] == tree_name]
                # # filter out the cube too
                # df_scatter_tree = df_scatter_tree[
                #     df_scatter_tree["thingi_file_id"] != 187279
                # ]
                sns.scatterplot(
                    x=abs(df_scatter_tree["occupancy_ratio"] - 0.5),
                    # x=abs(df_scatter_tree["tree_ones_ratio"]),
                    y=df_scatter_tree["l1error"],
                    color=colors[tree_name],
                    ax=ax,
                    label=tree_name + " " + str(interesting_allowed_tree_boxes),
                    # set marker size according to allowed_tree_boxes
                    s=math.sqrt(df_scatter_tree["allowed_tree_boxes"]) / 10,
                    alpha=0.5,
                    marker="o",
                )
        ax.legend()
        sns.pairplot(df)
        plt.show()

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
        aggregate_l1_by_tree_and_id(args.sobol_samples, plot=args.plot)
    else:
        aggregate_l1_by_tree(args.sobol_samples, plot=args.plot)
