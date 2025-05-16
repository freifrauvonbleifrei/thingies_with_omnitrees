#!/usr/bin/env python
import argparse as arg
import pandas as pd
from icecream import ic
from num2words import num2words
import math

from thingies_with_omnitrees_evaluate import ErrorL1File


def aggregate_l1_by_tree(num_sobol_samples):
    error_file = ErrorL1File(num_sobol_samples)

    df = pd.read_csv(error_file.l1fileName)
    tree_names = ["octree", "omnitree_1", "omnitree_2", "omnitree_3"]

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
            (values_per_thingi_file_id != num_allowed_box_variants).sum(),
        )
        if len(values_per_thingi_file_id) != 4166:
            print(
                f"Warning: {tree_name} has only {len(values_per_thingi_file_id)} different thingi_file_ids, expected 4166"
            )
        # group by allowed_tree_boxes
        df_tree_grouped = (
            df_tree.groupby("allowed_tree_boxes")["l1error"]
            .apply(list)
            .apply(pd.Series)
        )
        df_tree_grouped = df_tree_grouped.transpose()
        ic(tree_name, df_tree_grouped)

        # rename columns from int values to words
        df_tree_grouped.columns = [
            num2words(i).replace(" ", "").replace("-", "")
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
            "2": "omnitree_2",
            "3": "omnitree_3",
        }
    )
    ic(df_grouped)
    # output thingi_file_id for each entry where the octree can perfectly represent the function
    df_octree_perfect = df_grouped[(df_grouped[0] == 0)]
    ic(len(df_octree_perfect), df_octree_perfect)

    # output thingi_file_id for each entry where the octree has lower error than omnitree_2 and omnitree_3
    df_octree_better = df_grouped[(df_grouped[0] < df_grouped[2])]
    df_octree_better = df_octree_better[(df_octree_better[0] < df_octree_better[3])]
    ic(len(df_octree_better), df_octree_better)

    # output thingi_file_id for each entry where the omnitree_2 has lower error than omnitree_3
    df_omnitree_2_better = df_grouped[(df_grouped[2] < df_grouped[3])]
    ic(len(df_omnitree_2_better), df_omnitree_2_better)
    # maxloc of difference between omnitree_2 and omnitree_3
    df_omnitree_2_better["diff"] = df_omnitree_2_better[3] - df_omnitree_2_better[2]
    df_omnitree_2_best = df_omnitree_2_better[
        df_omnitree_2_better["diff"] == df_omnitree_2_better["diff"].max()
    ]
    ic(df_omnitree_2_best)
    # output thingi_file_id for the entry where the omnitree_3 has the best improvement in error compared to the omnitree_2
    df_omnitree_3_better = df_grouped[(df_grouped[3] < df_grouped[2])]
    # maxloc of difference between omnitree_3 and omnitree_2
    df_omnitree_3_better["diff"] = df_omnitree_3_better[2] - df_omnitree_3_better[3]
    ic(len(df_omnitree_3_better))
    df_omnitree_3_better = df_omnitree_3_better[
        df_omnitree_3_better["diff"] == df_omnitree_3_better["diff"].max()
    ]
    ic(df_omnitree_3_better)


if __name__ == "__main__":
    parser = arg.ArgumentParser()
    parser.add_argument(
        "--sobol_samples",
        type=int,
        help="number of samples used for the Sobol criterion",
        default=2048,
    )
    args = parser.parse_args()
    aggregate_l1_by_tree(args.sobol_samples)
