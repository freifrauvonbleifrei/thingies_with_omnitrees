#!/usr/bin/env python
import argparse as arg
import bitarray as ba
import functools
import numpy as np
from icecream import ic
import os
from queue import PriorityQueue
from SALib.sample import sobol, saltelli
from SALib.analyze import sobol
import thingi10k
import trimesh

import dyada
import dyada.coordinates
import dyada.linearization
import dyada.refinement

from thingies_utils import (
    mesh_to_unit_cube,
    check_inside_or_outside_mesh,
    check_inside_or_outside_mesh_temporal,
)


@functools.cache
def get_unit_cube_sa_problem(num_sobol_samples: int, num_dimensions: int = 3):
    # cf. https://salib.readthedocs.io/en/latest/user_guide/basics.html#an-example
    problem = {
        "num_vars": num_dimensions,
        "names": [f"x{i}" for i in range(num_dimensions)],
        "bounds": [[0.0, 1.0]] * num_dimensions,
    }
    param_values = saltelli.sample(problem, num_sobol_samples)
    return problem, param_values


def get_sobol_importances(
    mesh: trimesh.Trimesh,
    interval: dyada.coordinates.CoordinateInterval,
    refinements: list[ba.bitarray],
    num_sobol_samples: int,
):
    num_dimensions = len(interval.upper_bound)
    problem, sampling_points_unit_cube = get_unit_cube_sa_problem(
        num_sobol_samples, num_dimensions
    )
    # transform the sampling points to the current interval
    extent = interval.upper_bound - interval.lower_bound
    sampling_points_transformed = (
        sampling_points_unit_cube * extent + interval.lower_bound
    )
    # evaluate if points are inside or outside the mesh
    if num_dimensions == 3:
        is_inside = check_inside_or_outside_mesh(mesh, sampling_points_transformed)
    elif num_dimensions == 4:
        is_inside = check_inside_or_outside_mesh_temporal(
            mesh, sampling_points_transformed
        )
    total_variance = np.var(is_inside)

    # multiply by total variance and interval volume
    scaling_factor = total_variance * np.prod(
        interval.upper_bound - interval.lower_bound
    )
    if len(refinements) == 1 and len(refinements[0]) == refinements[0].count(1):
        # if only ones (=> octree), return the scaling factor
        return [scaling_factor]
    else:
        if not np.any(is_inside) or np.all(is_inside):
            return [0.0] * len(refinements)
        Si = sobol.analyze(problem, is_inside)
        # ic(Si)
        if Si["S2"][1, 2] == np.nan:
            return [0.0] * len(refinements)

        importances = []
        for refinement in refinements:
            refinement = ba.bitarray(refinement)
            if refinement == ba.bitarray("111"):
                importance = 1.0 - Si["S1"].sum() - Si["S2"].sum()
            elif refinement.count(1) == 1:
                # 1d sensitivity indices -> only
                index_of_1 = refinement.index(1)
                importance = Si["S1"][index_of_1]
            elif refinement.count(1) == 2:
                # 2d sensitivity indices
                index_of_1 = refinement.index(1)
                index_of_2 = refinement.index(1, index_of_1 + 1)
                importance = Si["S2"][index_of_1, index_of_2]
            else:
                raise ValueError("unknown refinement {}".format(refinement))
            importances.append(importance * scaling_factor)
    return importances


def skip_function_no_importance(
    mesh: trimesh.Trimesh,
    interval: dyada.coordinates.CoordinateInterval,
    importance: float,
) -> bool:
    if importance <= 0.0:
        return True
    return False


def put_box_into_priority_queue(
    box_index: int,
    priority_queue: PriorityQueue,
    discretization: dyada.refinement.Discretization,
    mesh: trimesh.Trimesh,
    importance_function,
    skip_function,
    allowed_refinements=[ba.bitarray("111")],
):
    level_index = dyada.refinement.get_level_index_from_linear_index(
        discretization._linearization, discretization._descriptor, box_index
    )
    interval = dyada.refinement.get_coordinates_from_level_index(level_index)

    importances = importance_function(mesh, interval, allowed_refinements)
    for refinement, importance in zip(allowed_refinements, importances):
        # skip if the level in a given dimension would become too large (> 30)
        skip_bc_too_fine = False
        for d in range(len(refinement)):
            if level_index.d_level[d] + refinement[d] > 30:
                skip_bc_too_fine = True
                break
        # skip if condition is met, e.g. too fine or 0 importance
        if not skip_bc_too_fine and (
            skip_function is None or not skip_function(mesh, interval, importance)
        ):
            priority_queue.put((-importance, refinement, box_index))


def get_initial_priority_queue(
    discretization: dyada.refinement.Discretization,
    mesh: trimesh.Trimesh,
    importance_function,
    allowed_refinements=[ba.bitarray("111")],
) -> PriorityQueue:
    # initialize the priority queue
    priority_queue: PriorityQueue = PriorityQueue()

    # calculate the importance of each box, for each allowed refinement
    for box_index in range(len(discretization)):
        put_box_into_priority_queue(
            box_index,
            priority_queue,
            discretization,
            mesh,
            importance_function,
            None,
            allowed_refinements,
        )
    return priority_queue


def get_initial_tree_and_queue(
    mesh: trimesh.Trimesh, importance_function, allowed_refinements
):
    num_dimensions = len(allowed_refinements[0])
    assert all([len(r) == num_dimensions for r in allowed_refinements])
    discretization = dyada.refinement.Discretization(
        dyada.linearization.MortonOrderLinearization(),
        dyada.refinement.RefinementDescriptor(num_dimensions, 0),
    )
    priority_queue: PriorityQueue = get_initial_priority_queue(
        discretization,
        mesh,
        importance_function,
        allowed_refinements,
    )
    return discretization, priority_queue


def tree_voxel_thingi(
    discretization,
    priority_queue,
    mesh: trimesh.Trimesh,
    max_num_boxes: int,
    importance_function,
    skip_function,
    allowed_refinements,
) -> tuple[dyada.refinement.Discretization, PriorityQueue]:

    too_fine = False
    # grow the tree until the desired number of boxes is reached
    while (
        len(discretization) < max_num_boxes
        and not priority_queue.empty()
        and not too_fine
    ):
        _, refinement, next_refinement_index = priority_queue.get()
        discretization, index_mapping = dyada.refinement.apply_single_refinement(
            discretization, next_refinement_index, refinement
        )
        # update the priority queue's old entries with the (potentially) changed indices
        new_priority_queue: PriorityQueue = PriorityQueue()
        while not priority_queue.empty():
            i_neg_importance, i_refinement, i = priority_queue.get()
            if allowed_refinements == [ba.bitarray("111")]:
                assert len(index_mapping[i]) == 1
            elif len(index_mapping[i]) != 1:
                # we're dealing with a box that was refined but in different dimensions
                # -> skip
                continue
            new_index = index_mapping[i][0]
            new_priority_queue.put((i_neg_importance, i_refinement, new_index))
        # calculate new importance for the new patches
        for i in index_mapping[next_refinement_index]:
            try:
                put_box_into_priority_queue(
                    i,
                    new_priority_queue,
                    discretization,
                    mesh,
                    importance_function,
                    skip_function,
                    allowed_refinements,
                )
            except dyada.coordinates.DyadaTooFineError as e:
                # the next boxes would be too fine, break
                # (because everything else will have little impact as well)
                ic(e)
                too_fine = True
                break
            except Exception as e:
                ic(i)
                ic(index_mapping[next_refinement_index])
                ic(len(discretization))
                ic(new_priority_queue.queue)
                raise e
        priority_queue = new_priority_queue
        if priority_queue.empty():
            ic(len(discretization), max_num_boxes, len(mesh.vertices))
            print("Priority queue is empty, stopping.")
            break

    ic(len(discretization))
    return discretization, priority_queue


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
        default=2048,
    )
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

    thingi10k.init()

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
        mesh_data = np.load(thingi["file_path"])
        mesh_vertices = mesh_data["vertices"]
        mesh_faces = mesh_data["facets"]
        mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
        try:
            if not mesh.is_watertight:
                continue
        except IndexError as e:
            ic(mesh.vertices)
            ic(mesh.faces)
            print(e)
            continue
        mesh = mesh_to_unit_cube(mesh)

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

        for allowed_refinements, discretization, queue, tree_name in tree_tuples:
            filename_finest = (
                str(thingi["file_id"])
                + "_"
                + tree_name
                + "_"
                + str(number_tree_boxes[-1])
                + "_s"
                + str(args.sobol_samples)
                + "_3d.bin"
            )
            if os.path.isfile(filename_finest):
                ic(filename_finest, " exists")
                continue
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
                filename = (
                    str(thingi["file_id"])
                    + "_"
                    + tree_name
                    + "_"
                    + str(allowed_tree_boxes)
                    + "_s"
                    + str(args.sobol_samples)
                )
                discretization.descriptor.to_file(filename)
