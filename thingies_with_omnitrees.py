#!/usr/bin/env python
import argparse as arg
import bitarray as ba
import functools
import numpy as np
from icecream import ic
from queue import PriorityQueue
from SALib.sample import sobol, saltelli
from SALib.analyze import sobol
import thingi10k
import trimesh

import dyada
import dyada.coordinates
import dyada.linearization
import dyada.refinement

from thingies_utils import mesh_to_unit_cube, check_inside_or_outside_mesh


@functools.cache
def get_unit_cube_sa_problem(num_sobol_samples: int):
    # cf. https://salib.readthedocs.io/en/latest/user_guide/basics.html#an-example
    problem = {
        "num_vars": 3,
        "names": ["x0", "x1", "x2"],
        "bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    }
    param_values = saltelli.sample(problem, num_sobol_samples)
    return problem, param_values


def get_sobol_importances(
    mesh: trimesh.Geometry,
    interval: dyada.coordinates.CoordinateInterval,
    refinements: list[ba.bitarray],
    num_sobol_samples: int,
):
    problem, sampling_points_unit_cube = get_unit_cube_sa_problem(num_sobol_samples)
    # transform the sampling points to the current interval
    extent = interval.upper_bound - interval.lower_bound
    sampling_points_transformed = (
        sampling_points_unit_cube * extent + interval.lower_bound
    )
    # evaluate if points are inside or outside the mesh
    is_inside = check_inside_or_outside_mesh(mesh, sampling_points_transformed)
    total_variance = np.var(is_inside)

    # multiply by total variance and interval volume
    scaling_factor = total_variance * np.prod(
        interval.upper_bound - interval.lower_bound
    )
    if refinements == [ba.bitarray("111")]:
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
            if refinement == ba.bitarray("111"):
                importance = 1.0
            elif refinement == ba.bitarray("100"):
                importance = Si["S1"][0]
            elif refinement == ba.bitarray("010"):
                importance = Si["S1"][1]
            elif refinement == ba.bitarray("001"):
                importance = Si["S1"][2]
            elif refinement == ba.bitarray("110"):
                importance = Si["S2"][0, 1]
            elif refinement == ba.bitarray("011"):
                importance = Si["S2"][1, 2]
            elif refinement == ba.bitarray("101"):
                importance = Si["S2"][0, 2]
            else:
                raise ValueError
            importances.append(importance * scaling_factor)
    return importances


def skip_function_no_importance(
    mesh: trimesh.Geometry,
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
    mesh: trimesh.Geometry,
    importance_function,
    skip_function,
    allowed_refinements=[ba.bitarray("111")],
):
    interval = dyada.refinement.coordinates_from_box_index(discretization, box_index)

    importances = importance_function(mesh, interval, allowed_refinements)
    for refinement, importance in zip(allowed_refinements, importances):
        # skip if condition is met, e.g. if there is already only one point in the partition
        if skip_function is None or not skip_function(mesh, interval, importance):
            priority_queue.put((-importance, refinement, box_index))


def get_initial_priority_queue(
    discretization: dyada.refinement.Discretization,
    mesh: trimesh.Geometry,
    importance_function,
    allowed_refinements=[ba.bitarray("111")],
) -> PriorityQueue:
    # initialize the priority queue
    priority_queue = PriorityQueue()

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
    mesh: trimesh.Geometry, importance_function, allowed_refinements
):
    discretization = dyada.refinement.Discretization(
        dyada.linearization.MortonOrderLinearization(),
        dyada.refinement.RefinementDescriptor(3, 1),
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
    mesh: trimesh.Geometry,
    max_num_boxes: int,
    importance_function,
    skip_function,
    allowed_refinements,
) -> tuple[dyada.refinement.Discretization, PriorityQueue]:

    # grow the tree until the desired number of boxes is reached
    while len(discretization) < max_num_boxes and not priority_queue.empty():
        _, refinement, next_refinement_index = priority_queue.get()
        discretization, index_mapping = dyada.refinement.apply_single_refinement(
            discretization, next_refinement_index, refinement
        )
        # update the priority queue's old entries with the (potentially) changed indices
        new_priority_queue = PriorityQueue()
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
        default=64,
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
