#!/usr/bin/env python
import bitarray as ba
import functools
import numpy as np
import math
from icecream import ic
from queue import PriorityQueue
from SALib.sample import sobol, saltelli
from SALib.analyze import sobol
import thingi10k
import trimesh

import dyada
import dyada.coordinates
import dyada.drawing
import dyada.linearization
import dyada.refinement


def check_inside_or_outside_mesh(
    mesh: trimesh.Geometry, points: np.ndarray
) -> np.array:
    is_inside = mesh.contains(points)
    return is_inside


def check_inside_or_outside_tree(
    discretization: dyada.refinement.Discretization,
    binary_discretization_occupancy: np.ndarray,
    points: np.ndarray,
) -> np.array:
    is_inside = np.zeros(len(points), dtype=bool)
    for p_i, point in enumerate(points):
        box_index = discretization.get_containing_box(point)
        if binary_discretization_occupancy[box_index]:
            is_inside[p_i] = True
    return is_inside


def get_monte_carlo_l1_error(
    mesh: trimesh.Geometry,
    discretization: dyada.refinement.Discretization,
    binary_discretization_occupancy: np.ndarray,
) -> float:
    # generate 10000 random points in the unit cube
    points = np.random.rand(10000, 3)

    is_inside_mesh = check_inside_or_outside_mesh(mesh, points)
    is_inside_tree = check_inside_or_outside_tree(
        discretization, binary_discretization_occupancy, points
    )

    # calculate the L1 error
    return (is_inside_mesh ^ is_inside_tree).mean()


def get_number_of_mesh_points_inside_interval(
    mesh: trimesh.Geometry, interval: dyada.coordinates.CoordinateInterval
) -> int:
    # filter mesh points that are inside the interval
    mesh_points = mesh.vertices[
        (mesh.vertices >= interval.lower_bound).all(axis=1)
        & (mesh.vertices <= interval.upper_bound).all(axis=1)
    ]
    return len(mesh_points)


@functools.cache
def get_unit_cube_sa_problem():
    # cf. https://salib.readthedocs.io/en/latest/user_guide/basics.html#an-example
    problem = {
        "num_vars": 3,
        "names": ["x0", "x1", "x2"],
        "bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    }
    param_values = saltelli.sample(problem, 32)
    return problem, param_values


def get_sobol_importances(
    mesh: trimesh.Geometry,
    interval: dyada.coordinates.CoordinateInterval,
    refinements: list[ba.bitarray],
):
    problem, sampling_points_unit_cube = get_unit_cube_sa_problem()
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
        ic(Si)
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


def tree_voxel_thingi(
    mesh,
    max_num_boxes,
    importance_function,
    skip_function,
    allowed_refinements=[ba.bitarray("111")],
) -> tuple[dyada.refinement.Discretization, np.array]:
    discretization = dyada.refinement.Discretization(
        dyada.linearization.MortonOrderLinearization(),
        dyada.refinement.RefinementDescriptor(3, 1),
    )
    priority_queue: PriorityQueue = get_initial_priority_queue(
        discretization,
        mesh,
        importance_function,
    )

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
    return discretization


def get_binary_discretization_occupancy(
    discretization: dyada.refinement.Discretization, mesh
):
    binary_discretization_occupancy = np.zeros(len(discretization), dtype=bool)
    for box_index in range(len(discretization)):
        interval = dyada.refinement.coordinates_from_box_index(
            discretization, box_index
        )
        # get random points in the interval
        random_points_in_interval = (
            np.random.rand(256, 3) * (interval.upper_bound - interval.lower_bound)
            + interval.lower_bound
        )
        if np.mean(check_inside_or_outside_mesh(mesh, random_points_in_interval)) > 0.5:
            binary_discretization_occupancy[box_index] = True
    return binary_discretization_occupancy


def mesh_to_unit_cube(mesh: trimesh.Geometry) -> trimesh.Geometry:
    # scale the mesh to fit in the unit cube (0, 1)^3
    mesh.apply_scale(1.0 / mesh.extents.max())

    # center the mesh
    bounds_mean = mesh.bounds.mean(axis=(0))
    mesh.apply_translation(-bounds_mean + 0.5)

    # ic(mesh.extents.max(), mesh.extents.min())
    assert np.isclose(mesh.bounds.max(), 1.0)
    assert np.isclose(mesh.bounds.min(), 0.0)
    assert np.isclose(mesh.extents.max(), 1.0)
    return mesh


def get_mesh_from_discretization(discretization, binary_discretization_occupancy):
    # construct a mesh from the discretization
    list_of_boxes = []
    for box_index in range(len(discretization)):
        if binary_discretization_occupancy[box_index]:
            interval = dyada.refinement.coordinates_from_box_index(
                discretization, box_index
            )
            # box = trimesh.creation.box(bounds=interval)
            box = trimesh.creation.box(
                extents=interval.upper_bound - interval.lower_bound,
                transform=trimesh.transformations.translation_matrix(
                    interval.lower_bound
                    + (interval.upper_bound - interval.lower_bound) / 2
                ),
            )
            list_of_boxes.append(box)
    return trimesh.util.concatenate(list_of_boxes)


if __name__ == "__main__":
    thingi10k.init(variant="raw")
    # select thingi meshes by closedness, having at most 10000 vertices, etc.
    # print(help(thingi10k.dataset))
    subset = thingi10k.dataset(
        num_vertices=(None, 10000),
        closed=True,
        self_intersecting=False,
        solid=True,
    )
    print(len(subset))
    print(
        len(
            thingi10k.dataset(
                num_vertices=(None, 10000),
                closed=True,
                self_intersecting=False,
                solid=True,
            )
        )
    )

    allowed_tree_boxes = 512
    # randomly sample three thingies and display them
    for thingi in np.random.choice(
        subset,
        1,
    ):  # interesting IDs:
        print(thingi)
        mesh = trimesh.load_mesh(thingi["file_path"], file_type="stl")
        if not mesh.is_watertight:
            continue
        mesh = mesh_to_unit_cube(mesh)
        discretization_octree = tree_voxel_thingi(
            mesh,
            allowed_tree_boxes,
            get_sobol_importances,
            skip_function_no_importance,
        )
        binary_discretization_occupancy_octree = get_binary_discretization_occupancy(
            discretization_octree, mesh
        )
        ic(
            get_monte_carlo_l1_error(
                mesh, discretization_octree, binary_discretization_occupancy_octree
            )
        )
        assert len(binary_discretization_occupancy_octree) == len(discretization_octree)

        discretization_omnitree = tree_voxel_thingi(
            mesh,
            allowed_tree_boxes,
            get_sobol_importances,
            skip_function_no_importance,
            # None,  # skip_function_previously_no_importance,
            allowed_refinements=[
                ba.bitarray("100"),
                ba.bitarray("010"),
                ba.bitarray("001"),
            ],
        )
        binary_discretization_occupancy_omnitree = get_binary_discretization_occupancy(
            discretization_omnitree, mesh
        )
        ic(
            get_monte_carlo_l1_error(
                mesh, discretization_omnitree, binary_discretization_occupancy_omnitree
            )
        )
        assert len(binary_discretization_occupancy_omnitree) == len(
            discretization_omnitree
        )

        mesh.show()
        # construct a mesh from the discretization
        mesh_from_octree = get_mesh_from_discretization(
            discretization_octree, binary_discretization_occupancy_octree
        )
        ic(np.sum(binary_discretization_occupancy_octree))
        dyada.drawing.plot_all_boxes_3d(discretization_octree, labels=None)
        ic(mesh_from_octree)
        if not mesh_from_octree.is_empty:
            mesh_from_octree.show()

        mesh_from_omnitree = get_mesh_from_discretization(
            discretization_omnitree, binary_discretization_occupancy_omnitree
        )
        ic(np.sum(binary_discretization_occupancy_omnitree))
        dyada.drawing.plot_all_boxes_3d(discretization_omnitree, labels=None)
        ic(mesh_from_omnitree)
        if not mesh_from_omnitree.is_empty:
            mesh_from_omnitree.show()
