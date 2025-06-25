from copy import deepcopy
import gc
import numpy as np
import numpy.typing as npt
import trimesh


def mesh_to_unit_cube(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    # scale the mesh to fit in the unit cube (0, 1)^3
    mesh.apply_scale(1.0 / mesh.extents.max())

    # center the mesh
    bounds_mean = mesh.bounds.mean(axis=(0))
    mesh.apply_translation(-bounds_mean + 0.5)

    assert np.isclose(mesh.bounds.max(), 1.0)
    assert np.isclose(mesh.bounds.min(), 0.0)
    assert np.isclose(mesh.extents.max(), 1.0)
    return mesh


def check_inside_or_outside_mesh(
    mesh: trimesh.Trimesh, points: npt.NDArray
) -> npt.NDArray[np.bool_]:
    is_inside = mesh.contains(points)
    return is_inside


def get_mesh_rotated(mesh: trimesh.Trimesh, time: float) -> trimesh.Trimesh:
    # rotate the mesh to the correct time; 0 is rotation angle 0 and 1 is rotation angle 2pi
    mesh_copy = deepcopy(mesh)
    mesh_copy.apply_transform(
        trimesh.transformations.rotation_matrix(
            angle=2 * np.pi * time, direction=[1, 1, 1], point=(0.5, 0.5, 0.5)
        )
    )
    mesh_copy = mesh_to_unit_cube(mesh_copy)
    return mesh_copy


def check_inside_or_outside_mesh_at_time(
    mesh: trimesh.Trimesh, points: npt.NDArray[np.float64], time: float
) -> npt.NDArray[np.bool_]:
    assert points.shape[1] == 3
    mesh_copy: trimesh.Trimesh | None = get_mesh_rotated(mesh, time)
    # check if they are inside the mesh
    is_inside_at_time = mesh_copy.contains(points)
    # cf. https://github.com/mikedh/trimesh/issues/2410
    mesh_copy = None
    return is_inside_at_time


def check_inside_or_outside_mesh_temporal(
    mesh: trimesh.Trimesh, points: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    assert points.shape[1] == 4
    # allocate empty array for results
    is_inside = np.empty(points.shape[0], dtype=bool)
    # group points by time coordinates
    different_time_points = np.unique(points[:, 3])
    # ic(different_time_points)
    for time in different_time_points:
        # get points at this time
        points_at_time = points[points[:, 3] == time, :3]
        is_inside_at_time = check_inside_or_outside_mesh_at_time(
            mesh, points_at_time, time
        )

        # assign the result to the correct place in the result array
        is_inside[points[:, 3] == time] = is_inside_at_time

    return is_inside
