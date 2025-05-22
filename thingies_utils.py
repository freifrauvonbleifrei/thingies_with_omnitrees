import trimesh
import numpy as np
import numpy.typing as npt


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
) -> np.array:
    is_inside = mesh.contains(points)
    return is_inside


def check_inside_or_outside_mesh_temporal(
    mesh: trimesh.Trimesh, points: npt.NDArray
) -> np.array:
    assert points.shape[1] == 4
    # allocate empty array for results
    is_inside = np.empty(points.shape[0], dtype=bool)
    # group points by time coordinates
    different_time_points = np.unique(points[:, 3])
    for time in different_time_points:
        # get points at this time
        points_at_time = points[points[:, 3] == time, :3]
        # rotate the mesh to the correct time; 0 is rotation angle 0 and 1 is rotation angle 2pi
        mesh_copy = mesh.copy()
        mesh_copy.apply_transform(
            trimesh.transformations.rotation_matrix(
                angle=2 * np.pi * time, direction=[1, 1, 1], point=(0.5, 0.5, 0.5)
            )
        )
        # check if they are inside the mesh
        is_inside_at_time = mesh_copy.contains(points_at_time)
        # assign the result to the correct place in the result array
        is_inside[points[:, 3] == time] = is_inside_at_time
    return is_inside
