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
