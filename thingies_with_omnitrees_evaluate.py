#!/usr/bin/env python
import argparse as arg
import bitarray as ba
from filelock import FileLock
from itertools import pairwise
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from icecream import ic
import os.path
import pandas as pd
import thingi10k
import trimesh

try:
    import OpenGL.GL as gl  # type: ignore
except ImportError:
    pass


import dyada.coordinates
import dyada.discretization
import dyada.linearization


from thingies_utils import (
    mesh_to_unit_cube,
    check_inside_or_outside_mesh,
    check_inside_or_outside_mesh_at_time,
)


class ErrorL1File:
    def __init__(self, num_sobol_samples):
        self.l1fileName = "l1_errors_s" + str(num_sobol_samples) + ".csv"
        self.lockFile = self.l1fileName + ".lock"
        self.columns = [
            "thingi_file_id",
            "tree",
            "allowed_tree_boxes",
            "num_sobol_samples",
            "num_occupancy_samples",
            "num_error_samples",
            "num_boxes",
            "num_boxes_occupied",
            "num_tree_nodes",
            "tree_number_of_1s",
            "l1error",
        ]

        with FileLock(self.lockFile):
            try:
                with open(self.l1fileName, "x") as f:
                    df = pd.DataFrame(columns=self.columns)
                    df.to_csv(f, index=False)
                    ic("created " + self.l1fileName)
            except FileExistsError as e:
                pass  # File already exists

    def append_row(self, row_dict):
        df = pd.DataFrame([row_dict])
        # check columns complete and make same order
        df = df[self.columns]
        with FileLock(self.lockFile):
            df.to_csv(self.l1fileName, mode="a", index=False, header=False)

    def check_row_exists(self, row_dict):
        rows_per_chunk = 1024
        for chunk in pd.read_csv(self.l1fileName, chunksize=rows_per_chunk):
            for _, row in chunk.iterrows():
                if all(row[k] == v for k, v in row_dict.items()):
                    return True
        return False


def check_inside_or_outside_tree(
    discretization: dyada.discretization.Discretization,
    binary_discretization_occupancy: np.ndarray,
    points: np.ndarray,
) -> npt.NDArray[np.bool_]:
    is_inside = np.zeros(len(points), dtype=bool)
    for p_i, point in enumerate(points):
        box_index = discretization.get_containing_box(point)
        if binary_discretization_occupancy[box_index]:
            is_inside[p_i] = True
    return is_inside


def get_all_time_slices(
    discretization_4d: dyada.discretization.Discretization,
    binary_discretization_occupancy: np.ndarray,
):
    # get all time slices of the discretization
    num_dimensions = discretization_4d.descriptor.get_num_dimensions()
    assert num_dimensions == 4
    time_slices = dyada.discretization.SliceDictInDimension(discretization_4d, 3, True)
    for time, _ in time_slices.items():
        time_slice, mapping = time_slices[time]
        sliced_binary_occupancy = np.zeros(len(time_slice), dtype=bool)
        for k, v in mapping.items():
            sliced_binary_occupancy[v] = binary_discretization_occupancy[k]
        time_slices[time] = (time_slice, sliced_binary_occupancy)
    return time_slices


def get_monte_carlo_l1_error(
    mesh: trimesh.Trimesh,
    discretization: dyada.discretization.Discretization,
    binary_discretization_occupancy: np.ndarray,
    num_samples: int,
) -> float:
    # generate random points in the unit cube
    num_dimensions = discretization.descriptor.get_num_dimensions()

    if num_dimensions == 4:
        num_temporal_samples = int(num_samples ** (1 / num_dimensions))
        temporal_samples = np.random.rand(num_temporal_samples)
        num_spatial_samples = num_samples // num_temporal_samples
        spatial_samples = np.random.rand(num_spatial_samples, num_dimensions - 1)
        means_per_time = np.zeros((num_temporal_samples), dtype=np.float64)
        tree_time_slices = get_all_time_slices(
            discretization, binary_discretization_occupancy
        )
        for i, time in enumerate(temporal_samples):
            is_inside_mesh = check_inside_or_outside_mesh_at_time(
                mesh, spatial_samples, time
            )
            # find the right time slice in the tree

            tree_time_slice, binary_discretization_occupancy_slice = tree_time_slices[
                time
            ]
            is_inside_tree = check_inside_or_outside_tree(
                tree_time_slice,
                binary_discretization_occupancy_slice,
                spatial_samples,
            )
            means_per_time[i] = (is_inside_mesh ^ is_inside_tree).mean()
        return means_per_time.mean()
    else:
        points = np.random.rand(num_samples, num_dimensions)
        is_inside_mesh = check_inside_or_outside_mesh(mesh, points)
        is_inside_tree = check_inside_or_outside_tree(
            discretization, binary_discretization_occupancy, points
        )
    # calculate the L1 error
    return (is_inside_mesh ^ is_inside_tree).mean()


def get_binary_discretization_occupancy(
    discretization: dyada.discretization.Discretization,
    mesh: trimesh.Trimesh,
    num_samples: int,
):
    num_dimensions = discretization.descriptor.get_num_dimensions()
    random_points_3d = np.random.rand(num_samples, 3)
    binary_discretization_occupancy = np.zeros(len(discretization), dtype=bool)
    if num_dimensions == 4:
        num_temporal_samples = int(num_samples ** (1 / num_dimensions))
        temporal_samples = np.random.rand(num_temporal_samples)
        num_spatial_samples = num_samples // num_temporal_samples
        for box_index in range(len(discretization)):
            interval = dyada.discretization.coordinates_from_box_index(
                discretization, box_index
            )
            random_points_in_interval = (
                random_points_3d[:num_spatial_samples]
                * (interval.upper_bound[:3] - interval.lower_bound[:3])
                + interval.lower_bound[:3]
            )
            times_in_interval = (
                temporal_samples * (interval.upper_bound[3] - interval.lower_bound[3])
                + interval.lower_bound[3]
            )
            is_inside_at_time = np.zeros(num_temporal_samples, dtype=np.float64)
            for i, time in enumerate(times_in_interval):
                is_inside_at_time[i] = np.mean(
                    check_inside_or_outside_mesh_at_time(
                        mesh, random_points_in_interval, time
                    )
                )
            if np.mean(is_inside_at_time) > 0.5:
                binary_discretization_occupancy[box_index] = True

    else:
        for box_index in range(len(discretization)):
            interval = dyada.discretization.coordinates_from_box_index(
                discretization, box_index
            )
            # get random points in the interval
            random_points_in_interval = (
                random_points_3d * (interval.upper_bound - interval.lower_bound)
                + interval.lower_bound
            )
            is_inside = check_inside_or_outside_mesh(mesh, random_points_in_interval)

            if np.mean(is_inside) > 0.5:
                binary_discretization_occupancy[box_index] = True
    return binary_discretization_occupancy


def get_mesh_from_discretization(discretization, binary_discretization_occupancy):
    # construct a mesh from the discretization and binary values
    list_of_boxes = []
    for box_index in range(len(discretization)):
        if binary_discretization_occupancy[box_index]:
            interval = dyada.discretization.coordinates_from_box_index(
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


def plot_mesh_with_pyplot(mesh: trimesh.Trimesh, azim=200, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.view_init(azim=azim, share=True)
    ax.plot([0, 1], [0, 1], [0, 1])
    ax.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
    )
    if filename is not None:
        fig.savefig(filename + ".svg", dpi=300)
        plt.close()
    else:
        plt.show()


def plot_mesh_with_opengl(mesh: trimesh.Trimesh, filename):
    # draw unit cube as bounding box
    coordinates = [dyada.coordinates.interval_from_sequences([0, 0, 0], [1, 1, 1])]
    projection = [0, 2, 1]
    dyada.drawing.plot_boxes_3d_pyopengl(
        coordinates,
        wireframe=True,
        alpha=0.3,
        colors="gray",
        filename=None,
        projection=projection,
        height=512,
        width=512,
    )
    gl.glEnable(gl.GL_DEPTH_TEST)

    vertices = np.array(mesh.vertices, dtype=np.float32)
    edges = mesh.edges_unique
    faces = np.array(mesh.faces, dtype=np.uint32)

    gl.glBegin(gl.GL_LINES)
    gl.glColor4fv((*to_rgb("black"), 1.0))
    for edge in edges:
        for idx in edge:
            gl.glVertex3fv(vertices[idx][projection])
    gl.glEnd()

    gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)  # try to avoid overdrawing
    gl.glPolygonOffset(2.0, 2.0)
    gl.glBegin(gl.GL_TRIANGLES)
    gl.glColor4fv((*to_rgb("orange"), 0.2))
    for face in mesh.faces:
        # normal = mesh.face_normals[face]
        # gl.glNormal3fv(normal)
        for idx in face:
            gl.glVertex3fv(vertices[idx][projection])
    gl.glEnd()
    gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
    dyada.drawing.gl_save_file(filename, width=512, height=512)


def shannon_information(probability_of_ones: float) -> float:
    if probability_of_ones == 0 or probability_of_ones == 1:
        return 0
    return -probability_of_ones * np.log2(probability_of_ones) - (
        1 - probability_of_ones
    ) * np.log2(1 - probability_of_ones)


def get_shannon_information(
    binary_array_or_bitarray,
) -> float:
    # cast to np.ndarray[bool] if necessary
    if isinstance(binary_array_or_bitarray, ba.bitarray):
        binary_array_or_bitarray = np.array(
            binary_array_or_bitarray.tolist(), dtype=bool
        )
    # calculate the Shannon information
    length = len(binary_array_or_bitarray)
    num_ones = np.sum(binary_array_or_bitarray)
    p_ones = num_ones / length
    return shannon_information(p_ones)


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
        default=512,
    )  # TODO replace by which files are found
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

    error_file = ErrorL1File(str(args.sobol_samples))

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
        plot_mesh_with_pyplot(
            mesh, azim=220, filename=str(thingi["file_id"]) + "_original"
        )

        tree_names = ["octree", "omnitree_1"]

        for allowed_tree_boxes in number_tree_boxes:
            for tree_name in tree_names:
                ic(allowed_tree_boxes, tree_name)
                filename_tree = (
                    str(thingi["file_id"])
                    + "_"
                    + tree_name
                    + "_"
                    + str(allowed_tree_boxes)
                    + "_s"
                    + str(args.sobol_samples)
                )
                filename_binary = filename_tree + "_3d.bin"
                if not os.path.isfile(filename_binary):
                    print(filename_binary + " does not exist, skipping")
                    continue

                discretization = dyada.discretization.Discretization(
                    dyada.linearization.MortonOrderLinearization(),
                    dyada.discretization.RefinementDescriptor.from_file(
                        filename_binary
                    ),
                )

                number_error_samples = 262144
                number_occupancy_samples = args.sobol_samples * 8

                partial_dict = {
                    "thingi_file_id": thingi["file_id"],
                    "tree": tree_name,
                    "allowed_tree_boxes": allowed_tree_boxes,
                    "num_sobol_samples": args.sobol_samples,
                    "num_occupancy_samples": number_occupancy_samples,
                    "num_error_samples": number_error_samples,
                }
                if error_file.check_row_exists(partial_dict):
                    print(partial_dict, " already evaluated, skipping")
                    continue

                binary_discretization_occupancy = get_binary_discretization_occupancy(
                    discretization, mesh, number_occupancy_samples
                )
                with open(filename_tree + "_occupancy.bin", "wb") as f:
                    ba.bitarray(binary_discretization_occupancy.tolist()).tofile(f)
                assert len(binary_discretization_occupancy) == len(discretization)
                monte_carlo_l1_error = get_monte_carlo_l1_error(
                    mesh,
                    discretization,
                    binary_discretization_occupancy,
                    number_error_samples,
                )
                ic(monte_carlo_l1_error)

                store_dict = partial_dict | {
                    "num_boxes": len(discretization),
                    "num_boxes_occupied": np.sum(binary_discretization_occupancy),
                    "num_tree_nodes": len(discretization.descriptor),
                    "tree_number_of_1s": discretization.descriptor.get_data().count(),
                    "l1error": monte_carlo_l1_error,
                }
                error_file.append_row(store_dict)
