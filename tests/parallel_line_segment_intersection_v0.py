#!/usr/bin/env python


import threading
import time

import numpy as np
import numba as nb

nthreads = 8


#@nb.jit(nopython=True, nogil=True)
def rotate_2D_vector_CCW(vector):
    x, y = vector

    result_vector = np.empty(2, dtype=np.float64)

    result_vector[0] = -1.0 * y
    result_vector[1] = x

    return result_vector


#@nb.jit(nopython=True, nogil=True)
def calculate_squared_dist(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]

    return dx * dx + dy * dy


#@nb.jit(nopython=True, nogil=True)
def cross_product_2D(a, b):
    return (a[0] * b[1]) - (a[1] * b[0])


#@nb.jit(nopython=True, nogil=True)
def is_given_vector_between_others(x, alpha, beta):
    cp1 = cross_product_2D(alpha, x)
    cp2 = cross_product_2D(x, beta)

    if abs(cp1) < 1e-15 or abs(cp2) < 1e-15:
        return 1
    elif cp1 > 0 and cp2 > 0:
        return 1
    else:
        return 0


#@nb.jit(nopython=True, nogil=True)
def check_if_line_segment_from_node_self_intersects(
    start_coord, end_coord, polygon_coords, start_coord_index
):
    num_vertices = polygon_coords.shape[0]
    si_minus1 = (start_coord_index - 1) % num_vertices
    si_plus1 = (start_coord_index + 1) % num_vertices

    si_plus1_coord = polygon_coords[si_plus1]
    si_minus1_coord = polygon_coords[si_minus1]
    edge_vector_to_plus = si_plus1_coord - start_coord

    edge_vector_from_minus = start_coord - si_minus1_coord

    rough_tangent_vector = edge_vector_to_plus + edge_vector_from_minus

    ipv = rotate_2D_vector_CCW(rough_tangent_vector)

    v = end_coord - start_coord

    if is_given_vector_between_others(v, ipv, -1 * edge_vector_from_minus) == 1:
        return 1
    if is_given_vector_between_others(v, edge_vector_to_plus, ipv) == 1:
        return 1
    else:
        return 0


#@nb.jit(nopython=True, nogil=True)
def check_if_line_segment_intersects_vertical_line(start, end, min_y, max_y, x):
    sx, sy = start
    ex, ey = end

    if sx < x and ex < x:
        return 0
    if sx > x and ex > x:
        return 0
    else:
        # y = m*x + b
        # sy = m*sx + b
        # ey = m*ex + b
        # ey = m*ex + (sy - m*sx)
        # ey - sy = m*(ex - sx)
        # m = (ey - sy)/(ex - sx)
        # b = sy - m*sx

        denom = ex - sx
        if abs(denom) < 1e-15:
            average_x = (sx + ex) / 2.0
            if (average_x - x) < 1e-8:
                return 1
            else:
                return 0

        m = (ey - sy) / denom
        b = sy - m * sx
        y_intersect = m * x + b

        if y_intersect > max_y:
            return 0
        elif y_intersect < min_y:
            return 0
        else:
            return 1


#@nb.jit(nopython=True, nogil=True)
def check_if_line_segment_intersects_box(start, end, min_x, max_x, min_y, max_y):

    if check_if_line_segment_intersects_vertical_line(start, end, min_y, max_y, min_x):
        return 1
    if check_if_line_segment_intersects_vertical_line(start, end, min_y, max_y, max_x):
        return 1
    if check_if_line_segment_intersects_horizontal_line(
        start, end, min_x, max_x, min_y
    ):
        return 1
    if check_if_line_segment_intersects_horizontal_line(
        start, end, min_x, max_x, max_y
    ):
        return 1

    return 0


#@nb.jit(nopython=True, nogil=True)
def check_if_line_segment_intersects_horizontal_line(start, end, min_x, max_x, y):
    sx, sy = start
    ex, ey = end

    if sy < y and ey < y:
        return 0
    if sy > y and ey > y:
        return 0
    else:
        # y = m*x + b
        # sx = m*sy + b
        # ex = m*ey + b
        # ex = m*ey + (sx - m*sy)
        # ex - sx = m*(ey - sy)
        # m = (ex - sx)/(ey - sy)
        # b = sy - m*sx

        denom = ey - sy
        if abs(denom) < 1e-15:
            average_y = (sy + ey) / 2.0
            if (average_y - y) < 1e-8:
                return 1
            else:
                return 0

        m = (ex - sx) / denom

        b = sx - m * sy
        x_intersect = m * y + b

        if x_intersect > max_x:
            return 0
        elif x_intersect < min_x:
            return 0
        else:
            return 1


#@nb.jit(nopython=True, nogil=True)
def check_if_line_segment_going_from_vertex_of_one_polygon_to_vertex_of_another_passes_through_any_polygon(
    pi_a, vi_a, pi_b, vi_b, all_polygon_coords, all_polygons_bounding_box_coords
):
    coords_a = all_polygon_coords[pi_a, vi_a]
    coords_b = all_polygon_coords[pi_b, vi_b]

    normal_to_line_segment = rotate_2D_vector_CCW(coords_b - coords_a)

    if (
        check_if_line_segment_from_node_self_intersects(
            coords_a, coords_b, all_polygon_coords[pi_a], vi_a
        )
        == 1
    ):
        return 100000
    elif (
        check_if_line_segment_from_node_self_intersects(
            coords_b, coords_a, all_polygon_coords[pi_b], vi_b
        )
        == 1
    ):
        return 100000

    num_intersections = 0
    num_polygons = all_polygon_coords.shape[0]
    for pi in range(num_polygons):
        if pi == pi_a or pi == pi_b:
            continue
        else:
            min_x, max_x, min_y, max_y = all_polygons_bounding_box_coords[pi]
            if (
                check_if_line_segment_intersects_box(
                    coords_a, coords_b, min_x, max_x, min_y, max_y
                )
                == 1
            ):
                if (
                    check_if_line_segment_intersects_polygon(
                        coords_a,
                        coords_b,
                        normal_to_line_segment,
                        all_polygon_coords[pi],
                        -1,
                    )
                    == 1
                ):
                    num_intersections += 1

    return num_intersections


#@nb.jit(nopython=True, nogil=True)
def is_left(p0, p1, p2):
    """
    Input:  three points P0, P1, and P2
    Return: > 0 for P2 left of the line through P0 to P1
            = 0 for P2 on the line
            < 0 for P2 right of the line
    """
    p0x, p0y = p0
    p1x, p1y = p1
    p2x, p2y = p2

    return (p1x - p0x) * (p2y - p0y) - (p2x - p0x) * (p1y - p0y)


#@nb.jit(nopython=True, nogil=True)
def check_if_line_segment_intersects_polygon(
    a, b, normal_to_line_segment, polygon, ignore_vertex_index
):
    num_vertices = polygon.shape[0]

    nls_x, nls_y = normal_to_line_segment

    for vi in range(num_vertices):
        next_index = (vi + 1) % num_vertices

        if ignore_vertex_index != -1:
            if vi == ignore_vertex_index:
                continue
            if next_index == ignore_vertex_index:
                continue

        this_coords = polygon[vi]
        next_coords = polygon[next_index]

        is_left_a = is_left(this_coords, next_coords, a)
        is_left_b = is_left(this_coords, next_coords, b)

        if is_left_a < 0 and is_left_b < 0:
            continue
        elif is_left_a > 0 and is_left_b > 0:
            continue

        if is_left_a == 0 or is_left_b == 0:
            return True

        alpha_x, alpha_y = a - this_coords
        beta_x, beta_y = next_coords - this_coords

        denominator = beta_x * nls_x + beta_y * nls_y
        if denominator == 0:
            return False

        t = (alpha_x * nls_x + alpha_y * nls_y) / denominator

        if 0 <= t < 1:
            return True
        else:
            continue

    return False


#
#@nb.jit(
    "void(float64[:,:,:,:], int64[:,:,:,:], float64[:,:], float64[:,:,:], int64[:,:])",
    nopython=True,
    nogil=True,
)
def dist_squared_and_line_segment_calculation_worker(
    dist_squared_matrix,
    line_segment_intersect_matrix,
    polygon_bounding_boxes,
    polygons,
    task_addresses,
):
    """
    Function under test.
    """

    for n in range(task_addresses.shape[0]):
        a_ci, a_ni, b_ci, b_ni = (
            task_addresses[n][0],
            task_addresses[n][1],
            task_addresses[n][2],
            task_addresses[n][3],
        )
        dist_squared = calculate_squared_dist(
            polygons[a_ci][a_ni], polygons[b_ci][b_ni]
        )
        num_intersects = check_if_line_segment_going_from_vertex_of_one_polygon_to_vertex_of_another_passes_through_any_polygon(
            a_ci, a_ni, b_ci, b_ni, polygons, polygon_bounding_boxes
        )

        dist_squared_matrix[a_ci][a_ni][b_ci][b_ni] = dist_squared
        dist_squared_matrix[b_ci][b_ni][a_ci][a_ni] = dist_squared
        line_segment_intersect_matrix[a_ci][a_ni][b_ci][b_ni] = num_intersects
        line_segment_intersect_matrix[b_ci][b_ni][a_ci][a_ni] = num_intersects


#@nb.jit(nopython=True, nogil=True)
def create_dist_and_line_segment_interesection_test_args(
    num_cells,
    num_nodes_per_cell,
    init_cells_bounding_box_arrays,
    init_all_cells_node_coords,
):
    info_update_tracker = -1 * np.ones(
        (num_cells, num_nodes_per_cell, num_cells, num_nodes_per_cell), dtype=np.float64
    )
    tasks = []

    for ci in range(num_cells):
        for ni in range(num_nodes_per_cell):
            for other_ci in range(num_cells):
                if ci != other_ci:
                    relevant_info_update_tracker_slice = info_update_tracker[ci][ni][
                        other_ci
                    ]
                    for other_ni in range(num_nodes_per_cell):
                        if relevant_info_update_tracker_slice[other_ni] != 1:
                            info_update_tracker[ci][ni][other_ci][other_ni] = 1
                            info_update_tracker[other_ci][other_ni][ci][ni] = 1
                            tasks.append((ci, ni, other_ci, other_ni))

    return tasks


def singlethread_func(
    inner_func, num_cells, num_nodes, polygon_bounding_boxes, polygons
):
    """
    Run the given function inside a single thread.
    """

    tasks = create_dist_and_line_segment_interesection_test_args(
        num_cells, num_nodes, polygon_bounding_boxes, polygons
    )
    tasks = np.array(tasks, dtype=np.int64)

    dist_squared_matrix = -1 * np.ones(
        (num_cells, num_nodes, num_cells, num_nodes), dtype=np.float64
    )
    line_segment_intersect_matrix = -1 * np.ones(
        (num_cells, num_nodes, num_cells, num_nodes), dtype=np.int64
    )

    inner_func(
        dist_squared_matrix,
        line_segment_intersect_matrix,
        polygon_bounding_boxes,
        polygons,
        tasks,
    )

    return dist_squared_matrix, line_segment_intersect_matrix


def multithread_func(
    inner_func, num_threads, num_cells, num_nodes, polygon_bounding_boxes, polygons
):
    """
    Run the given function inside *num_threads* threads, splitting its
    arguments into equal-sized chunks.
    """

    tasks = create_dist_and_line_segment_interesection_test_args(
        num_cells, num_nodes, polygon_bounding_boxes, polygons
    )
    tasks = np.array(tasks, dtype=np.int64)
    num_tasks = len(tasks)

    dist_squared_matrix = -1 * np.ones(
        (num_cells, num_nodes, num_cells, num_nodes), dtype=np.float64
    )
    line_segment_intersect_matrix = -1 * np.ones(
        (num_cells, num_nodes, num_cells, num_nodes), dtype=np.int64
    )

    chunklen = (num_tasks + num_threads - 1) // num_threads
    # Create argument tuples for each input chunk
    chunks = []
    for i in range(num_threads):
        relevant_tasks = tasks[i * chunklen : (i + 1) * chunklen]
        chunks.append(
            (
                dist_squared_matrix,
                line_segment_intersect_matrix,
                polygon_bounding_boxes,
                polygons,
                relevant_tasks,
            )
        )
    # Spawn one thread per chunk
    threads = [threading.Thread(target=inner_func, args=(c,)) for c in chunks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return dist_squared_matrix, line_segment_intersect_matrix


#@nb.jit(nopython=True, nogil=True)
def calculate_polygon_bounding_box(polygon):
    num_vertices = polygon.shape[0]

    min_x = 0.0
    max_x = 0.0

    min_y = 0.0
    max_y = 0.0

    for i in range(num_vertices):
        this_x, this_y = polygon[i]
        if i == 0:
            min_x = this_x
            max_x = min_x

            min_y = this_y
            max_y = min_y
        else:
            if this_x < min_x:
                min_x = this_x
            elif this_x > max_x:
                max_x = this_x

            if this_y < min_y:
                min_y = this_y
            elif this_y > max_y:
                max_y = this_y

    return min_x, max_x, min_y, max_y


def create_initial_bounding_box_polygon_array(
    num_cells, num_nodes_per_cell, environment_cells_node_coords
):
    bounding_box_polygon_array = np.zeros((num_cells, 4), dtype=np.float64)

    for ci in range(num_cells):
        bounding_box_polygon_array[ci] = calculate_polygon_bounding_box(
            environment_cells_node_coords[ci]
        )

    return bounding_box_polygon_array


def generate_polygons(
    num_polygons, num_vertices_per_polygon, max_radius=1.0, min_max_origin=[0.0, 5.0]
):
    polygons = np.zeros((num_polygons, num_vertices_per_polygon, 2), dtype=np.float64)

    thetas = np.linspace(0, 2 * np.pi, num=num_vertices_per_polygon)
    xs = np.cos(thetas)
    ys = np.sin(thetas)
    delta_space = (min_max_origin[1] - max_radius) - (min_max_origin[0] + max_radius)

    for n in range(num_polygons):
        origin_x = np.random.rand() * delta_space + min_max_origin[0] + max_radius
        origin_y = np.random.rand() * delta_space + min_max_origin[0] + max_radius
        radii = max_radius * np.random.rand(num_vertices_per_polygon)
        polygons[n, :, 0] = origin_x + radii * xs
        polygons[n, :, 1] = origin_y + radii * ys

    return polygons


num_cells = 50
num_nodes = 16
min_max_origin = [0, 20.0]
polygons = generate_polygons(num_cells, num_nodes, min_max_origin=min_max_origin)
polygon_bounding_boxes = create_initial_bounding_box_polygon_array(
    num_cells, num_nodes, polygons
)

dist_squared_matrix0, line_segment_intersect_matrix0 = singlethread_func(
    dist_squared_and_line_segment_calculation_worker,
    num_cells,
    num_nodes,
    polygon_bounding_boxes,
    polygons,
)

num_repeats = 10
st = time.time()
for n in range(10):
    dist_squared_matrix0, line_segment_intersect_matrix0 = singlethread_func(
        dist_squared_and_line_segment_calculation_worker,
        num_cells,
        num_nodes,
        polygon_bounding_boxes,
        polygons,
    )
et = time.time()

print("avg time to run single_thread_func: ", (et - st) / num_repeats)

st = time.time()
for n in range(10):
    dist_squared_matrix, line_segment_intersect_matrix = multithread_func(
        dist_squared_and_line_segment_calculation_worker,
        nthreads,
        num_cells,
        num_nodes,
        polygon_bounding_boxes,
        polygons,
    )
et = time.time()

print("avg. time to run multithread_func: ", (et - st) / num_repeats)

print(
    "dist_squared_matrix agrees: ",
    np.all(np.abs(dist_squared_matrix - dist_squared_matrix0) < 1e-12),
)

print(
    "line_segment_intersect_matrix agrees: ",
    np.all(np.abs(line_segment_intersect_matrix - line_segment_intersect_matrix0) == 0),
)
