import random
import numpy as np
from shapely import Polygon, MultiPolygon, affinity

from vectorizer import vectorize_wkt

from polygenerator import (
    random_polygon,
    random_star_shaped_polygon,
    random_convex_polygon,
)


def normalize(data, method="min_max"):
    if method == "min_max":
        # Compute min and max values along the num_features dimension
        min_vals = np.min(data[:, :, :2], axis=1, keepdims=True)
        max_vals = np.max(data[:, :, :2], axis=1, keepdims=True)
        # Perform min-max normalization
        data[:, :, :2] -= min_vals
        data[:, :, :2] /=  max_vals - min_vals

    else:
        mean = np.mean(data[:, :, :2], axis=1, keepdims=True)
        std = np.std(data[:, :, :2], axis=1, keepdims=True)

        data[:, :, :2] -= mean
        data[:, :, :2] /=  std
    return data


def get_num_points(func):
    if func==random_polygon:
        num_points = random.randint(3, 32)
    elif func==random_star_shaped_polygon:
        num_points = random.randint(3, 32)
    else:
        num_points = random.randint(3, 12)
    return num_points


def generate_polygon():
    # Define the list of functions and their corresponding probabilities
    functions_shell = [random_polygon, random_star_shaped_polygon, random_convex_polygon]
    functions_hole = [random_star_shaped_polygon, random_convex_polygon]
    probs_shell = [0.5, 0.35, 0.15]
    probs_hole = [0.5, 0.5]

    valid_poly = False
    while not valid_poly:
        # Choose a function based on the specified probabilities
        function_shell = random.choices(functions_shell, weights=probs_shell, k=1)[0]
        function_hole = random.choices(functions_hole, weights=probs_hole, k=1)[0]

        num_holes = random.choices([0, 1, 2], weights=[0.5, 0.25, 0.25], k=1)[0]

        if num_holes == 0:
            shell = Polygon(function_shell(get_num_points(function_shell)))
            poly = Polygon(shell=shell)

        else:
            while not valid_poly:
                shell = Polygon(function_shell(get_num_points(function_shell)))
                holes = [Polygon(function_hole(get_num_points(function_hole))) for _ in range(num_holes)]

                if num_holes == 1:
                    hole_group = holes[0]

                elif num_holes == 2:
                    # seperate holes to make sure they are not overlap
                    while holes[0].intersects(holes[1]):
                        shit_x, shit_y = random.random(), random.random()
                        holes[1] = affinity.translate(holes[1], shit_x, shit_y)
                    hole_group = MultiPolygon(holes)

                else:
                    raise(NotImplementedError)

                # Step down scale up factor, shrink and move holes into the shell to make complex polygons
                for i in np.arange(1, 0, -0.05):
                    group_centroid = hole_group.centroid
                    dx, dy = shell.centroid.x - group_centroid.x, shell.centroid.x - group_centroid.y
                    # shrink holes
                    hole_group_scale = affinity.scale(hole_group, xfact=i, yfact=i, origin=group_centroid)
                    # move the shell's center
                    hole_group_shift = affinity.translate(hole_group_scale, dx, dy)

                    try:
                        if isinstance(hole_group_shift, Polygon):
                            poly = Polygon(shell=shell.boundary, holes=[hole_group_shift.boundary])
                        else:
                            poly = Polygon(shell=shell.boundary, holes=[hole.boundary for hole in hole_group_shift.geoms])

                        if poly.is_valid:
                            valid_poly = True
                            break
                    # If not a valid polygon, continue
                    except:
                        continue

        if not poly.is_valid:
            print("invalid poly")
            continue
        else:
            valid_poly = True

    return poly


def generate_polygons(num_poly, wkt=True):
    polygons = []

    for i in range(num_poly):
        poly = generate_polygon()
        polygons.append(poly)

    return polygons


def generate_polygons_and_vectorize(num_poly, max_points=64):
    geoms = []
    wkts = []

    for i in range(num_poly):
        poly = generate_polygon()
        poly_wkt = poly.wkt
        geom = vectorize_wkt(poly_wkt, max_points=max_points, fixed_size=True, simplify=True)
        geoms.append(geom)
        wkts.append(poly_wkt)

    geoms = np.stack(geoms, axis=0)
    geoms = normalize(geoms, method="min_max")

    return geoms, wkts


if __name__ == "__main__":
    geoms = generate_polygons_and_vectorize(10)
    print(geoms[0])



