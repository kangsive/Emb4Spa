import random
import numpy as np
from shapely.geometry import Polygon
from utils.vectorizer import vectorize_wkt

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


def generate_polygon(num_poly, wkt=True):
    polygons = []
    # Define the list of functions and their corresponding probabilities
    functions = [random_polygon, random_star_shaped_polygon, random_convex_polygon]
    probabilities = [0.5, 0.35, 0.15]
    for i in range(num_poly):
        # Choose a function based on the specified probabilities
        chosen_function = random.choices(functions, weights=probabilities, k=1)[0]

        if chosen_function==random_polygon:
            num_points = random.randint(3, 32)
        elif chosen_function==random_star_shaped_polygon:
            num_points = random.randint(3, 32)
        else:
            num_points = random.randint(3, 12)

        poly = Polygon(chosen_function(num_points))

        if wkt:
            poly = poly.wkt

        polygons.append(poly)
    return polygons


def generate_polygon_and_vectorize(num_poly, max_points=64):
    geoms = []
    wkts = []
    # Define the list of functions and their corresponding probabilities
    functions = [random_polygon, random_star_shaped_polygon, random_convex_polygon]
    probabilities = [0.5, 0.35, 0.15]
    for i in range(num_poly):
        valid_poly = False
        while not valid_poly:
          # Choose a function based on the specified probabilities
          chosen_function = random.choices(functions, weights=probabilities, k=1)[0]

          if chosen_function==random_polygon:
              num_points = random.randint(3, 32)
          elif chosen_function==random_star_shaped_polygon:
              num_points = random.randint(3, 32)
          else:
              num_points = random.randint(3, 12)

          poly = Polygon(chosen_function(num_points))

          if not poly.is_valid:
              print("invalid poly")
              continue
          else:
              valid_poly = True

        poly_wkt = poly.wkt

        geom = vectorize_wkt(poly_wkt, max_points=max_points, fixed_size=True)

        geoms.append(geom)
        wkts.append(poly_wkt)

    geoms = np.stack(geoms, axis=0)
    geoms = normalize(geoms, method="min_max")

    return geoms, wkts


if __name__ == "__main__":
    geoms = generate_polygon_and_vectorize(10)
    print(geoms[0])



