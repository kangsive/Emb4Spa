import numpy as np
from shapely.geometry import Polygon, MultiPolygon


GEOMETRY_TYPES = ["GeometryCollection", "Point", "LineString", "Polygon", "MultiPoint", "MultiLineString",
                  "MultiPolygon", "Geometry"]
X_INDEX = 0  # the X coordinate position
Y_INDEX = 1  # the Y coordinate position
RENDER_INDEX = Y_INDEX + 1  # Render index start
RENDER_LEN = 3  # Render one-hot vector length
ONE_HOT_LEN = 2 + RENDER_LEN  # Length of the one-hot encoded part
STOP_INDEX = RENDER_INDEX + 1  # Stop index for the first geometry. A second one follows
FULL_STOP_INDEX = STOP_INDEX + 1  # Full stop index. No more points to follow
GEO_VECTOR_LEN = FULL_STOP_INDEX + 1  # The length needed to describe the features of a geometry point


def reverse_vector_polygon(geom_vector):
    """
    Reverse vectorized geometry back to geometry shapes

    Parameters
    ----------
    geom_vector: 2D numpy array, or Tensor

    Returns
    -------
    shaply geometry, Polygon or MultiPolygon
    
    """
    start_indices = np.where((geom_vector[:, RENDER_INDEX] == 1) & (np.roll(geom_vector[:, RENDER_INDEX], 1) == 0))[0]
    end_indices = np.where((geom_vector[:, RENDER_INDEX] == 0) & (np.roll(geom_vector[:, RENDER_INDEX], 1) == 1))[0]
    shapes = []
    for start_idx, end_idx in zip(start_indices, end_indices):
        if end_idx < start_idx and geom_vector[end_idx, FULL_STOP_INDEX] == 1: # Fix 111101111 like case
            start_idx = 0
        shape = geom_vector[start_idx:end_idx+1, X_INDEX:Y_INDEX+1]
        shapes.append(shape)

    if len(shapes) > 0: # TODO - consider inner and outer points
        polygons = [Polygon(shape) for shape in shapes if shape.shape[0]>=4]
        # return MultiPolygon(polygons)
    # elif len(shapes) == 1:
        return Polygon(shapes[0])
    else:
        return Polygon(geom_vector[:, X_INDEX:Y_INDEX+1])
    

if __name__ == "__main__":

    geom_vector = np.array([[ 0.5521, -0.0549,  1.0000,  0.0000,  0.0000],
        [ 0.2319,  0.1135,  1.0000,  0.0000,  0.0000],
        [ 0.0953,  0.0166,  1.0000,  0.0000,  0.0000],
        [-0.0904,  0.1147,  1.0000,  0.0000,  0.0000],
        [-0.1198,  0.0945,  1.0000,  0.0000,  0.0000],
        [-0.2867,  0.0837,  1.0000,  0.0000,  0.0000],
        [-0.2767, -0.0193,  1.0000,  0.0000,  0.0000],
        [-0.3070, -0.0411,  1.0000,  0.0000,  0.0000],
        [ 0.2012, -0.3078,  1.0000,  0.0000,  0.0000],
        [ 0.5521, -0.0549,  0.0000,  0.0000,  1.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  1.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  1.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  1.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000,  1.0000]])
    
    polygon = reverse_vector_polygon(geom_vector)

    x, y = polygon.exterior.xy

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y)
    plt.title('Polygon')
    plt.axis('off')  # Turn off axis，trun on if compare scale
    # plt.grid(True) 
    plt.show()