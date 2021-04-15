# main.py, for install verification

# LRPK LRPK E2.9

import LRPK
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

# define polygon
polygon_vertices = [
    [18, 11],
    [25, 13.5],
    [20, 14],
    [21, 17],
    [17, 15],
    [16, 14],
    [13, 17],
    [14.5, 15],
    [15, 13],
    [11, 12],
    ]

# define free workspace
free_workspace_vertices = [
    [7, 8],
    [30, 8],
    [30, 20],
    [7, 20]
]

polygon = LRPK.Polygon(polygon_vertices)
free_workspace = LRPK.Polygon(free_workspace_vertices)

# conduct trapezoidation over free workspace and polygon
trapezoids = LRPK.trapezoidation(free_workspace, [polygon])

# plot convex / non-convex vertices
fig, ax = plt.subplots()
patches = [mpatches.Polygon(polygon.vertex_array), mpatches.Polygon(free_workspace.vertex_array)]
colors = np.linspace(0, 1, 9)
collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
collection.set_array(colors)
ax.add_collection(collection)
for vertex in polygon.vertices:
    if vertex.convex:
        ax.annotate("convex", (vertex.x, vertex.y))
    else:
        ax.annotate("non-convex", (vertex.x, vertex.y))
plt.axis('equal')
plt.tight_layout()

# plot vertex coordinates
fig, ax = plt.subplots()
patches = [mpatches.Polygon(polygon.vertex_array), mpatches.Polygon(free_workspace.vertex_array)]
colors = np.linspace(0, 1, 9)
collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
collection.set_array(colors)
ax.add_collection(collection)
for vertex in polygon.vertices:
    ax.annotate(vertex.cartesian, (vertex.x, vertex.y))
for vertex in free_workspace.vertices:
    ax.annotate(vertex.cartesian, (vertex.x, vertex.y))
plt.axis('equal')
plt.tight_layout()

# plot vertex type
fig, ax = plt.subplots()
patches = [mpatches.Polygon(polygon.vertex_array), mpatches.Polygon(free_workspace.vertex_array)]
colors = np.linspace(0, 1, 9)
collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
collection.set_array(colors)
ax.add_collection(collection)
for vertex in polygon.vertices:
    ax.annotate(vertex.type, (vertex.x, vertex.y))
plt.axis('equal')
plt.tight_layout()

# plot trapezoidation
fig, ax = plt.subplots()
patches = [
    mpatches.Polygon(polygon.vertex_array),
    mpatches.Polygon(free_workspace.vertex_array),
    ]
for trapezoid in trapezoids:
    patches.append(mpatches.Polygon(trapezoid.vertex_array))
colors = np.linspace(0, 1, 9)
collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
collection.set_array(colors)
ax.add_collection(collection)
for i, trapezoid in enumerate(trapezoids):
    ax.annotate(f"T{i}", trapezoid.center_cartesian)
plt.axis('equal')
plt.tight_layout()
plt.show()
