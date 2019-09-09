import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import DrawingArea, AnnotationBbox


def draw_circle_arrangement(
    ax,
    drawing_origin,
    radius,
    num_circles,
    box_height,
    box_width,
    corridor_height,
    corridor_width,
):

    da = DrawingArea(box_width * 10, box_height * 10, 0, 0)

    y_delta_index = 0
    x_delta_index = 0

    origin = np.array([radius, radius])
    y_delta = np.array([0.0, 2 * radius])
    x_delta = np.array([2 * radius, 0.0])

    corridor_boundary_coords = (
        np.array([[cw, 0.0], [0.0, 0.0], [0.0, ch], [cw, ch]], dtype=np.float64)
        + origin
    )
    corridor_boundary_patch = mpatch.Polygon(
        corridor_boundary_coords,
        closed=False,
        fill=False,
        color="r",
        ls="solid",
        clip_on=False,
    )

    for ci in range(num_circles):
        cell_patch = mpatches.Circle(
            origin + y_delta_index * y_delta + x_delta_index * x_delta,
            radius=radius,
            color="k",
            fill=False,
            ls="solid",
            clip_on=False,
        )
        da.add_artist(cell_patch)

        if y_delta_index == box_height - 1:
            y_delta_index = 0
            x_delta_index += 1
        else:
            y_delta_index += 1
    ab = AnnotationBbox(
        da,
        xy=(drawing_origin[0], 0),
        xybox=drawing_origin,
        xycoords="data",
        boxcoords=("data", "axes fraction"),
        box_alignment=(0.5, 0.5),
        frameon=False,
    )

    ax.add_artist(ab)


fig, ax = plt.subplots()
# each tuple is: number of circles, height of box containing circles, width of
# box containing circle
circle_arrangements = [(10, 2, 5), (3, 1, 3), (1, 1, 1)]
data_scale = 0.02
data = data_scale * np.random.rand(3)
ax.set_ylim([0, data_scale])
ax.plot(np.arange(3) + 1, data, marker="o")
ax.get_xaxis().set_ticklabels([])


for i, ca in enumerate(circle_arrangements):
    do = np.array([1.0 + i, -0.13])
    nc, bh, bw = ca
    draw_circle_arrangement(ax, do, 10, nc, bh, bw)

plt.show()
