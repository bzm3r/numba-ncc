import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

chemotaxis_successes_per_variant_per_num_cells = np.array(
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.4], [0.3, 0.2, 0.1]]
)
num_experiment_repeats = [20, 20, 20]
num_cells = [1, 2, 4]
box_widths = [1, 2, 2]
box_heights = [1, 1, 2]
colors = ["r", "g", "b"]

per_variant_space = 1.0
num_bars = len(chemotaxis_successes_per_variant_per_num_cells)
bar_width = (0.75 / num_bars) * per_variant_space
per_variant_initial_offset = per_variant_space * 0.5
within_variant_offset = (1.0 - (bar_width * num_bars)) / (num_bars + 1)
within_variant_initial_offset = 0.5 * bar_width + within_variant_offset
within_variant_tick_delta = bar_width + within_variant_offset

for per_variant_index, setup_data in enumerate(
    zip(
        chemotaxis_successes_per_variant_per_num_cells,
        num_experiment_repeats,
        num_cells,
        box_widths,
        box_heights,
    )
):

    successes_per_variant, nr, nc, bw, bh = setup_data

    requested_color = colors[per_variant_index]

    within_variant_indices = np.arange(len(successes_per_variant))

    this_variant_offset = (
        per_variant_initial_offset + per_variant_space * per_variant_index
    )
    ax.bar(
        [
            this_variant_offset
            + within_variant_initial_offset
            + j * within_variant_tick_delta
            for j in within_variant_indices
        ],
        successes_per_variant,
        bar_width,
        label="nc={} ({}x{}), nr={}".format(nc, bw, bh, nr),
        color=requested_color,
    )

plt.show()
