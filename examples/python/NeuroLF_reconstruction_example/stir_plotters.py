import stir
import stirextra
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

stir.Verbosity.set(0)

plt.rcParams["figure.figsize"] = [10, 5]

# these lines change the width of the notebook to always fit the browser
from IPython.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))
display(HTML("<style>.output_result { max-width:100% !important; }</style>"))


# function to plot STIR image
def plot_image(
    voxels_on_cartesian_grid,
    vmax=0,
    vmin=0,
    colourmap="viridis",
    intuitive_orientation=True,
    labels=True,
    overlay=None,
):
    image = stirextra.to_numpy(voxels_on_cartesian_grid)
    dim_z, dim_y, dim_x = image.shape

    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    plt.subplots_adjust(bottom=0.25)

    min_extent = [
        voxels_on_cartesian_grid.get_physical_coordinates_for_indices(
            voxels_on_cartesian_grid.get_min_indices()
        )[i]
        for i in [1, 2, 3]
    ]
    max_extent = [
        min_extent[i - 1]
        + voxels_on_cartesian_grid.get_grid_spacing()[i]
        * (voxels_on_cartesian_grid.get_lengths()[i] - 1)
        for i in [1, 2, 3]
    ]

    if vmax == 0:
        vmax = np.max(image)

    start_index = dim_z // 2
    l_z = axs[0].imshow(
        image[start_index, :, :],
        vmax=vmax,
        vmin=vmin,
        cmap=colourmap,
        extent=[min_extent[2], max_extent[2], max_extent[1], min_extent[1]],
    )
    if overlay is not None:
        overlay_z = axs[0].imshow(
            overlay[start_index, :, :],
            vmin=0,
            vmax=1,
            cmap="bwr",
            extent=[min_extent[2], max_extent[2], max_extent[1], min_extent[1]],
            alpha=0.5,
        )
    fig.colorbar(l_z, ax=axs[0], location="left", pad=0.2)
    # axs[0].set_title("Transverse (x-y) Plane")
    axs[0].set_title("Transverse Plane")
    if intuitive_orientation:
        axs[0].invert_xaxis()
    if labels:
        axs[0].text(
            (min_extent[2] + max_extent[2]) / 2,
            0.95 * min_extent[1] + 0.05 * max_extent[1],
            "anterior",
            alpha=0.4,
            c="white",
            horizontalalignment="center",
            verticalalignment="center",
        )
        axs[0].text(
            (min_extent[2] + max_extent[2]) / 2,
            0.05 * min_extent[1] + 0.95 * max_extent[1],
            "posterior",
            alpha=0.4,
            c="white",
            horizontalalignment="center",
            verticalalignment="center",
        )
        axs[0].text(
            0.95 * min_extent[2] + 0.05 * max_extent[2],
            (min_extent[1] + max_extent[1]) / 2,
            "R",
            alpha=0.4,
            c="white",
            horizontalalignment="center",
            verticalalignment="center",
        )
        axs[0].text(
            0.05 * min_extent[2] + 0.95 * max_extent[2],
            (min_extent[1] + max_extent[1]) / 2,
            "L",
            alpha=0.4,
            c="white",
            horizontalalignment="center",
            verticalalignment="center",
        )
    ax_slider_z = plt.axes([0.15, 0.1, 0.18, 0.03])
    slider_z = Slider(
        ax_slider_z,
        "z slice",
        0,
        dim_z - 1,
        valinit=start_index,
        valstep=1,
        initcolor="none",  # Remove the line marking the valinit position.
    )

    def update_z(val):
        l_z.set_data(image[slider_z.val, :, :])
        if overlay is not None:
            overlay_z.set_data(overlay[slider_z.val, :, :])
        fig.canvas.draw_idle()

    slider_z.on_changed(update_z)

    start_index = dim_y // 2
    l_y = axs[1].imshow(
        image[:, start_index, :],
        vmax=vmax,
        vmin=vmin,
        cmap=colourmap,
        extent=[min_extent[2], max_extent[2], max_extent[0], min_extent[0]],
    )
    if overlay is not None:
        overlay_y = axs[1].imshow(
            overlay[:, start_index, :],
            vmin=0,
            vmax=1,
            cmap="bwr",
            extent=[min_extent[2], max_extent[2], max_extent[0], min_extent[0]],
            alpha=0.5,
        )
    # axs[1].set_title("Coronal (x-z) Plane")
    axs[1].set_title("Coronal Plane")
    if intuitive_orientation:
        axs[1].invert_xaxis()
    if labels:
        axs[1].text(
            (min_extent[2] + max_extent[2]) / 2,
            0.95 * min_extent[0] + 0.05 * max_extent[0],
            "superior",
            alpha=0.4,
            c="white",
            horizontalalignment="center",
            verticalalignment="center",
        )
        axs[1].text(
            (min_extent[2] + max_extent[2]) / 2,
            0.05 * min_extent[0] + 0.95 * max_extent[0],
            "inferior",
            alpha=0.4,
            c="white",
            horizontalalignment="center",
            verticalalignment="center",
        )
        axs[1].text(
            0.95 * min_extent[2] + 0.05 * max_extent[2],
            (min_extent[0] + max_extent[0]) / 2,
            "R",
            alpha=0.4,
            c="white",
            horizontalalignment="center",
            verticalalignment="center",
        )
        axs[1].text(
            0.05 * min_extent[2] + 0.95 * max_extent[2],
            (min_extent[0] + max_extent[0]) / 2,
            "L",
            alpha=0.4,
            c="white",
            horizontalalignment="center",
            verticalalignment="center",
        )
    ax_slider_y = plt.axes([0.44, 0.1, 0.18, 0.03])
    slider_y = Slider(
        ax_slider_y,
        "y slice",
        0,
        dim_y - 1,
        valinit=start_index,
        valstep=1,
        initcolor="none",  # Remove the line marking the valinit position.
    )

    def update_y(val):
        l_y.set_data(image[:, slider_y.val, :])
        if overlay is not None:
            overlay_y.set_data(overlay[:, slider_y.val, :])
        fig.canvas.draw_idle()

    slider_y.on_changed(update_y)

    start_index = dim_x // 2
    l_x = axs[2].imshow(
        image[:, :, start_index],
        vmax=vmax,
        vmin=vmin,
        cmap=colourmap,
        extent=[min_extent[1], max_extent[1], max_extent[0], min_extent[0]],
    )
    if overlay is not None:
        overlay_x = axs[2].imshow(
            overlay[:, :, start_index],
            vmin=0,
            vmax=1,
            cmap="bwr",
            extent=[min_extent[1], max_extent[1], max_extent[0], min_extent[0]],
            alpha=0.5,
        )
    # axs[2].set_title("Sagittal (y-z) Plane")
    axs[2].set_title("Sagittal Plane")
    if intuitive_orientation:
        axs[2].invert_xaxis()
    if labels:
        axs[2].text(
            (min_extent[1] + max_extent[1]) / 2,
            0.95 * min_extent[0] + 0.05 * max_extent[0],
            "superior",
            alpha=0.4,
            c="white",
            horizontalalignment="center",
            verticalalignment="center",
        )
        axs[2].text(
            (min_extent[1] + max_extent[1]) / 2,
            0.05 * min_extent[0] + 0.95 * max_extent[0],
            "inferior",
            alpha=0.4,
            c="white",
            horizontalalignment="center",
            verticalalignment="center",
        )
        axs[2].text(
            0.95 * min_extent[1] + 0.05 * max_extent[1],
            (min_extent[0] + max_extent[0]) / 2,
            "anterior",
            rotation="vertical",
            alpha=0.4,
            c="white",
            horizontalalignment="center",
            verticalalignment="center",
        )
        axs[2].text(
            0.05 * min_extent[1] + 0.95 * max_extent[1],
            (min_extent[0] + max_extent[0]) / 2,
            "posterior",
            rotation="vertical",
            alpha=0.4,
            c="white",
            horizontalalignment="center",
            verticalalignment="center",
        )
    ax_slider_x = plt.axes([0.72, 0.1, 0.18, 0.03])
    slider_x = Slider(
        ax_slider_x,
        "x slice",
        0,
        dim_x - 1,
        valinit=start_index,
        valstep=1,
        initcolor="none",  # Remove the line marking the valinit position.
    )

    def update_x(val):
        l_x.set_data(image[:, :, slider_x.val])
        if overlay is not None:
            overlay_x.set_data(overlay[:, :, slider_x.val])
        fig.canvas.draw_idle()

    slider_x.on_changed(update_x)

    plt.show()


# function to plot ProjData
def plot_proj(proj_data_obj, vmax=0):
    seg = proj_data_obj.get_segment_by_sinogram(0)
    seg_array = stirextra.to_numpy(seg)
    # seg_array=stirextra.to_numpy(proj_data_obj);

    fig, axs = plt.subplots(1, 2, figsize=(18, 4))
    plt.subplots_adjust(bottom=0.25)

    middle_slice = seg_array.shape[0] // 2
    if vmax:
        sino = axs[0].imshow(seg_array[middle_slice, :, :], vmax=vmax)
    else:
        sino = axs[0].imshow(seg_array[middle_slice, :, :])
    axs[0].set_title("projection as sinogram")
    axs[0].set_xlabel("tangential")
    axs[0].set_ylabel("view")
    ax_slider_sino = plt.axes([0.2, 0.1, 0.2, 0.03])
    slider_sino = Slider(
        ax_slider_sino,
        "axial index",
        0,
        seg_array.shape[0] - 1,
        valinit=middle_slice,
        valstep=1,
        initcolor="none",  # Remove the line marking the valinit position.
    )

    def update_sino(val):
        sino.set_data(seg_array[slider_sino.val, :, :])
        fig.canvas.draw_idle()

    slider_sino.on_changed(update_sino)

    if vmax:
        view = axs[1].imshow(seg_array[:, 0, :], vmax=vmax)
    else:
        view = axs[1].imshow(seg_array[:, 0, :])
    axs[1].set_title("projection as viewgram")
    axs[1].set_xlabel("tangential")
    axs[1].set_ylabel("plane")
    ax_slider_view = plt.axes([0.57, 0.1, 0.33, 0.03])
    slider_view = Slider(
        ax_slider_view,
        "angular index",
        0,
        seg_array.shape[1] - 1,
        valinit=0,
        valstep=1,
        initcolor="none",  # Remove the line marking the valinit position.
    )

    def update_view(val):
        view.set_data(seg_array[:, slider_view.val, :])
        fig.canvas.draw_idle()

    slider_view.on_changed(update_view)
