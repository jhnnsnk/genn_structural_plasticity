import matplotlib

plot_dict = {
    "figure_width": 15./2.54,  # 15cm
    "color_elimination": "#228833",  # green
    "color_formation": "#DDAA33",  # yellow
    "SMALL_SIZE": 8,
    "MEDIUM_SIZE": 10,
    "BIGGER_SIZE": 12,
}


def get_dx_dy_pbc(xpre, xpost, ypre, ypost, grid_num_x, grid_num_y):
    """
    Displacement in x- and y-direction
    """
    dx = xpost - xpre
    dy = ypost - ypre

    if dx > 0.5 * grid_num_x:
        dx -= grid_num_x
    if dx <= -0.5 * grid_num_x:
        dx += grid_num_x

    if dy > 0.5 * grid_num_y:
        dy -= grid_num_y
    if dy <= -0.5 * grid_num_y:
        dy += grid_num_y

    return dx, dy


def get_dx_dy_pbc_from_ids(pre_id, post_id, grid_num_x, grid_num_y):
    xpre = pre_id % grid_num_x
    xpost = post_id % grid_num_x
    ypre = int(pre_id / grid_num_y)
    ypost = int(post_id / grid_num_y)

    dx, dy = get_dx_dy_pbc(xpre, xpost, ypre, ypost, grid_num_x, grid_num_y)
    return dx, dy


def add_label(ax, label, offset=[0, 0],
              weight='bold', fontsize_scale=1.2):
    """
    Adds label to axis with given offset.

    Parameters
    ----------
    ax
        Axis to add label to.
    label
        Label should be a letter.
    offset
        x-,y-Offset.
    weight
        Weight of font.
    fontsize_scale
        Scaling factor for font size.
    """
    label_pos = [0. + offset[0], 1. + offset[1]]
    ax.text(label_pos[0], label_pos[1], label,
            ha='left', va='bottom',
            transform=ax.transAxes,
            weight=weight,
            fontsize=matplotlib.rcParams['font.size'] * fontsize_scale)
    return
