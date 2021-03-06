#!/usr/bin/env python

"""@package docstring
File: graphs.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: File containing modular graphing functions for Fokker-Planck data.
"""

import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import (Circle, RegularPolygon, FancyArrowPatch,
                                ArrowStyle)
# import matplotlib.pyplot as plt


def convert_size_units(d, ax, reference='y'):
    """
    Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    d: float
        Linewidth in points
    """
    fig = ax.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * ax.get_position().width
        value_range = np.diff(ax.get_xlim())[0]
    elif reference == 'y':
        length = fig.bbox_inches.height * ax.get_position().height
        value_range = np.diff(ax.get_ylim())[0]
    # Convert length to points
    length *= 72
    # Scale linewidth to value range
    return d * (length / value_range)


def xlink_end_pos(r_vec, u_vec, s):
    """!Get spatial location of a xlink end using rod position and orientation.

    @param r_vec: Position vector of rods center
    @param u_vec: Orientation unit vector of rod
    @param s: Location of xlink end with respect to center of rod. Can be negative.
    @return: Position of xlink end in system
    """
    return (r_vec + (u_vec * s))


def get_max_min_ends(r_i, r_j, u_i, u_j, L_i, L_j):
    """!Get the maximum and minimum end position value in a direction for two
    rods.

    @param r_i: Array of rod i center positions
    @param r_j: Array of rod j center positions
    @param u_i: Array of rod i orientation unit vectors
    @param u_j: Array of rod j orientation unit vectors
    @param L_i: Length of rod i
    @param L_j: Length of rod j
    @return: List of all possible maximums and minimum rod end positions for
    rods i and j. Both plus and minus rod ends are considered.

    """
    return [np.amax(0.5 * L_i * u_i + r_i), np.amin(0.5 * L_i * u_i + r_i),
            np.amax(-.5 * L_i * u_i + r_i), np.amin(-.5 * L_i * u_i + r_i),
            np.amax(0.5 * L_j * u_j + r_j), np.amin(0.5 * L_j * u_j + r_j),
            np.amax(-.5 * L_j * u_j + r_j), np.amin(-.5 * L_j * u_j + r_j)]


class LineDataUnits(Line2D):

    """!Class that rescales a 2D matplotlib line to have the proper width and
    length with respect to axis unit values.
    """

    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72. / self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data)) - trans((0, 0))) * ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)


def draw_rod(ax, r_vec, u_vec, L, rod_diam, color='tab:green', tip_color='b'):
    """!Draw a diagramitic representation of a rod on a matplotlib axis object.

    @param ax: Matplotlib axis object
    @param r_vec: Position vector of rod's center
    @param u_vec: Orientation unit vector of rod
    @param L: Length of rod
    @param rod_diam: Diameter of rod
    @param color: Color of rod body
    @param tip_color: Color of plus end of rod
    @return: None

    """
    line = LineDataUnits((r_vec[1] - .5 * L * u_vec[1],
                          r_vec[1] + .5 * L * u_vec[1]),
                         (r_vec[2] - .5 * L * u_vec[2],
                          r_vec[2] + .5 * L * u_vec[2]),
                         linewidth=rod_diam, solid_capstyle='round',
                         color=color, clip_on=False, )

    tip = Circle((r_vec[1] + .5 * L * u_vec[1], r_vec[2] + .5 * L * u_vec[2]),
                 .5 * rod_diam, color=tip_color, zorder=3)
    ax.add_patch(tip)
    ax.add_line(line)


def draw_xlink(ax, e_i, e_j, lw=10, color='k', alpha=.5):
    """!Draw a diagramitic representation of an xlink density on a matplotlib
    axis object.
    @param ax: Matplotlib axis object
    @param e_i: End of xlink on rod i
    @param e_j: End of xlink on rod j
    @param lw: Width of line representing xlink density
    @param color: Color of line representing xlink density
    @param alpha: Transparency of line representing xlink density
    return: None
    """
    line = LineDataUnits((e_i[1], e_j[1]), (e_i[2], e_j[2]),
                         linewidth=lw,  # solid_capstyle='round',
                         color=color, clip_on=False, alpha=alpha)
    ax.add_line(line)


def draw_moment_rod(ax, r_vec, u_vec, L, rod_diam,
                    mu00, mu10, mu20, num_max=50):
    """!Draw a diagramitic representation of a rod and moments of xlink end
    density on rod.

    @param ax: Matplotlib axis object
    @param r_vec: Position vector of rod's center
    @param u_vec: Orientation unit vector of rod
    @param L: Length of rod
    @param rod_diam: Diameter of rod
    @param mu00: Zeroth moment of xlink density (respresented as rod color)
    @param mu10: First moment of xlink density corresponding to average end
    position on rod (represented by position of polygon)
    @param mu20: First moment of xlink density corresponding to variance of end
    position on rod with respect to rod center (used to calculate sigma)
    @param num_max: Maximum number of xlinks to set standard colormap
    @return: colorbar set by num_max

    """
    cb = mpl.cm.ScalarMappable(
        mpl.colors.Normalize(0, num_max), 'viridis')
    draw_rod(ax, r_vec, u_vec, L, rod_diam)
    scaled_mu10 = mu10 / mu00 if mu00 else 0
    mu10_loc = RegularPolygon((r_vec[1] + scaled_mu10 * u_vec[1],
                               r_vec[2] + scaled_mu10 * u_vec[2]),
                              5, rod_diam, color=cb.to_rgba(mu00), zorder=4)
    variance = (mu20 / mu00) - (scaled_mu10**2) if mu00 > 1e-3 else 0.

    sigma_dist = np.sqrt(variance) if variance >= 1e-3 else 0.
    # mu20_ellipse = Ellipse((r_vec[1], r_vec[2]), mu20_dist*2., rod_diam,
    # angle=np.arctan(u_vec[2]/u_vec[1]), zorder=4, fill=False)
    sigma_bar = FancyArrowPatch(
        (r_vec[1] + (scaled_mu10 - sigma_dist) * u_vec[1],
         r_vec[2] + (scaled_mu10 - sigma_dist) * u_vec[2]),
        (r_vec[1] + (scaled_mu10 + sigma_dist) * u_vec[1],
         r_vec[2] + (scaled_mu10 + sigma_dist) * u_vec[2]),
        arrowstyle=ArrowStyle('|-|',
                              widthA=convert_size_units(.5 * rod_diam, ax),
                              widthB=convert_size_units(.5 * rod_diam, ax)),
        zorder=3)
    ax.add_patch(sigma_bar)
    ax.add_patch(mu10_loc)
    return cb


def graph_vs_time(ax, time, y, n=-1, color='b', fillstyle='full'):
    """!TODO: Docstring for graph_vs_t.

    @param ax: TODO
    @param time: TODO
    @param y: TODO
    @param n: TODO
    @return: TODO

    """
    s = ax.plot(time[:n], y[:n], c=color, marker='o',
                fillstyle=fillstyle, linestyle='')
    return s


def graph_xl_dens(ax, psi, s_i, s_j, **kwargs):
    """!Graph an instance in time of the crosslinker density for the PDE

    @param psi: crosslinker density
    @param **kwargs: TODO
    @return: TODO

    """
    s_i = np.asarray(s_i)
    s_j = np.asarray(s_j)
    psi = np.transpose(np.asarray(psi))
    if "max_dens_val" in kwargs:
        max_val = kwargs["max_dens_val"]
        c = ax.pcolormesh(s_i, s_j, psi, vmin=0, vmax=max_val)
    else:
        c = ax.pcolormesh(s_i, s_j, psi)
    return c


def graph_2d_rod_diagram(ax, anal, n=-1):
    """!TODO: Docstring for graph_2d_rod_diagram.

    @param ax: TODO
    @param anal: TODO
    @param n: TODO
    @return: TODO

    """
    params = anal._params
    L_i = params["L1"]
    L_j = params["L2"]
    lw = params['rod_diameter']
    if hasattr(anal, 'phi_arr') and not hasattr(anal, 'R1_vec'):
        hphi = anal.phi_arr[n] * .5
        line1 = LineDataUnits((0, L_i * np.cos(hphi)),
                              (0, L_i * np.sin(hphi)),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:green', clip_on=False)
        line2 = LineDataUnits((0, L_j * np.cos(hphi)),
                              (0, -L_j * np.sin(hphi)),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:purple', clip_on=False)
        ax.add_line(line1)
        ax.add_line(line2)
    elif hasattr(anal, 'R_arr'):
        r = anal.R_arr[n, :]
        line1 = LineDataUnits((-.5 * L_i, .5 * L_i),
                              (0, 0),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:green', clip_on=False)
        line2 = LineDataUnits((-.5 * L_i + r[0], .5 * L_i + r[0]),
                              (r[1], r[1]),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:purple', clip_on=False)
        ax.add_line(line1)
        ax.add_line(line2)
    else:
        r_i_arr = anal.R1_pos
        r_j_arr = anal.R2_pos
        u_i_arr = anal.R1_vec
        u_j_arr = anal.R2_vec

        draw_rod(ax, r_i_arr[n], u_i_arr[n], L_i, lw, color='tab:green')
        draw_rod(ax, r_j_arr[n], u_j_arr[n], L_j, lw, color='tab:purple')

        # if anal.OT1_pos is not None:
        #     ot1 = Circle((anal.OT1_pos[n, 1], anal.OT1_pos[n, 2]),
        #                  3 * lw, color='y', alpha=.5)
        #     mtip1 = Circle((-.5 * L_i * u1[1] + r_i[1], -.5 * L_i * u1[2] + r_i[2]),
        #                    lw, color='b', zorder=4)
        #     ax.add_patch(ot1)
        #     ax.add_patch(mtip1)

        # if anal.OT2_pos is not None:
        #     ot2 = Circle((anal.OT2_pos[n, 1], anal.OT2_pos[n, 2]),
        #                  3 * lw, color='y', alpha=.5)
        #     mtip2 = Circle((-.5 * L_j * u2[1] + r_j[1], -.5 * L_j * u2[2] + r_j[2]),
        #                    lw, color='b', zorder=4)
        #     ax.add_patch(ot2)
        #     ax.add_patch(mtip2)

        # Get all extreme positions of tips in the first dimension to maintain
        # consistent graphing size
        x_ends = get_max_min_ends(
            r_i_arr[:, 1], r_j_arr[:, 1], u_i_arr[:, 1], u_j_arr[:, 1], L_i, L_j)
        # Get all extreme positions of tips in the second dimension to maintain
        # consistent graphing size
        y_ends = get_max_min_ends(
            r_i_arr[:, 2], r_j_arr[:, 2], u_i_arr[:, 2], u_j_arr[:, 2], L_i, L_j)

        max_x = max(x_ends + y_ends)
        max_x = max_x * 1.25 if max_x > 0 else .75 * max_x
        min_x = min(x_ends + y_ends)
        min_x = min_x * 1.25 if min_x < 0 else .75 * min_x

        # Make a square box always
        max_y = max_x
        min_y = min_x

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_xlabel(r'x (nm)')
        ax.set_ylabel(r'y (nm)')
        # labels = ["fil$_i$", "fil$_j$", "Plus-end"]
        # if anal.OT1_pos is not None or anal.OT2_pos is not None:
        #     labels += ["Optical trap", "Bead"]
        # ax.legend(labels, loc="upper right")


def graph_2d_rod_pde_diagram(ax, anal, n=-1, scale=50):
    """!TODO: Docstring for graph_2d_rod_diagram.

    @param ax: TODO
    @param anal: TODO
    @param n: TODO
    @return: TODO

    """
    graph_2d_rod_diagram(ax, anal, n)
    L_i = anal._params["L1"]
    L_j = anal._params["L2"]
    rod_diam = anal._params['rod_diameter']
    r_i = anal.R1_pos[n]
    r_j = anal.R2_pos[n]
    u_i = anal.R1_vec[n]
    u_j = anal.R2_vec[n]
    xl_distr = anal.xl_distr[:, :, n]

    N, M = int(L_i / rod_diam) + 1, int(L_j / rod_diam) + 1
    a, b = int(xl_distr.shape[0] / N), int(xl_distr.shape[1] / M)

    # s_i = anal.s_i.reshape(N, a).mean(axis=1)
    # s_j = anal.s_j.reshape(M, b).mean(axis=1)
    s_i = np.arange(-.5 * (L_i + rod_diam), .5 * (L_i + rod_diam), rod_diam)
    s_j = np.arange(-.5 * (L_j + rod_diam), .5 * (L_j + rod_diam), rod_diam)
    xl_distr_coarse = xl_distr[:a * N, :b * M].reshape(N, a, M, b)
    xl_distr_coarse = xl_distr_coarse.sum(axis=(1, 3))
    for index, val in np.ndenumerate(xl_distr_coarse):
        e_i = xlink_end_pos(r_i, u_i, s_i[index[0]])
        e_j = xlink_end_pos(r_j, u_j, s_j[index[1]])
        # print(e_i, e_j)
        draw_xlink(ax, e_i, e_j, color='r',
                   alpha=np.clip(val * scale / (a * b), 0, 1))


def graph_2d_rod_moment_diagram(ax, anal, n=-1):
    """!TODO: Docstring for graph_2d_rod_diagram.

    @param ax: TODO
    @param anal: TODO
    @param n: TODO
    @return: TODO

    """
    params = anal._params
    L_i = params["L1"]
    L_j = params["L2"]
    rod_diam = params['rod_diameter']
    r_i_arr = anal.R1_pos
    r_j_arr = anal.R2_pos
    u_i_arr = anal.R1_vec
    u_j_arr = anal.R2_vec
    mu00_max = np.amax(anal.mu00)

    # Get all extreme positions of tips in the first dimension to maintain
    # consistent graphing size
    x_ends = get_max_min_ends(
        r_i_arr[:, 1], r_j_arr[:, 1], u_i_arr[:, 1], u_j_arr[:, 1], L_i, L_j)
    # Get all extreme positions of tips in the second dimension to maintain
    # consistent graphing size
    y_ends = get_max_min_ends(
        r_i_arr[:, 2], r_j_arr[:, 2], u_i_arr[:, 2], u_j_arr[:, 2], L_i, L_j)

    max_x = max(x_ends + y_ends)
    max_x = max_x * 1.25 if max_x > 0 else .75 * max_x
    min_x = min(x_ends + y_ends)
    min_x = min_x * 1.25 if min_x < 0 else .75 * min_x

    # Make a square box always
    max_y = max_x
    min_y = min_x

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_xlabel(r'x (nm)')
    ax.set_ylabel(r'y (nm)')

    cb = draw_moment_rod(ax, r_i_arr[n], u_i_arr[n], L_i, rod_diam,
                         anal.mu00[n], anal.mu10[n], anal.mu20[n],
                         num_max=mu00_max)
    cb = draw_moment_rod(ax, r_j_arr[n], u_j_arr[n], L_j, rod_diam,
                         anal.mu00[n], anal.mu01[n], anal.mu02[n],
                         num_max=mu00_max)

    # labels = ["fil$_i$", "fil$_j$", "Plus-end", r"$\mu^{{10}}$", r"$\mu^{{20}}$"]
    # if anal.OT1_pos is not None or anal.OT2_pos is not None:
    #     labels += ["Optical trap", "Bead"]
    # ax.legend(labels, loc="upper right")
    return cb


def me_graph_min_data_2d(fig, axarr, n, me_anal):
    # Clean up if lines on axis object to speed up movie making
    if not me_anal.init_flag:
        for ax in axarr.flatten():
            ax.clear()
        for artist in fig.gca().lines + fig.gca().collections:
            artist.remove()
            del artist

    cb = graph_2d_rod_moment_diagram(axarr[0], me_anal, n)
    cb1 = graph_xl_dens(axarr[1],
                        me_anal.xl_distr[:, :, n],
                        me_anal.s_i,
                        me_anal.s_j,
                        max_dens_val=me_anal.max_dens_val)

    axarr[1].set_xlabel(
        'Head distance from \n center of fil$_i$ $s_i$ (nm)')
    axarr[1].set_ylabel(
        'Head distance from \n center of fil$_j$ $s_j$ (nm)')

    if me_anal.init_flag:
        axarr[0].set_aspect(1.0)
        axarr[1].set_aspect(1.0)
        cbar = fig.colorbar(cb, ax=axarr[0])
        cbar.set_label(
            r'Motor number $\langle N_{i,j} \rangle$')
        cbar1 = fig.colorbar(cb1, ax=axarr[1])
        cbar1.set_label(
            r'Reconstructed motor density $\psi_{i,j}$')
    me_anal.init_flag = False

    axarr[0].text(.05, .90, "Time = {:.2f} sec".format(me_anal.time[n]),
                  horizontalalignment='left',
                  verticalalignment='bottom',
                  transform=axarr[0].transAxes)
    return fig.gca().lines + fig.gca().collections


def me_graph_all_data_2d(fig, axarr, n, me_anal):
    # Clean up if lines on axis object to speed up movie making
    if not me_anal.init_flag:
        for ax in axarr.flatten():
            ax.clear()
        for artist in fig.gca().lines + fig.gca().collections:
            artist.remove()
            del artist

    axarr[1].set_xlabel(r'Time (sec)')
    axarr[1].set_ylabel('Distance between fils \n centers of mass (nm)')
    axarr[1].set_xlim(left=0, right=me_anal.time[-1])
    axarr[1].set_ylim(np.amin(me_anal.dR_arr),
                      np.amax(me_anal.dR_arr))

    axarr[2].set_xlabel(r'Time (sec)')
    axarr[2].set_ylabel('Angle between fil \n orientation vectors (rad)')
    axarr[2].set_xlim(left=0, right=me_anal.time[-1])
    axarr[2].set_ylim(np.nanmin(me_anal.phi_arr),
                      np.nanmax(me_anal.phi_arr))

    axarr[3].set_xlabel(r'Time (sec)')
    axarr[3].set_ylabel(r'Motor number')
    axarr[3].set_xlim(left=0, right=me_anal.time[-1])
    axarr[3].set_ylim(np.amin(me_anal.mu00),
                      np.amax(me_anal.mu00))

    p_n = np.stack((me_anal.mu10, me_anal.mu01))
    axarr[4].set_xlabel(r'Time (sec)')
    axarr[4].set_ylabel(r'First moments (nm)')
    axarr[4].set_xlim(left=0, right=me_anal.time[-1])
    axarr[4].set_ylim(np.amin(p_n), np.amax(p_n))

    # mu_kl = me_anal._h5_data['/xl_data/second_moments'][...]
    mu_kl = np.stack((me_anal.mu11, me_anal.mu20, me_anal.mu02))
    axarr[5].set_xlabel(r'Time (sec)')
    axarr[5].set_ylabel(r'Second moments (nm$^2$)')
    axarr[5].set_xlim(left=0, right=me_anal.time[-1])
    axarr[5].set_ylim(np.amin(mu_kl), np.amax(mu_kl))

    # Draw rods
    if me_anal.graph_type == 'min':
        graph_2d_rod_diagram(axarr[0], me_anal, n)
    else:
        cb = graph_2d_rod_moment_diagram(axarr[0], me_anal, n)

    if me_anal.init_flag:
        axarr[0].set_aspect(1.0)
        if me_anal.graph_type == 'all':
            fig.colorbar(cb, ax=axarr[0])

    me_anal.init_flag = False

    # Graph rod center separations
    graph_vs_time(axarr[1], me_anal.time, me_anal.dR_arr, n)
    # Graph angle between rod orientations
    graph_vs_time(axarr[2], me_anal.time, me_anal.phi_arr, n)

    # Graph zeroth moment aka number of crosslinkers
    graph_vs_time(axarr[3], me_anal.time, me_anal.mu00, n)

    # Graph first moments of crosslink distribution
    graph_vs_time(axarr[4], me_anal.time, me_anal.mu10, n,
                  color='tab:green')
    graph_vs_time(axarr[4], me_anal.time, me_anal.mu01, n,
                  color='tab:purple')

    # Graph second moments of crosslinker distribution
    graph_vs_time(axarr[5], me_anal.time, me_anal.mu11, n,
                  color='b')
    graph_vs_time(axarr[5], me_anal.time, me_anal.mu20, n,
                  color='tab:green')
    graph_vs_time(axarr[5], me_anal.time, me_anal.mu02, n,
                  color='tab:purple')

    # # Effective moment graphing
    # ## Zeroth moment
    # graph_vs_time(axarr[3], me_anal.time,
    #               me_anal.mu_kl_eff[:, 0], n, fillstyle='none')
    # ## First moments
    # graph_vs_time(axarr[4], me_anal.time, me_anal.mu_kl_eff[:, 1], n,
    #               color='tab:green', fillstyle='none')
    # graph_vs_time(axarr[4], me_anal.time, me_anal.mu_kl_eff[:, 2], n,
    #               color='tab:purple', fillstyle='none')
    # ## Second moments
    # graph_vs_time(axarr[5], me_anal.time, me_anal.mu_kl_eff[:, 3], n,
    #               color='b', fillstyle='none')
    # graph_vs_time(axarr[5], me_anal.time, me_anal.mu_kl_eff[:, 4], n,
    #               color='tab:green', fillstyle='none')
    # graph_vs_time(axarr[5], me_anal.time, me_anal.mu_kl_eff[:, 5], n,
    #               color='tab:purple', fillstyle='none')

    # Legend information
    axarr[1].legend([r"$\Delta$R({:.2f}) = {:.1f} nm".format(
        me_anal.time[n], me_anal.dR_arr[n])])
    axarr[2].legend([r"$\phi$({:.2f}) = {:.1f} rad".format(
        me_anal.time[n], me_anal.phi_arr[n])])
    axarr[3].legend([r"N({:.2f})={:.1f}".format(
        me_anal.time[n], me_anal.mu00[n])])
    axarr[4].legend([r"$\mu^{{1,0}}$({:.2f}) = {:.1f}".format(me_anal.time[n],
                                                              me_anal.mu10[n]),
                     r"$\mu^{{0,1}}$({:.2f}) = {:.1f}".format(me_anal.time[n],
                                                              me_anal.mu01[n])])
    axarr[5].legend([r"$\mu^{{1,1}}$({:.2f}) = {:.1f}".format(me_anal.time[n],
                                                              me_anal.mu11[n]),
                     r"$\mu^{{2,0}}$({:.2f}) = {:.1f}".format(me_anal.time[n],
                                                              me_anal.mu20[n]),
                     r"$\mu^{{0,2}}$({:.2f}) = {:.1f}".format(me_anal.time[n],
                                                              me_anal.mu02[n])])
    return fig.gca().lines + fig.gca().collections


def me_graph_distr_data_2d(fig, axarr, n, me_anal):
    # Clean up if lines on axis object to speed up movie making
    if not me_anal.init_flag:
        for ax in axarr.flatten():
            ax.clear()
        for artist in fig.gca().lines + fig.gca().collections:
            artist.remove()
            del artist

    # Draw rods
    graph_2d_rod_diagram(axarr[0], me_anal, n)

    cb1 = graph_xl_dens(axarr[1],
                        me_anal.xl_distr[:, :, n],
                        me_anal.s_i,
                        me_anal.s_j,
                        max_dens_val=me_anal.max_dens_val)

    # Graph rod center separations
    axarr[3].set_xlabel(r'Time (sec)')
    axarr[3].set_ylabel('Distance between fils \n centers of mass (nm)')
    axarr[3].set_xlim(left=0, right=me_anal.time[-1])
    axarr[3].set_ylim(np.amin(me_anal.dR_arr),
                      np.amax(me_anal.dR_arr))
    graph_vs_time(axarr[3], me_anal.time, me_anal.dR_arr, n)

    # Graph angle between rod orientations
    axarr[4].set_xlabel(r'Time (sec)')
    axarr[4].set_ylabel('Angle between fil \n orientation vectors (rad)')
    axarr[4].set_xlim(left=0, right=me_anal.time[-1])
    axarr[4].set_ylim(np.nanmin(me_anal.phi_arr),
                      np.nanmax(me_anal.phi_arr))
    graph_vs_time(axarr[4], me_anal.time, me_anal.phi_arr, n)

    # Graph zeroth moment aka number of crosslinkers
    axarr[2].set_xlabel(r'Time (sec)')
    axarr[2].set_ylabel(r'Motor number')
    axarr[2].set_xlim(left=0, right=me_anal.time[-1])
    axarr[2].set_ylim(np.amin(me_anal.mu00),
                      np.amax(me_anal.mu00))
    graph_vs_time(axarr[2], me_anal.time, me_anal.mu00, n)
    if me_anal._params['ODE_type'] == 'zrl_bvg':
        graph_vs_time(axarr[2], me_anal.time, me_anal.mu_kl_eff[:, 0],
                      n, fillstyle='none')

    # Graph first moments of crosslink distribution
    p_n = np.stack((me_anal.mu10, me_anal.mu01))
    axarr[5].set_xlabel(r'Time (sec)')
    axarr[5].set_ylabel(r'First moments (nm)')
    axarr[5].set_xlim(left=0, right=me_anal.time[-1])
    axarr[5].set_ylim(np.amin(p_n), np.amax(p_n))
    graph_vs_time(axarr[5], me_anal.time, me_anal.mu10, n,
                  color='tab:green')
    graph_vs_time(axarr[5], me_anal.time, me_anal.mu01, n,
                  color='tab:purple')
    if me_anal._params['ODE_type'] == 'zrl_bvg':
        graph_vs_time(axarr[5], me_anal.time, me_anal.mu_kl_eff[:, 1], n,
                      color='tab:green', fillstyle='none')
        graph_vs_time(axarr[5], me_anal.time, me_anal.mu_kl_eff[:, 2], n,
                      color='tab:purple', fillstyle='none')

    # Graph second moments of crosslinker distribution
    mu_kl = np.stack((me_anal.mu11, me_anal.mu20, me_anal.mu02))
    axarr[8].set_xlabel(r'Time (sec)')
    axarr[8].set_ylabel(r'Second moments (nm$^2$)')
    axarr[8].set_xlim(left=0, right=me_anal.time[-1])
    axarr[8].set_ylim(np.amin(mu_kl), np.amax(mu_kl))
    graph_vs_time(axarr[8], me_anal.time, me_anal.mu11, n,
                  color='b')
    graph_vs_time(axarr[8], me_anal.time, me_anal.mu20, n,
                  color='tab:green')
    graph_vs_time(axarr[8], me_anal.time, me_anal.mu02, n,
                  color='tab:purple')
    if me_anal._params['ODE_type'] == 'zrl_bvg':
        graph_vs_time(axarr[8], me_anal.time, me_anal.mu_kl_eff[:, 3], n,
                      color='b', fillstyle='none')
        graph_vs_time(axarr[8], me_anal.time, me_anal.mu_kl_eff[:, 4], n,
                      color='tab:green', fillstyle='none')
        graph_vs_time(axarr[8], me_anal.time, me_anal.mu_kl_eff[:, 5], n,
                      color='tab:purple', fillstyle='none')

    if me_anal.init_flag:
        axarr[0].set_aspect(1.0)
        axarr[1].set_aspect(1.0)
        fig.colorbar(cb1, ax=axarr[1])
    me_anal.init_flag = False

    # Legend information
    # axarr[1].legend([r"$\Delta$R({:.2f}) = {:.1f} nm".format(
    #     me_anal.time[n], me_anal.dR_arr[n])])
    # axarr[2].legend([r"$\phi$({:.2f}) = {:.1f} rad".format(
    #     me_anal.time[n], me_anal.phi_arr[n])])
    # axarr[3].legend([r"N({:.2f})={:.1f}".format(
    #     me_anal.time[n], me_anal.mu00[n])])
    # axarr[4].legend([r"$\mu^{{1,0}}$({:.2f}) = {:.1f}".format(me_anal.time[n],
    #                                                           me_anal.mu10[n]),
    #                  r"$\mu^{{0,1}}$({:.2f}) = {:.1f}".format(me_anal.time[n],
    #                                                           me_anal.mu01[n])])
    # axarr[5].legend([r"$\mu^{{1,1}}$({:.2f}) = {:.1f}".format(me_anal.time[n],
    #                                                           me_anal.mu11[n]),
    #                  r"$\mu^{{2,0}}$({:.2f}) = {:.1f}".format(me_anal.time[n],
    #                                                           me_anal.mu20[n]),
    #                  r"$\mu^{{0,2}}$({:.2f}) = {:.1f}".format(me_anal.time[n],
    #                                                           me_anal.mu02[n])])
    return fig.gca().lines + fig.gca().collections


def pde_graph_all_data_2d(fig, axarr, n, pde_anal):
    # Clean up if lines
    if not pde_anal.init_flag:
        for ax in axarr.flatten():
            ax.clear()
        for artist in fig.gca().lines + fig.gca().collections:
            artist.remove()
            del artist

    # Init axis labels and ranges
    axarr[1].set_xlabel(
        'Head distance from \n center of fil$_i$ $s_i$ (nm)')
    axarr[1].set_ylabel(
        'Head distance from \n center of fil$_j$ $s_j$ (nm)')

    axarr[2].set_xlabel(r'Time (sec)')
    axarr[2].set_ylabel(r'Motor number')
    axarr[2].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[2].set_ylim(np.amin(pde_anal.mu00), np.amax(pde_anal.mu00))

    axarr[3].set_xlabel(r'Time (sec)')
    axarr[3].set_ylabel(r'Motor force (pN)')
    axarr[3].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[3].set_ylim(np.amin(pde_anal.force_arr), np.amax(pde_anal.force_arr))

    axarr[4].set_xlabel(r'Time (sec)')
    axarr[4].set_ylabel(r'Motor torque (pN*nm)')
    axarr[4].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[4].set_ylim(np.amin(pde_anal.torque_arr),
                      np.amax(pde_anal.torque_arr))

    axarr[5].set_xlabel(r'Time (sec)')
    axarr[5].set_ylabel(r'First moments (nm)')
    axarr[5].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[5].set_ylim(min(np.amin(pde_anal.mu10), np.amin(pde_anal.mu01)),
                      max(np.amax(pde_anal.mu10), np.amax(pde_anal.mu01)))

    axarr[6].set_xlabel(r'Time (sec)')
    axarr[6].set_ylabel('Distance between fils \n centers of mass (nm)')
    axarr[6].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[6].set_ylim(np.amin(pde_anal.dR_arr),
                      np.amax(pde_anal.dR_arr))

    axarr[7].set_xlabel(r'Time (sec)')
    axarr[7].set_ylabel('Angle between fil \n orientation vectors (rad)')
    axarr[7].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[7].set_ylim(np.nanmin(pde_anal.phi_arr),
                      np.nanmax(pde_anal.phi_arr))

    axarr[8].set_xlabel(r'Time (sec)')
    axarr[8].set_ylabel(r'Second moments (nm$^2$)')
    axarr[8].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[8].set_ylim(min(np.amin(pde_anal.mu11),
                          np.amin(pde_anal.mu20),
                          np.amin(pde_anal.mu02)),
                      max(np.amax(pde_anal.mu11),
                          np.amax(pde_anal.mu20),
                          np.amax(pde_anal.mu02)))

    # Draw rods
    graph_2d_rod_diagram(axarr[0], pde_anal, n)

    # Make crosslinker density plot
    c = graph_xl_dens(axarr[1],
                      pde_anal.xl_distr[:, :, n],
                      pde_anal.s_i,
                      pde_anal.s_j,
                      max_dens_val=pde_anal.max_dens_val)
    if pde_anal.init_flag:
        axarr[0].set_aspect(1.0)
        axarr[1].set_aspect(1.0)
        fig.colorbar(c, ax=axarr[1])
        pde_anal.init_flag = False

    # Graph zeroth moment aka number of crosslinkers
    graph_vs_time(axarr[2], pde_anal.time, pde_anal.mu00, n)
    # Graph forces
    graph_vs_time(axarr[3], pde_anal.time, pde_anal.force_arr[:, 0], n,
                  color='tab:green')
    graph_vs_time(axarr[3], pde_anal.time, pde_anal.force_arr[:, 1], n,
                  color='tab:purple')
    # Graph torques
    graph_vs_time(axarr[4], pde_anal.time, pde_anal.torque_arr[:, 0], n,
                  color='tab:green')
    graph_vs_time(axarr[4], pde_anal.time, pde_anal.torque_arr[:, 1], n,
                  color='tab:purple')
    # Graph first moments of crosslink distribution
    graph_vs_time(axarr[5], pde_anal.time, pde_anal.mu10, n,
                  color='tab:green')
    graph_vs_time(axarr[5], pde_anal.time, pde_anal.mu01, n,
                  color='tab:purple')
    # Graph rod center separations
    graph_vs_time(axarr[6], pde_anal.time, pde_anal.dR_arr, n)
    # Graph angle between rod orientations
    graph_vs_time(axarr[7], pde_anal.time, pde_anal.phi_arr, n)
    # Graph second moments of crosslinker distribution
    graph_vs_time(axarr[8], pde_anal.time, pde_anal.mu11, n,
                  color='b')
    graph_vs_time(axarr[8], pde_anal.time, pde_anal.mu20, n,
                  color='tab:green')
    graph_vs_time(axarr[8], pde_anal.time, pde_anal.mu02, n,
                  color='tab:purple')

    # Legend information
    axarr[2].legend(["N({:.2f}) = {:.1f}".format(
        pde_anal.time[n], pde_anal.mu00[n])])
    axarr[3].legend([r"F$_i$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                      pde_anal.force_arr[n, 0]),
                     r"F$_j$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                      pde_anal.force_arr[n, 1])])
    axarr[4].legend([r"$T_i$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                      pde_anal.torque_arr[n, 0]),
                     r"$T_j$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                      pde_anal.torque_arr[n, 1])])
    axarr[5].legend([r"$\mu^{{1,0}}$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                              pde_anal.mu10[n]),
                     r"$\mu^{{0,1}}$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                              pde_anal.mu01[n])])
    axarr[6].legend([r"$\Delta$R({:.2f}) = {:.1f} nm".format(
        pde_anal.time[n], pde_anal.dR_arr[n])])
    axarr[7].legend([r"$\phi$({:.2f}) = {:.1f} rad".format(
        pde_anal.time[n], pde_anal.phi_arr[n])])
    axarr[8].legend([r"$\mu^{{1,1}}$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                              pde_anal.mu11[n]),
                     r"$\mu^{{2,0}}$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                              pde_anal.mu20[n]),
                     r"$\mu^{{0,2}}$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                              pde_anal.mu02[n])])
    return fig.gca().lines + fig.gca().collections


def pde_graph_moment_data_2d(fig, axarr, n, pde_anal):
    # Clean up if lines
    if not pde_anal.init_flag:
        for ax in axarr.flatten():
            ax.clear()
        for artist in fig.gca().lines + fig.gca().collections:
            artist.remove()
            del artist

    # Init axis labels and ranges
    axarr[1].set_xlabel(
        'Head distance from \n center of fil$_i$ $s_i$ (nm)')
    axarr[1].set_ylabel(
        'Head distance from \n center of fil$_j$ $s_j$ (nm)')

    axarr[2].set_xlabel(r'Time (sec)')
    axarr[2].set_ylabel(r'Motor number')
    axarr[2].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[2].set_ylim(np.amin(pde_anal.mu00),
                      np.amax(pde_anal.mu00))

    axarr[3].set_xlabel(r'Time (sec)')
    axarr[3].set_ylabel(r'First moments (nm)')
    axarr[3].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[3].set_ylim(min(np.amin(pde_anal.mu10), np.amin(pde_anal.mu01)),
                      max(np.amax(pde_anal.mu10), np.amax(pde_anal.mu01)))

    axarr[4].set_xlabel(r'Time (sec)')
    axarr[4].set_ylabel(r'Second moments (nm$^2$)')
    axarr[4].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[4].set_ylim(min(np.amin(pde_anal.mu11),
                          np.amin(pde_anal.mu20),
                          np.amin(pde_anal.mu02)),
                      max(np.amax(pde_anal.mu11),
                          np.amax(pde_anal.mu20),
                          np.amax(pde_anal.mu02)))

    axarr[5].set_xlabel(r'Time (sec)')
    axarr[5].set_ylabel(r'Motor force (pN)')
    axarr[5].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[5].set_ylim(np.amin(pde_anal.force_arr),
                      np.amax(pde_anal.force_arr))

    axarr[6].set_xlabel(r'Time (sec)')
    axarr[6].set_ylabel(r'Motor torque (pN$\cdot$nm)')
    axarr[6].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[6].set_ylim(np.amin(pde_anal.torque_arr),
                      np.amax(pde_anal.torque_arr))

    # Draw rods
    graph_2d_rod_diagram(axarr[0], pde_anal, n)

    # Make crosslinker density plot
    c = graph_xl_dens(axarr[1],
                      pde_anal.xl_distr[:, :, n],
                      pde_anal.s_i,
                      pde_anal.s_j,
                      max_dens_val=pde_anal.max_dens_val)
    if pde_anal.init_flag:
        axarr[0].set_aspect(1.0)
        axarr[1].set_aspect(1.0)
        fig.colorbar(c, ax=axarr[1])
        pde_anal.init_flag = False

    # Graph zeroth moment aka number of crosslinkers
    graph_vs_time(axarr[2], pde_anal.time, pde_anal.mu00, n)
    # Graph first moments of crosslink distribution
    graph_vs_time(axarr[3], pde_anal.time, pde_anal.mu10, n,
                  color='tab:green')
    graph_vs_time(axarr[3], pde_anal.time, pde_anal.mu01, n,
                  color='tab:purple')
    # Graph second moments of crosslinker distribution
    graph_vs_time(axarr[4], pde_anal.time, pde_anal.mu11, n,
                  color='b')
    graph_vs_time(axarr[4], pde_anal.time, pde_anal.mu20, n,
                  color='tab:green')
    graph_vs_time(axarr[4], pde_anal.time, pde_anal.mu02, n,
                  color='tab:purple')
    # Graph forces
    graph_vs_time(axarr[5], pde_anal.time, pde_anal.force_arr[:, 0], n,
                  color='tab:green')
    graph_vs_time(axarr[5], pde_anal.time, pde_anal.force_arr[:, 1], n,
                  color='tab:purple')
    # Graph torques
    graph_vs_time(axarr[6], pde_anal.time, pde_anal.torque_arr[:, 0], n,
                  color='tab:green')
    graph_vs_time(axarr[6], pde_anal.time, pde_anal.torque_arr[:, 1], n,
                  color='tab:purple')
    # Legend information
    axarr[2].legend([r"N({:.2f}) = {:.1f}".format(
        pde_anal.time[n], pde_anal.mu00[n])])
    axarr[3].legend([r"$\mu^{{1,0}}$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                              pde_anal.mu10[n]),
                     r"$\mu^{{0,1}}$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                              pde_anal.mu01[n])])
    axarr[4].legend([r"$\mu^{{1,1}}$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                              pde_anal.mu11[n]),
                     r"$\mu^{{2,0}}$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                              pde_anal.mu20[n]),
                     r"$\mu^{{0,2}}$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                              pde_anal.mu02[n])])
    axarr[5].legend([r"F$_i$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                      pde_anal.force_arr[n, 0]),
                     r"F$_j$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                      pde_anal.force_arr[n, 1])])
    axarr[6].legend([r"$T_i$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                      pde_anal.torque_arr[n, 0]),
                     r"$T_j$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                      pde_anal.torque_arr[n, 1])])
    return fig.gca().lines + fig.gca().collections


def pde_graph_mts_xlink_distr_2d(fig, axarr, n, pde_anal):
    # Clean up if lines
    if not pde_anal.init_flag:
        for ax in axarr.flatten():
            ax.clear()
        for artist in fig.gca().lines + fig.gca().collections:
            artist.remove()
            del artist

    # Draw rods
    graph_2d_rod_pde_diagram(axarr[0], pde_anal, n,
                             scale=1. / (pde_anal.max_dens_val))

    # Make density plot
    c = graph_xl_dens(axarr[1],
                      pde_anal.xl_distr[:, :, n],
                      pde_anal.s_i,
                      pde_anal.s_j,
                      max_dens_val=pde_anal.max_dens_val)
    axarr[1].set_xlabel(
        'Head distance from \n center of fil$_i$ $s_i$ (nm)')
    axarr[1].set_ylabel(
        'Head distance from \n center of fil$_j$ $s_j$ (nm)')

    if pde_anal.init_flag:
        axarr[0].set_aspect(1.0)
        axarr[1].set_aspect(1.0)
        fig.colorbar(c, ax=axarr[1])
        pde_anal.init_flag = False
    axarr[0].text(.05, .90, "Time = {:.2f} sec".format(pde_anal.time[n]),
                  horizontalalignment='left',
                  verticalalignment='bottom',
                  transform=axarr[0].transAxes)

    # pde_anal.time[n])], facecolor='inherit')
    return fig.gca().lines + fig.gca().collections


def pde_graph_stationary_runs_2d(fig, axarr, n, pde_anal):
    # Clean up if lines
    if not pde_anal.init_flag:
        for ax in axarr.flatten():
            ax.clear()
        for artist in fig.gca().lines + fig.gca().collections:
            artist.remove()
            del artist

    # Draw rods
    graph_2d_rod_diagram(axarr[0], pde_anal, n)

    # Make density plot
    c = graph_xl_dens(axarr[1],
                      pde_anal.xl_distr[:, :, n],
                      pde_anal.s_i,
                      pde_anal.s_j,
                      max_dens_val=pde_anal.max_dens_val)
    axarr[1].set_xlabel(
        'Head distance from \n center of fil$_i$ $s_i$ (nm)')
    axarr[1].set_ylabel(
        'Head distance from \n center of fil$_j$ $s_j$ (nm)')

    axarr[2].set_xlabel(r'Time (sec)')
    axarr[2].set_ylabel(r'Motor number')
    axarr[2].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[2].set_ylim(np.amin(pde_anal.mu00),
                      np.amax(pde_anal.mu00))

    axarr[3].set_xlabel(r'Time (sec)')
    axarr[3].set_ylabel(r'Motor force (pN)')
    axarr[3].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[3].set_ylim(np.amin(pde_anal.force_arr),
                      np.amax(pde_anal.force_arr))

    axarr[4].set_xlabel(r'Time (sec)')
    axarr[4].set_ylabel(r'Motor torque (pN$\cdotnm)')
    axarr[4].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[4].set_ylim(np.amin(pde_anal.torque_arr),
                      np.amax(pde_anal.torque_arr))

    if pde_anal.init_flag:
        axarr[0].set_aspect(1.0)
        axarr[1].set_aspect(1.0)
        fig.colorbar(c, ax=axarr[1])
        pde_anal.init_flag = False

    graph_vs_time(axarr[2], pde_anal.time, pde_anal.mu00, n)
    graph_vs_time(axarr[3], pde_anal.time, pde_anal.force_arr[:, 0], n,
                  color='tab:green')
    graph_vs_time(axarr[3], pde_anal.time, pde_anal.force_arr[:, 1], n,
                  color='tab:purple')
    graph_vs_time(axarr[4], pde_anal.time, pde_anal.torque_arr[:, 0], n,
                  color='tab:green')
    graph_vs_time(axarr[4], pde_anal.time, pde_anal.torque_arr[:, 1], n,
                  color='tab:purple')

    return fig.gca().lines + fig.gca().collections


######################################
#  Crosslinker distribution moments  #
######################################

def pde_graph_recreate_xlink_distr_2d(fig, axarr, n, pde_anal):
    # Clean up if lines
    if not pde_anal.init_flag:
        for ax in axarr.flatten():
            ax.clear()
        for artist in fig.gca().lines + fig.gca().collections:
            artist.remove()
            del artist

    # Draw rods
    graph_2d_rod_pde_diagram(axarr[0], pde_anal, n,
                             scale=1. / (pde_anal.max_dens_val))

    # Make a function of a recreated distribution

    # Make density plot
    cb1 = graph_xl_dens(axarr[1],
                        pde_anal.xl_distr[:, :, n],
                        pde_anal.s_i,
                        pde_anal.s_j,
                        max_dens_val=pde_anal.max_dens_val)
    # Make recreation of distribution
    xl_distr_rec_func = pde_anal.create_distr_approx_func()
    s_j_grid, s_i_grid = np.meshgrid(pde_anal.s_j, pde_anal.s_i)
    xl_distr_rec = xl_distr_rec_func(s_i_grid, s_j_grid, n)
    cb2 = graph_xl_dens(axarr[2],
                        xl_distr_rec,
                        pde_anal.s_i,
                        pde_anal.s_j,
                        max_dens_val=pde_anal.max_dens_val)

    axarr[1].set_xlabel(
        'Head distance from \n center of fil$_i$ $s_i$ (nm)')
    axarr[1].set_ylabel(
        'Head distance from \n center of fil$_j$ $s_j$ (nm)')
    axarr[2].set_xlabel(
        'Head distance from \n center of fil$_i$ $s_i$ (nm)')
    axarr[2].set_ylabel(
        'Head distance from \n center of fil$_j$ $s_j$ (nm)')

    if pde_anal.init_flag:
        fig.colorbar(cb1, ax=axarr[1])
        fig.colorbar(cb2, ax=axarr[2])
        axarr[0].set_aspect(1.0)
        axarr[1].set_aspect(1.0)
        axarr[2].set_aspect(1.0)

        pde_anal.init_flag = False
    axarr[0].text(.05, .95, "Time = {:.2f} sec".format(pde_anal.time[n]),
                  horizontalalignment='left',
                  verticalalignment='bottom',
                  transform=axarr[0].transAxes)

    # pde_anal.time[n])], facecolor='inherit')
    return fig.gca().lines + fig.gca().collections
