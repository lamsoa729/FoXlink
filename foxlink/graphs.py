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


class LineDataUnits(Line2D):
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


def draw_rod(ax, r_vec, u_vec, L, rod_diam, color='tab:green'):
    line = LineDataUnits((r_vec[1] - .5 * L * u_vec[1],
                          r_vec[1] + .5 * L * u_vec[1]),
                         (r_vec[2] - .5 * L * u_vec[2],
                          r_vec[2] + .5 * L * u_vec[2]),
                         linewidth=rod_diam, solid_capstyle='round',
                         color=color, clip_on=False, )

    tip = Circle((r_vec[1] + .5 * L * u_vec[1], r_vec[2] + .5 * L * u_vec[2]),
                 .5 * rod_diam, color='r', zorder=3)
    ax.add_patch(tip)
    ax.add_line(line)


def draw_xlink(ax, e_i, e_j, lw=10, color='k', alpha=.5):
    line = LineDataUnits((e_i[1], e_j[1]), (e_i[2], e_j[2]),
                         linewidth=lw,  # solid_capstyle='round',
                         color=color, clip_on=False, alpha=alpha)
    ax.add_line(line)


def draw_moment_rod(ax, r_vec, u_vec, L, rod_diam,
                    mu00, mu10, mu20, num_max=50.):
    cmap = mpl.cm.get_cmap('viridis')
    scaled_mu10 = mu10 / mu00 if mu00 else 0
    mu10_loc = RegularPolygon((r_vec[1] + scaled_mu10 * u_vec[1],
                               r_vec[2] + scaled_mu10 * u_vec[2]),
                              5, rod_diam, zorder=3)
    mu20_dist = np.sqrt(mu20 / mu00) if mu00 else 0
    # mu20_ellipse = Ellipse((r_vec[1], r_vec[2]), mu20_dist*2., rod_diam,
    # angle=np.arctan(u_vec[2]/u_vec[1]), zorder=4, fill=False)
    mu20_bar = FancyArrowPatch((r_vec[1] - mu20_dist * u_vec[1],
                                r_vec[2] - mu20_dist * u_vec[2]),
                               (r_vec[1] + mu20_dist * u_vec[1],
                                r_vec[2] + mu20_dist * u_vec[2]),
                               arrowstyle=ArrowStyle(
                                   '|-|', widthA=convert_size_units(.5 * rod_diam, ax),
                                   widthB=convert_size_units(.5 *
                                                             rod_diam, ax)),
                               zorder=4)
    ax.add_patch(mu10_loc)
    ax.add_patch(mu20_bar)
    draw_rod(ax, r_vec, u_vec, L, rod_diam, color=cmap(mu00 / num_max))


def graph_vs_time(ax, time, y, n=-1, color='b'):
    """!TODO: Docstring for graph_vs_t.

    @param ax: TODO
    @param time: TODO
    @param y: TODO
    @param n: TODO
    @return: TODO

    """
    s = ax.scatter(time[:n], y[:n], c=color)
    return s


def graph_xl_dens(ax, psi, s1, s2, **kwargs):
    """!Graph an instance in time of the crosslinker density for the PDE

    @param psi: crosslinker density
    @param **kwargs: TODO
    @return: TODO

    """
    s1 = np.asarray(s1)
    s2 = np.asarray(s2)
    psi = np.transpose(np.asarray(psi))
    if "max_dens_val" in kwargs:
        max_val = kwargs["max_dens_val"]
        c = ax.pcolormesh(s1, s2, psi, vmin=0, vmax=max_val)
    else:
        c = ax.pcolormesh(s1, s2, psi)
    return c


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
    x_ends = [np.amax(0.5 * L_i * u_i_arr[:, 1] + r_i_arr[:, 1]),
              np.amin(0.5 * L_i * u_i_arr[:, 1] + r_i_arr[:, 1]),
              np.amax(-.5 * L_i * u_i_arr[:, 1] + r_i_arr[:, 1]),
              np.amin(-.5 * L_i * u_i_arr[:, 1] + r_i_arr[:, 1]),
              np.amax(0.5 * L_j * u_j_arr[:, 1] + r_j_arr[:, 1]),
              np.amin(0.5 * L_j * u_j_arr[:, 1] + r_j_arr[:, 1]),
              np.amax(-.5 * L_j * u_j_arr[:, 1] + r_j_arr[:, 1]),
              np.amin(-.5 * L_j * u_j_arr[:, 1] + r_j_arr[:, 1])]

    # Get all extreme positions of tips in the second dimension to maintain
    # consistent graphing size
    y_ends = [np.amax(0.5 * L_i * u_i_arr[:, 2] + r_i_arr[:, 2]),
              np.amin(0.5 * L_i * u_i_arr[:, 2] + r_i_arr[:, 2]),
              np.amax(-.5 * L_i * u_i_arr[:, 2] + r_i_arr[:, 2]),
              np.amin(-.5 * L_i * u_i_arr[:, 2] + r_i_arr[:, 2]),
              np.amax(0.5 * L_j * u_j_arr[:, 2] + r_j_arr[:, 2]),
              np.amin(0.5 * L_j * u_j_arr[:, 2] + r_j_arr[:, 2]),
              np.amax(-.5 * L_j * u_j_arr[:, 2] + r_j_arr[:, 2]),
              np.amin(-.5 * L_j * u_j_arr[:, 2] + r_j_arr[:, 2])]

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

    draw_moment_rod(ax, r_i_arr[n], u_i_arr[n], L_i, rod_diam,
                    anal.mu00[n], anal.mu10[n], anal.mu20[n],
                    num_max=mu00_max)
    draw_moment_rod(ax, r_j_arr[n], u_j_arr[n], L_j, rod_diam,
                    anal.mu00[n], anal.mu01[n], anal.mu02[n],
                    num_max=mu00_max)

    labels = ["MT$_1$", "MT$_2$", "Plus-end"]
    # if anal.OT1_pos is not None or anal.OT2_pos is not None:
    #     labels += ["Optical trap", "Bead"]
    ax.legend(labels, loc="upper right")


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
        x_ends = [np.amax(0.5 * L_i * u_i_arr[:, 1] + r_i_arr[:, 1]),
                  np.amin(0.5 * L_i * u_i_arr[:, 1] + r_i_arr[:, 1]),
                  np.amax(-.5 * L_i * u_i_arr[:, 1] + r_i_arr[:, 1]),
                  np.amin(-.5 * L_i * u_i_arr[:, 1] + r_i_arr[:, 1]),
                  np.amax(0.5 * L_j * u_j_arr[:, 1] + r_j_arr[:, 1]),
                  np.amin(0.5 * L_j * u_j_arr[:, 1] + r_j_arr[:, 1]),
                  np.amax(-.5 * L_j * u_j_arr[:, 1] + r_j_arr[:, 1]),
                  np.amin(-.5 * L_j * u_j_arr[:, 1] + r_j_arr[:, 1])]

        # Get all extreme positions of tips in the second dimension to maintain
        # consistent graphing size
        y_ends = [np.amax(0.5 * L_i * u_i_arr[:, 2] + r_i_arr[:, 2]),
                  np.amin(0.5 * L_i * u_i_arr[:, 2] + r_i_arr[:, 2]),
                  np.amax(-.5 * L_i * u_i_arr[:, 2] + r_i_arr[:, 2]),
                  np.amin(-.5 * L_i * u_i_arr[:, 2] + r_i_arr[:, 2]),
                  np.amax(0.5 * L_j * u_j_arr[:, 2] + r_j_arr[:, 2]),
                  np.amin(0.5 * L_j * u_j_arr[:, 2] + r_j_arr[:, 2]),
                  np.amax(-.5 * L_j * u_j_arr[:, 2] + r_j_arr[:, 2]),
                  np.amin(-.5 * L_j * u_j_arr[:, 2] + r_j_arr[:, 2])]

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
        labels = ["MT$_1$", "MT$_2$", "Plus-end"]
        # if anal.OT1_pos is not None or anal.OT2_pos is not None:
        #     labels += ["Optical trap", "Bead"]
        ax.legend(labels, loc="upper right")


def me_graph_all_data_2d(fig, axarr, n, me_anal):
    # Clean up if lines on axis object to speed up movie making
    if not me_anal.init_flag:
        for ax in axarr.flatten():
            ax.clear()
        for artist in fig.gca().lines + fig.gca().collections:
            artist.remove()
            del artist

    axarr[1].set_xlabel(r'Time (sec)')
    axarr[1].set_ylabel('Distance between MTs \n centers of mass (nm)')
    axarr[1].set_xlim(left=0, right=me_anal.time[-1])
    axarr[1].set_ylim(np.amin(me_anal.dR_arr),
                      np.amax(me_anal.dR_arr))

    axarr[2].set_xlabel(r'Time (sec)')
    axarr[2].set_ylabel('Angle between MT \n orientation vectors (rad)')
    axarr[2].set_xlim(left=0, right=me_anal.time[-1])
    axarr[2].set_ylim(np.nanmin(me_anal.phi_arr),
                      np.nanmax(me_anal.phi_arr))

    axarr[3].set_xlabel(r'Time (sec)')
    axarr[3].set_ylabel(r'Crosslinker number')
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
    graph_2d_rod_diagram(axarr[0], me_anal, n)

    if me_anal.init_flag:
        axarr[0].set_aspect(1.0)
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
        'Head distance from \n center of MT$_1$ $s_1$ (nm)')
    axarr[1].set_ylabel(
        'Head distance from \n center of MT$_2$ $s_2$ (nm)')

    axarr[2].set_xlabel(r'Time (sec)')
    axarr[2].set_ylabel(r'Crosslinker number')
    axarr[2].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[2].set_ylim(np.amin(pde_anal.mu00),
                      np.amax(pde_anal.mu00))

    axarr[3].set_xlabel(r'Time (sec)')
    axarr[3].set_ylabel(r'Total crosslinker force (pN)')
    axarr[3].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[3].set_ylim(np.amin(pde_anal.force_arr),
                      np.amax(pde_anal.force_arr))

    axarr[4].set_xlabel(r'Time (sec)')
    axarr[4].set_ylabel(r'Total crosslinker torque (pN*nm)')
    axarr[4].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[4].set_ylim(np.amin(pde_anal.torque_arr),
                      np.amax(pde_anal.torque_arr))

    axarr[5].set_xlabel(r'Time (sec)')
    axarr[5].set_ylabel(r'First moments (nm)')
    axarr[5].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[5].set_ylim(min(np.amin(pde_anal.mu10), np.amin(pde_anal.mu01)),
                      max(np.amax(pde_anal.mu10), np.amax(pde_anal.mu01)))

    axarr[6].set_xlabel(r'Time (sec)')
    axarr[6].set_ylabel('Distance between MTs \n centers of mass (nm)')
    axarr[6].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[6].set_ylim(np.amin(pde_anal.dR_arr),
                      np.amax(pde_anal.dR_arr))

    axarr[7].set_xlabel(r'Time (sec)')
    axarr[7].set_ylabel('Angle between MT \n orientation vectors (rad)')
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
                      pde_anal.s1,
                      pde_anal.s2,
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
    axarr[3].legend([r"F$_1$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                      pde_anal.force_arr[n, 0]),
                     r"F$_2$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                      pde_anal.force_arr[n, 1])])
    axarr[4].legend([r"$T_1$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                      pde_anal.torque_arr[n, 0]),
                     r"$T_2$({:.2f}) = {:.1f}".format(pde_anal.time[n],
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
        'Head distance from \n center of MT$_1$ $s_1$ (nm)')
    axarr[1].set_ylabel(
        'Head distance from \n center of MT$_2$ $s_2$ (nm)')

    axarr[2].set_xlabel(r'Time (sec)')
    axarr[2].set_ylabel(r'Crosslinker number')
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
    axarr[5].set_ylabel(r'Total crosslinker force (pN)')
    axarr[5].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[5].set_ylim(np.amin(pde_anal.force_arr),
                      np.amax(pde_anal.force_arr))

    axarr[6].set_xlabel(r'Time (sec)')
    axarr[6].set_ylabel(r'Total crosslinker torque (pN*nm)')
    axarr[6].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[6].set_ylim(np.amin(pde_anal.torque_arr),
                      np.amax(pde_anal.torque_arr))

    # Draw rods
    graph_2d_rod_diagram(axarr[0], pde_anal, n)

    # Make crosslinker density plot
    c = graph_xl_dens(axarr[1],
                      pde_anal.xl_distr[:, :, n],
                      pde_anal.s1,
                      pde_anal.s2,
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
    axarr[5].legend([r"F$_1$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                      pde_anal.force_arr[n, 0]),
                     r"F$_2$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                      pde_anal.force_arr[n, 1])])
    axarr[6].legend([r"$T_1$({:.2f}) = {:.1f}".format(pde_anal.time[n],
                                                      pde_anal.torque_arr[n, 0]),
                     r"$T_2$({:.2f}) = {:.1f}".format(pde_anal.time[n],
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
    graph_2d_rod_diagram(axarr[0], pde_anal, n)

    # Make density plot
    c = graph_xl_dens(axarr[1],
                      pde_anal.xl_distr[:, :, n],
                      pde_anal.s1,
                      pde_anal.s2,
                      max_dens_val=pde_anal.max_dens_val)
    axarr[1].set_xlabel(
        'Head distance from \n center of MT$_1$ $s_1$ (nm)')
    axarr[1].set_ylabel(
        'Head distance from \n center of MT$_2$ $s_2$ (nm)')

    if pde_anal.init_flag:
        axarr[0].set_aspect(1.0)
        axarr[1].set_aspect(1.0)
        fig.colorbar(c, ax=axarr[1])
        pde_anal.init_flag = False
    axarr[0].text(.05, .95, "Time = {:.2f} sec".format(pde_anal.time[n]),
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
                      pde_anal.s1,
                      pde_anal.s2,
                      max_dens_val=pde_anal.max_dens_val)
    axarr[1].set_xlabel(
        'Head distance from \n center of MT$_1$ $s_1$ (nm)')
    axarr[1].set_ylabel(
        'Head distance from \n center of MT$_2$ $s_2$ (nm)')

    axarr[2].set_xlabel(r'Time (sec)')
    axarr[2].set_ylabel(r'Crosslinker number')
    axarr[2].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[2].set_ylim(np.amin(pde_anal.mu00),
                      np.amax(pde_anal.mu00))

    axarr[3].set_xlabel(r'Time (sec)')
    axarr[3].set_ylabel(r'Total crosslinker force (pN)')
    axarr[3].set_xlim(left=0, right=pde_anal.time[-1])
    axarr[3].set_ylim(np.amin(pde_anal.force_arr),
                      np.amax(pde_anal.force_arr))

    axarr[4].set_xlabel(r'Time (sec)')
    axarr[4].set_ylabel(r'Total crosslinker torque (pN*nm)')
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
