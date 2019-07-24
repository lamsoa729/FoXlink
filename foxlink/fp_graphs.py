#!/usr/bin/env python
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
# import matplotlib.pyplot as plt


"""@package docstring
File: fp_graphs.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description: File containing modular graphing functions for Fokker-Planck data.
"""


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
    """!Graph an instance in time of the crosslinker density for the FP equation

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


def fp_graph_all_data_2d(fig, axarr, n, FP_anal):

    params = FP_anal._params
    # Clean up if lines
    if not FP_anal.init_flag:
        for ax in axarr.flatten():
            ax.clear()
        for artist in fig.gca().lines + fig.gca().collections:
            artist.remove()
            del artist

    # Draw rods
    L1 = params["L1"]
    L2 = params["L2"]
    lw = params['rod_diameter']
    if hasattr(FP_anal, 'phi_arr') and not hasattr(FP_anal, 'R1_vec'):
        hphi = FP_anal.phi_arr[n] * .5
        line1 = LineDataUnits((0, L1 * np.cos(hphi)),
                              (0, L1 * np.sin(hphi)),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:green', clip_on=False)
        line2 = LineDataUnits((0, L2 * np.cos(hphi)),
                              (0, -L2 * np.sin(hphi)),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:purple', clip_on=False)
        axarr[0].add_line(line1)
        axarr[0].add_line(line2)
    elif hasattr(FP_anal, 'R_arr'):
        r = FP_anal.R_arr[n, :]
        line1 = LineDataUnits((-.5 * L1, .5 * L1),
                              (0, 0),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:green', clip_on=False)
        line2 = LineDataUnits((-.5 * L1 + r[0], .5 * L1 + r[0]),
                              (r[1], r[1]),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:purple', clip_on=False)
        axarr[0].add_line(line1)
        axarr[0].add_line(line2)
    else:
        # r = self.R_arr[n, :]
        r1 = FP_anal.R1_pos[n]
        r2 = FP_anal.R2_pos[n]
        u1 = FP_anal.R1_vec[n]
        u2 = FP_anal.R2_vec[n]

        line1 = LineDataUnits((-.5 * L1 * u1[1] + r1[1],
                               .5 * L1 * u1[1] + r1[1]),
                              (-.5 * L1 * u1[2] + r1[2],
                               .5 * L1 * u1[2] + r1[2]),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:green', clip_on=False)
        tip1 = Circle((.5 * L1 * u1[1] + r1[1], .5 * L1 * u1[2] + r1[2]),
                      .5 * lw, color='r', zorder=3)
        axarr[0].add_patch(tip1)
        axarr[0].add_line(line1)
        line2 = LineDataUnits((-.5 * L2 * u2[1] + r2[1],
                               .5 * L2 * u2[1] + r2[1]),
                              (-.5 * L2 * u2[2] + r2[2],
                               .5 * L2 * u2[2] + r2[2]),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:purple', clip_on=False)
        tip2 = Circle((.5 * L2 * u2[1] + r2[1], .5 * L2 * u2[2] + r2[2]),
                      .5 * lw, color='r', zorder=3)
        axarr[0].add_line(line2)
        axarr[0].add_patch(tip2)
        r1 = FP_anal.R1_pos
        r2 = FP_anal.R2_pos
        u1 = FP_anal.R1_vec
        u2 = FP_anal.R2_vec

        # Get all extreme positions of tips in the first dimension
        x_ends = [np.amax(.5 * L1 * u1[:, 1] + r1[:, 1]),
                  np.amin(.5 * L1 * u1[:, 1] + r1[:, 1]),
                  np.amax(-.5 * L1 * u1[:, 1] + r1[:, 1]),
                  np.amin(-.5 * L1 * u1[:, 1] + r1[:, 1]),
                  np.amax(.5 * L2 * u2[:, 1] + r2[:, 1]),
                  np.amin(.5 * L2 * u2[:, 1] + r2[:, 1]),
                  np.amax(-.5 * L2 * u2[:, 1] + r2[:, 1]),
                  np.amin(-.5 * L2 * u2[:, 1] + r2[:, 1])]

        # Get all extreme positions of tips in the second dimension
        y_ends = [np.amax(.5 * L1 * u1[:, 2] + r1[:, 2]),
                  np.amin(.5 * L1 * u1[:, 2] + r1[:, 2]),
                  np.amax(-.5 * L1 * u1[:, 2] + r1[:, 2]),
                  np.amin(-.5 * L1 * u1[:, 2] + r1[:, 2]),
                  np.amax(.5 * L2 * u2[:, 2] + r2[:, 2]),
                  np.amin(.5 * L2 * u2[:, 2] + r2[:, 2]),
                  np.amax(-.5 * L2 * u2[:, 2] + r2[:, 2]),
                  np.amin(-.5 * L2 * u2[:, 2] + r2[:, 2])]

        max_x = max(x_ends + y_ends)
        max_x = max_x * 1.25 if max_x > 0 else .75 * max_x
        min_x = min(x_ends + y_ends)
        min_x = min_x * 1.25 if min_x < 0 else .75 * min_x

        # Make a square box always
        max_y = max_x
        min_y = min_x

        axarr[0].set_xlim(min_x, max_x)
        axarr[0].set_ylim(min_y, max_y)
        axarr[0].set_xlabel(r'x (nm)')
        axarr[0].set_ylabel(r'y (nm)')
        axarr[1].set_xlabel(
            'Head distance from \n center of MT$_1$ $s_1$ (nm)')
        axarr[1].set_ylabel(
            'Head distance from \n center of MT$_2$ $s_2$ (nm)')

        axarr[2].set_xlabel(r'Time (sec)')
        axarr[2].set_ylabel(r'Crosslinker number')
        axarr[2].set_xlim(left=0, right=FP_anal.time[-1])
        axarr[2].set_ylim(np.amin(FP_anal.Nxl_arr),
                          np.amax(FP_anal.Nxl_arr))

        axarr[3].set_xlabel(r'Time (sec)')
        axarr[3].set_ylabel(r'Total crosslinker force (pN)')
        axarr[3].set_xlim(left=0, right=FP_anal.time[-1])
        axarr[3].set_ylim(np.amin(FP_anal.force_arr),
                          np.amax(FP_anal.force_arr))

        axarr[4].set_xlabel(r'Time (sec)')
        axarr[4].set_ylabel(r'Total crosslinker torque (pN*nm)')
        axarr[4].set_xlim(left=0, right=FP_anal.time[-1])
        axarr[4].set_ylim(np.amin(FP_anal.torque_arr),
                          np.amax(FP_anal.torque_arr))

        axarr[5].set_xlabel(r'Time (sec)')
        axarr[5].set_ylabel('Distance between MTs \n centers of mass (nm)')
        axarr[5].set_xlim(left=0, right=FP_anal.time[-1])
        axarr[5].set_ylim(np.amin(FP_anal.dR_arr),
                          np.amax(FP_anal.dR_arr))

        axarr[6].set_xlabel(r'Time (sec)')
        axarr[6].set_ylabel('Angle between MT \n orientation vectors (rad)')
        axarr[6].set_xlim(left=0, right=FP_anal.time[-1])
        axarr[6].set_ylim(np.amin(FP_anal.phi_arr),
                          np.amax(FP_anal.phi_arr))

    # Make density plot
    c = graph_xl_dens(axarr[1],
                      FP_anal.xl_distr[:, :, n],
                      FP_anal.s1,
                      FP_anal.s2,
                      max_dens_val=FP_anal.max_dens_val)
    if FP_anal.init_flag:
        fig.colorbar(c, ax=axarr[1])
        axarr[0].set_aspect(1.0)
        axarr[1].set_aspect(1.0)
        FP_anal.init_flag = False
    graph_vs_time(axarr[2], FP_anal.time, FP_anal.Nxl_arr, n)
    graph_vs_time(axarr[3], FP_anal.time, FP_anal.force_arr, n)
    graph_vs_time(axarr[4], FP_anal.time, FP_anal.torque_arr, n)
    graph_vs_time(axarr[5], FP_anal.time, FP_anal.dR_arr, n)
    graph_vs_time(axarr[6], FP_anal.time, FP_anal.phi_arr, n)
    axarr[0].legend(["MT$_1$", "MT$_2$", "Plus-end"], loc="upper right")
    axarr[2].legend(["N({:.2f}) = {:.1f}".format(
        FP_anal.time[n], FP_anal.Nxl_arr[n])])
    axarr[3].legend(["F({:.2f}) = {:.1f} Pn".format(
        FP_anal.time[n], FP_anal.force_arr[n])])
    axarr[4].legend(
        [r'$\tau$({:.2f}) = {:.1f} Pn*nm'.format(FP_anal.time[n], FP_anal.torque_arr[n])])
    axarr[5].legend([r"$\Delta$R({:.2f}) = {:.1f} nm".format(
        FP_anal.time[n], FP_anal.dR_arr[n])])
    axarr[6].legend([r"$\phi$({:.2f}) = {:.1f} rad".format(
        FP_anal.time[n], FP_anal.phi_arr[n])])
    return fig.gca().lines + fig.gca().collections


def fp_graph_mts_xlink_distr_2d(fig, axarr, n, FP_anal):

    params = FP_anal._params
    # Clean up if lines
    if not FP_anal.init_flag:
        for ax in axarr.flatten():
            ax.clear()
        for artist in fig.gca().lines + fig.gca().collections:
            artist.remove()
            del artist

    # Draw rods
    L1 = params["L1"]
    L2 = params["L2"]
    lw = params['rod_diameter']
    if hasattr(FP_anal, 'phi_arr') and not hasattr(FP_anal, 'R1_vec'):
        hphi = FP_anal.phi_arr[n] * .5
        line1 = LineDataUnits((0, L1 * np.cos(hphi)),
                              (0, L1 * np.sin(hphi)),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:green', clip_on=False)
        line2 = LineDataUnits((0, L2 * np.cos(hphi)),
                              (0, -L2 * np.sin(hphi)),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:purple', clip_on=False)
        axarr[0].add_line(line1)
        axarr[0].add_line(line2)
    elif hasattr(FP_anal, 'R_arr'):
        r = FP_anal.R_arr[n, :]
        line1 = LineDataUnits((-.5 * L1, .5 * L1),
                              (0, 0),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:green', clip_on=False)
        line2 = LineDataUnits((-.5 * L1 + r[0], .5 * L1 + r[0]),
                              (r[1], r[1]),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:purple', clip_on=False)
        axarr[0].add_line(line1)
        axarr[0].add_line(line2)
    else:
        r1 = FP_anal.R1_pos[n]
        r2 = FP_anal.R2_pos[n]
        u1 = FP_anal.R1_vec[n]
        u2 = FP_anal.R2_vec[n]

        line1 = LineDataUnits((-.5 * L1 * u1[1] + r1[1],
                               .5 * L1 * u1[1] + r1[1]),
                              (-.5 * L1 * u1[2] + r1[2],
                               .5 * L1 * u1[2] + r1[2]),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:green', clip_on=False)
        tip1 = Circle((.5 * L1 * u1[1] + r1[1], .5 * L1 * u1[2] + r1[2]),
                      .5 * lw, color='r', zorder=3)
        axarr[0].add_patch(tip1)
        axarr[0].add_line(line1)
        line2 = LineDataUnits((-.5 * L2 * u2[1] + r2[1],
                               .5 * L2 * u2[1] + r2[1]),
                              (-.5 * L2 * u2[2] + r2[2],
                               .5 * L2 * u2[2] + r2[2]),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:purple', clip_on=False)
        tip2 = Circle((.5 * L2 * u2[1] + r2[1], .5 * L2 * u2[2] + r2[2]),
                      .5 * lw, color='r', zorder=3)
        axarr[0].add_line(line2)
        axarr[0].add_patch(tip2)
        r1 = FP_anal.R1_pos
        r2 = FP_anal.R2_pos
        u1 = FP_anal.R1_vec
        u2 = FP_anal.R2_vec
        # if FP_anal._h5_data['/

        # Get all extreme positions of tips in the first dimension
        x_ends = [np.amax(.5 * L1 * u1[:, 1] + r1[:, 1]),
                  np.amin(.5 * L1 * u1[:, 1] + r1[:, 1]),
                  np.amax(-.5 * L1 * u1[:, 1] + r1[:, 1]),
                  np.amin(-.5 * L1 * u1[:, 1] + r1[:, 1]),
                  np.amax(.5 * L2 * u2[:, 1] + r2[:, 1]),
                  np.amin(.5 * L2 * u2[:, 1] + r2[:, 1]),
                  np.amax(-.5 * L2 * u2[:, 1] + r2[:, 1]),
                  np.amin(-.5 * L2 * u2[:, 1] + r2[:, 1])]

        # Get all extreme positions of tips in the second dimension
        y_ends = [np.amax(.5 * L1 * u1[:, 2] + r1[:, 2]),
                  np.amin(.5 * L1 * u1[:, 2] + r1[:, 2]),
                  np.amax(-.5 * L1 * u1[:, 2] + r1[:, 2]),
                  np.amin(-.5 * L1 * u1[:, 2] + r1[:, 2]),
                  np.amax(.5 * L2 * u2[:, 2] + r2[:, 2]),
                  np.amin(.5 * L2 * u2[:, 2] + r2[:, 2]),
                  np.amax(-.5 * L2 * u2[:, 2] + r2[:, 2]),
                  np.amin(-.5 * L2 * u2[:, 2] + r2[:, 2])]

        max_x = max(x_ends + y_ends)
        max_x = max_x * 1.25 if max_x > 0 else .75 * max_x
        min_x = min(x_ends + y_ends)
        min_x = min_x * 1.25 if min_x < 0 else .75 * min_x

        max_y = max_x
        min_y = min_x

        axarr[0].set_xlim(min_x, max_x)
        axarr[0].set_ylim(min_y, max_y)
        axarr[0].set_xlabel(r'x (nm)')
        axarr[0].set_ylabel(r'y (nm)')
        axarr[1].set_xlabel(
            'Head distance from \n center of MT$_1$ $s_1$ (nm)')
        axarr[1].set_ylabel(
            'Head distance from \n center of MT$_2$ $s_2$ (nm)')

    # Make density plot
    c = graph_xl_dens(axarr[1],
                      FP_anal.xl_distr[:, :, n],
                      FP_anal.s1,
                      FP_anal.s2,
                      max_dens_val=FP_anal.max_dens_val)

    if FP_anal.init_flag:
        axarr[0].set_aspect(1.0)
        axarr[1].set_aspect(1.0)
        fig.colorbar(c, ax=axarr[1])
        FP_anal.init_flag = False
    axarr[0].text(.05, .95, "Time = {:.2f} sec".format(FP_anal.time[n]),
                  horizontalalignment='left',
                  verticalalignment='bottom',
                  transform=axarr[0].transAxes)

    axarr[0].legend(["MT$_1$", "MT$_2$", "Plus-end"], loc="upper right")
    # FP_anal.time[n])], facecolor='inherit')
    return fig.gca().lines + fig.gca().collections
