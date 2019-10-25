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


def graph_2d_rod_diagram(ax, FP_anal, n=-1):
    """!TODO: Docstring for graph_2d_rod_diagram.

    @param ax: TODO
    @param FP_anal: TODO
    @return: TODO

    """
    params = FP_anal._params
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
        ax.add_line(line1)
        ax.add_line(line2)
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
        ax.add_line(line1)
        ax.add_line(line2)
    else:
        r1_arr = FP_anal.R1_pos
        r2_arr = FP_anal.R2_pos
        u1_arr = FP_anal.R1_vec
        u2_arr = FP_anal.R2_vec
        r1 = r1_arr[n]
        r2 = r2_arr[n]
        u1 = u1_arr[n]
        u2 = u2_arr[n]

        line1 = LineDataUnits((-.5 * L1 * u1[1] + r1[1],
                               0.5 * L1 * u1[1] + r1[1]),
                              (-.5 * L1 * u1[2] + r1[2],
                               0.5 * L1 * u1[2] + r1[2]),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:green', clip_on=False)
        tip1 = Circle((.5 * L1 * u1[1] + r1[1], .5 * L1 * u1[2] + r1[2]),
                      .5 * lw, color='r', zorder=3)
        ax.add_patch(tip1)
        ax.add_line(line1)
        if FP_anal.OT1_pos is not None:
            ot1 = Circle((FP_anal.OT1_pos[n, 1], FP_anal.OT1_pos[n, 2]),
                         3 * lw, color='y', alpha=.5)
            mtip1 = Circle((-.5 * L1 * u1[1] + r1[1], -.5 * L1 * u1[2] + r1[2]),
                           lw, color='b', zorder=4)
            ax.add_patch(ot1)
            ax.add_patch(mtip1)

        line2 = LineDataUnits((-.5 * L2 * u2[1] + r2[1],
                               0.5 * L2 * u2[1] + r2[1]),
                              (-.5 * L2 * u2[2] + r2[2],
                               0.5 * L2 * u2[2] + r2[2]),
                              linewidth=lw, solid_capstyle='round',
                              color='tab:purple', clip_on=False)
        tip2 = Circle((.5 * L2 * u2[1] + r2[1], .5 * L2 * u2[2] + r2[2]),
                      .5 * lw, color='r', zorder=3)
        ax.add_line(line2)
        ax.add_patch(tip2)
        if FP_anal.OT2_pos is not None:
            ot2 = Circle((FP_anal.OT2_pos[n, 1], FP_anal.OT2_pos[n, 2]),
                         3 * lw, color='y', alpha=.5)
            mtip2 = Circle((-.5 * L2 * u2[1] + r2[1], -.5 * L2 * u2[2] + r2[2]),
                           lw, color='b', zorder=4)
            ax.add_patch(ot2)
            ax.add_patch(mtip2)

        # Get all extreme positions of tips in the first dimension
        x_ends = [np.amax(0.5 * L1 * u1_arr[:, 1] + r1_arr[:, 1]),
                  np.amin(0.5 * L1 * u1_arr[:, 1] + r1_arr[:, 1]),
                  np.amax(-.5 * L1 * u1_arr[:, 1] + r1_arr[:, 1]),
                  np.amin(-.5 * L1 * u1_arr[:, 1] + r1_arr[:, 1]),
                  np.amax(0.5 * L2 * u2_arr[:, 1] + r2_arr[:, 1]),
                  np.amin(0.5 * L2 * u2_arr[:, 1] + r2_arr[:, 1]),
                  np.amax(-.5 * L2 * u2_arr[:, 1] + r2_arr[:, 1]),
                  np.amin(-.5 * L2 * u2_arr[:, 1] + r2_arr[:, 1])]

        # Get all extreme positions of tips in the second dimension
        y_ends = [np.amax(0.5 * L1 * u1_arr[:, 2] + r1_arr[:, 2]),
                  np.amin(0.5 * L1 * u1_arr[:, 2] + r1_arr[:, 2]),
                  np.amax(-.5 * L1 * u1_arr[:, 2] + r1_arr[:, 2]),
                  np.amin(-.5 * L1 * u1_arr[:, 2] + r1_arr[:, 2]),
                  np.amax(0.5 * L2 * u2_arr[:, 2] + r2_arr[:, 2]),
                  np.amin(0.5 * L2 * u2_arr[:, 2] + r2_arr[:, 2]),
                  np.amax(-.5 * L2 * u2_arr[:, 2] + r2_arr[:, 2]),
                  np.amin(-.5 * L2 * u2_arr[:, 2] + r2_arr[:, 2])]

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
        if FP_anal.OT1_pos is not None or FP_anal.OT2_pos is not None:
            labels += ["Optical trap", "Bead"]
        ax.legend(labels, loc="upper right")


def me_graph_all_data_2d(fig, axarr, n, ME_anal):
    # Clean up if lines
    if not ME_anal.init_flag:
        for ax in axarr.flatten():
            ax.clear()
        for artist in fig.gca().lines + fig.gca().collections:
            artist.remove()
            del artist

    axarr[1].set_xlabel(r'Time (sec)')
    axarr[1].set_ylabel('Distance between MTs \n centers of mass (nm)')
    axarr[1].set_xlim(left=0, right=ME_anal.time[-1])
    axarr[1].set_ylim(np.amin(ME_anal.dR_arr),
                      np.amax(ME_anal.dR_arr))

    axarr[2].set_xlabel(r'Time (sec)')
    axarr[2].set_ylabel('Angle between MT \n orientation vectors (rad)')
    axarr[2].set_xlim(left=0, right=ME_anal.time[-1])
    axarr[2].set_ylim(np.nanmin(ME_anal.phi_arr),
                      np.nanmax(ME_anal.phi_arr))

    axarr[3].set_xlabel(r'Time (sec)')
    axarr[3].set_ylabel(r'Crosslinker number')
    axarr[3].set_xlim(left=0, right=ME_anal.time[-1])
    axarr[3].set_ylim(np.amin(ME_anal.rho),
                      np.amax(ME_anal.rho))

    axarr[4].set_xlabel(r'Time (sec)')
    axarr[4].set_ylabel(r'First moments (nm)')
    axarr[4].set_xlim(left=0, right=ME_anal.time[-1])
    axarr[4].set_ylim(np.amin(ME_anal.P_n), np.amax(ME_anal.P_n))

    axarr[5].set_xlabel(r'Time (sec)')
    axarr[5].set_ylabel(r'Second moments (nm$^2$)')
    axarr[5].set_xlim(left=0, right=ME_anal.time[-1])
    axarr[5].set_ylim(np.amin(ME_anal.mu_kl), np.amax(ME_anal.mu_kl))

    # Draw rods
    graph_2d_rod_diagram(axarr[0], ME_anal, n)

    # # Make crosslinker density plot
    # c = graph_xl_dens(axarr[1],
    #                   ME_anal.xl_distr[:, :, n],
    #                   ME_anal.s1,
    #                   ME_anal.s2,
    #                   max_dens_val=ME_anal.max_dens_val)
    if ME_anal.init_flag:
        axarr[0].set_aspect(1.0)
        # axarr[1].set_aspect(1.0)
        # fig.colorbar(c, ax=axarr[1])
        ME_anal.init_flag = False

    # Graph rod center separations
    graph_vs_time(axarr[1], ME_anal.time, ME_anal.dR_arr, n)
    # Graph angle between rod orientations
    graph_vs_time(axarr[2], ME_anal.time, ME_anal.phi_arr, n)
    # Graph zeroth moment aka number of crosslinkers
    graph_vs_time(axarr[3], ME_anal.time, ME_anal.rho, n)
    # Graph first moments of crosslink distribution
    graph_vs_time(axarr[4], ME_anal.time, ME_anal.P1, n,
                  color='tab:green')
    graph_vs_time(axarr[4], ME_anal.time, ME_anal.P2, n,
                  color='tab:purple')
    # Graph second moments of crosslinker distribution
    graph_vs_time(axarr[5], ME_anal.time, ME_anal.u11, n,
                  color='b')
    graph_vs_time(axarr[5], ME_anal.time, ME_anal.u20, n,
                  color='tab:green')
    graph_vs_time(axarr[5], ME_anal.time, ME_anal.u02, n,
                  color='tab:purple')

    # Legend information
    axarr[1].legend([r"$\Delta$R({:.2f}) = {:.1f} nm".format(
        ME_anal.time[n], ME_anal.dR_arr[n])])
    axarr[2].legend([r"$\phi$({:.2f}) = {:.1f} rad".format(
        ME_anal.time[n], ME_anal.phi_arr[n])])
    axarr[3].legend([r"$\rho$({:.2f})={:.1f}".format(
        ME_anal.time[n], ME_anal.rho[n])])
    # axarr[3].legend(["F$_1$({:.2f}) = {:.1f}".format(ME_anal.time[n],
    #                                                  ME_anal.force_arr[n, 0]),
    #                  "F$_2$({:.2f}) = {:.1f}".format(ME_anal.time[n],
    #                                                  ME_anal.force_arr[n, 1])])
    # axarr[4].legend(["$T_1$({:.2f}) = {:.1f}".format(ME_anal.time[n],
    #                                                  ME_anal.torque_arr[n, 0]),
    #                  "$T_2$({:.2f}) = {:.1f}".format(ME_anal.time[n],
    # ME_anal.torque_arr[n, 1])])
    axarr[4].legend(["P$_1$({:.2f}) = {:.1f}".format(ME_anal.time[n],
                                                     ME_anal.P1[n]),
                     "P$_2$({:.2f}) = {:.1f}".format(ME_anal.time[n],
                                                     ME_anal.P2[n])])
    axarr[5].legend([r"$\mu^{{11}}$({:.2f}) = {:.1f}".format(ME_anal.time[n],
                                                             ME_anal.u11[n]),
                     r"$\mu^{{20}}$({:.2f}) = {:.1f}".format(ME_anal.time[n],
                                                             ME_anal.u20[n]),
                     r"$\mu^{{02}}$({:.2f}) = {:.1f}".format(ME_anal.time[n],
                                                             ME_anal.u02[n])])
    return fig.gca().lines + fig.gca().collections


def fp_graph_all_data_2d(fig, axarr, n, FP_anal):
    # Clean up if lines
    if not FP_anal.init_flag:
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
    axarr[5].set_ylabel(r'First moments (nm)')
    axarr[5].set_xlim(left=0, right=FP_anal.time[-1])
    axarr[5].set_ylim(min(np.amin(FP_anal.P1), np.amin(FP_anal.P2)),
                      max(np.amax(FP_anal.P1), np.amax(FP_anal.P2)))

    axarr[6].set_xlabel(r'Time (sec)')
    axarr[6].set_ylabel('Distance between MTs \n centers of mass (nm)')
    axarr[6].set_xlim(left=0, right=FP_anal.time[-1])
    axarr[6].set_ylim(np.amin(FP_anal.dR_arr),
                      np.amax(FP_anal.dR_arr))

    axarr[7].set_xlabel(r'Time (sec)')
    axarr[7].set_ylabel('Angle between MT \n orientation vectors (rad)')
    axarr[7].set_xlim(left=0, right=FP_anal.time[-1])
    axarr[7].set_ylim(np.nanmin(FP_anal.phi_arr),
                      np.nanmax(FP_anal.phi_arr))

    axarr[8].set_xlabel(r'Time (sec)')
    axarr[8].set_ylabel(r'Second moments (nm$^2$)')
    axarr[8].set_xlim(left=0, right=FP_anal.time[-1])
    axarr[8].set_ylim(min(np.amin(FP_anal.u11),
                          np.amin(FP_anal.u20),
                          np.amin(FP_anal.u02)),
                      max(np.amax(FP_anal.u11),
                          np.amax(FP_anal.u20),
                          np.amax(FP_anal.u02)))

    # Draw rods
    graph_2d_rod_diagram(axarr[0], FP_anal, n)

    # Make crosslinker density plot
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

    # Graph zeroth moment aka number of crosslinkers
    graph_vs_time(axarr[2], FP_anal.time, FP_anal.Nxl_arr, n)
    # Graph forces
    graph_vs_time(axarr[3], FP_anal.time, FP_anal.force_arr[:, 0], n,
                  color='tab:green')
    graph_vs_time(axarr[3], FP_anal.time, FP_anal.force_arr[:, 1], n,
                  color='tab:purple')
    # Graph torques
    graph_vs_time(axarr[4], FP_anal.time, FP_anal.torque_arr[:, 0], n,
                  color='tab:green')
    graph_vs_time(axarr[4], FP_anal.time, FP_anal.torque_arr[:, 1], n,
                  color='tab:purple')
    # Graph first moments of crosslink distribution
    graph_vs_time(axarr[5], FP_anal.time, FP_anal.P1, n,
                  color='tab:green')
    graph_vs_time(axarr[5], FP_anal.time, FP_anal.P2, n,
                  color='tab:purple')
    # Graph rod center separations
    graph_vs_time(axarr[6], FP_anal.time, FP_anal.dR_arr, n)
    # Graph angle between rod orientations
    graph_vs_time(axarr[7], FP_anal.time, FP_anal.phi_arr, n)
    # Graph second moments of crosslinker distribution
    graph_vs_time(axarr[8], FP_anal.time, FP_anal.u11, n,
                  color='b')
    graph_vs_time(axarr[8], FP_anal.time, FP_anal.u20, n,
                  color='tab:green')
    graph_vs_time(axarr[8], FP_anal.time, FP_anal.u02, n,
                  color='tab:purple')

    # Legend information
    axarr[2].legend(["N({:.2f}) = {:.1f}".format(
        FP_anal.time[n], FP_anal.Nxl_arr[n])])
    axarr[3].legend([r"F$_1$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                      FP_anal.force_arr[n, 0]),
                     r"F$_2$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                      FP_anal.force_arr[n, 1])])
    axarr[4].legend([r"$T_1$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                      FP_anal.torque_arr[n, 0]),
                     r"$T_2$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                      FP_anal.torque_arr[n, 1])])
    axarr[5].legend([r"P$_1$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                      FP_anal.P1[n]),
                     r"P$_2$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                      FP_anal.P2[n])])
    axarr[6].legend([r"$\Delta$R({:.2f}) = {:.1f} nm".format(
        FP_anal.time[n], FP_anal.dR_arr[n])])
    axarr[7].legend([r"$\phi$({:.2f}) = {:.1f} rad".format(
        FP_anal.time[n], FP_anal.phi_arr[n])])
    axarr[8].legend([r"$\mu^{{11}}$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                             FP_anal.u11[n]),
                     r"$\mu^{{20}}$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                             FP_anal.u20[n]),
                     r"$\mu^{{02}}$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                             FP_anal.u02[n])])
    return fig.gca().lines + fig.gca().collections


def fp_graph_moment_data_2d(fig, axarr, n, FP_anal):
    # Clean up if lines
    if not FP_anal.init_flag:
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
    axarr[2].set_xlim(left=0, right=FP_anal.time[-1])
    axarr[2].set_ylim(np.amin(FP_anal.Nxl_arr),
                      np.amax(FP_anal.Nxl_arr))

    axarr[3].set_xlabel(r'Time (sec)')
    axarr[3].set_ylabel(r'First moments (nm)')
    axarr[3].set_xlim(left=0, right=FP_anal.time[-1])
    axarr[3].set_ylim(min(np.amin(FP_anal.P1), np.amin(FP_anal.P2)),
                      max(np.amax(FP_anal.P1), np.amax(FP_anal.P2)))

    axarr[4].set_xlabel(r'Time (sec)')
    axarr[4].set_ylabel(r'Second moments (nm$^2$)')
    axarr[4].set_xlim(left=0, right=FP_anal.time[-1])
    axarr[4].set_ylim(min(np.amin(FP_anal.u11),
                          np.amin(FP_anal.u20),
                          np.amin(FP_anal.u02)),
                      max(np.amax(FP_anal.u11),
                          np.amax(FP_anal.u20),
                          np.amax(FP_anal.u02)))

    axarr[5].set_xlabel(r'Time (sec)')
    axarr[5].set_ylabel(r'Total crosslinker force (pN)')
    axarr[5].set_xlim(left=0, right=FP_anal.time[-1])
    axarr[5].set_ylim(np.amin(FP_anal.force_arr),
                      np.amax(FP_anal.force_arr))

    axarr[6].set_xlabel(r'Time (sec)')
    axarr[6].set_ylabel(r'Total crosslinker torque (pN*nm)')
    axarr[6].set_xlim(left=0, right=FP_anal.time[-1])
    axarr[6].set_ylim(np.amin(FP_anal.torque_arr),
                      np.amax(FP_anal.torque_arr))

    # Draw rods
    graph_2d_rod_diagram(axarr[0], FP_anal, n)

    # Make crosslinker density plot
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

    # Graph zeroth moment aka number of crosslinkers
    graph_vs_time(axarr[2], FP_anal.time, FP_anal.Nxl_arr, n)
    # Graph first moments of crosslink distribution
    graph_vs_time(axarr[3], FP_anal.time, FP_anal.P1, n,
                  color='tab:green')
    graph_vs_time(axarr[3], FP_anal.time, FP_anal.P2, n,
                  color='tab:purple')
    # Graph second moments of crosslinker distribution
    graph_vs_time(axarr[4], FP_anal.time, FP_anal.u11, n,
                  color='b')
    graph_vs_time(axarr[4], FP_anal.time, FP_anal.u20, n,
                  color='tab:green')
    graph_vs_time(axarr[4], FP_anal.time, FP_anal.u02, n,
                  color='tab:purple')
    # Graph forces
    graph_vs_time(axarr[5], FP_anal.time, FP_anal.force_arr[:, 0], n,
                  color='tab:green')
    graph_vs_time(axarr[5], FP_anal.time, FP_anal.force_arr[:, 1], n,
                  color='tab:purple')
    # Graph torques
    graph_vs_time(axarr[6], FP_anal.time, FP_anal.torque_arr[:, 0], n,
                  color='tab:green')
    graph_vs_time(axarr[6], FP_anal.time, FP_anal.torque_arr[:, 1], n,
                  color='tab:purple')
    # Legend information
    axarr[2].legend([r"N({:.2f}) = {:.1f}".format(
        FP_anal.time[n], FP_anal.Nxl_arr[n])])
    axarr[3].legend([r"P$_1$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                      FP_anal.P1[n]),
                     r"P$_2$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                      FP_anal.P2[n])])
    axarr[4].legend([r"$\mu^{{11}}$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                             FP_anal.u11[n]),
                     r"$\mu^{{20}}$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                             FP_anal.u20[n]),
                     r"$\mu^{{02}}$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                             FP_anal.u02[n])])
    axarr[5].legend([r"F$_1$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                      FP_anal.force_arr[n, 0]),
                     r"F$_2$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                      FP_anal.force_arr[n, 1])])
    axarr[6].legend([r"$T_1$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                      FP_anal.torque_arr[n, 0]),
                     r"$T_2$({:.2f}) = {:.1f}".format(FP_anal.time[n],
                                                      FP_anal.torque_arr[n, 1])])
    return fig.gca().lines + fig.gca().collections


def fp_graph_mts_xlink_distr_2d(fig, axarr, n, FP_anal):
    # Clean up if lines
    if not FP_anal.init_flag:
        for ax in axarr.flatten():
            ax.clear()
        for artist in fig.gca().lines + fig.gca().collections:
            artist.remove()
            del artist

    # Draw rods
    graph_2d_rod_diagram(axarr[0], FP_anal, n)

    # Make density plot
    c = graph_xl_dens(axarr[1],
                      FP_anal.xl_distr[:, :, n],
                      FP_anal.s1,
                      FP_anal.s2,
                      max_dens_val=FP_anal.max_dens_val)
    axarr[1].set_xlabel(
        'Head distance from \n center of MT$_1$ $s_1$ (nm)')
    axarr[1].set_ylabel(
        'Head distance from \n center of MT$_2$ $s_2$ (nm)')

    if FP_anal.init_flag:
        axarr[0].set_aspect(1.0)
        axarr[1].set_aspect(1.0)
        fig.colorbar(c, ax=axarr[1])
        FP_anal.init_flag = False
    axarr[0].text(.05, .95, "Time = {:.2f} sec".format(FP_anal.time[n]),
                  horizontalalignment='left',
                  verticalalignment='bottom',
                  transform=axarr[0].transAxes)

    # FP_anal.time[n])], facecolor='inherit')
    return fig.gca().lines + fig.gca().collections


def fp_graph_stationary_runs_2d(fig, axarr, n, FP_anal):
    # Clean up if lines
    if not FP_anal.init_flag:
        for ax in axarr.flatten():
            ax.clear()
        for artist in fig.gca().lines + fig.gca().collections:
            artist.remove()
            del artist

    # Draw rods
    graph_2d_rod_diagram(axarr[0], FP_anal, n)

    # Make density plot
    c = graph_xl_dens(axarr[1],
                      FP_anal.xl_distr[:, :, n],
                      FP_anal.s1,
                      FP_anal.s2,
                      max_dens_val=FP_anal.max_dens_val)
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

    if FP_anal.init_flag:
        axarr[0].set_aspect(1.0)
        axarr[1].set_aspect(1.0)
        fig.colorbar(c, ax=axarr[1])
        FP_anal.init_flag = False

    graph_vs_time(axarr[2], FP_anal.time, FP_anal.Nxl_arr, n)
    graph_vs_time(axarr[3], FP_anal.time, FP_anal.force_arr[:, 0], n,
                  color='tab:green')
    graph_vs_time(axarr[3], FP_anal.time, FP_anal.force_arr[:, 1], n,
                  color='tab:purple')
    graph_vs_time(axarr[4], FP_anal.time, FP_anal.torque_arr[:, 0], n,
                  color='tab:green')
    graph_vs_time(axarr[4], FP_anal.time, FP_anal.torque_arr[:, 1], n,
                  color='tab:purple')

    return fig.gca().lines + fig.gca().collections

######################################
#  Crosslinker distribution moments  #
######################################
