#!/usr/bin/env python
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

"""@package docstring
File: animation_funcs.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


def makeAnimation(FPanal, writer=FFMpegWriter):
    """!Make animation of time slices
    @return: TODO

    """
    fig = plt.figure(constrained_layout=True, figsize=(15, 13))
    graph_stl = {
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "font.size": 15
    }
    with plt.style.context(graph_stl):
        plt.style.use(graph_stl)
        gs = fig.add_gridspec(3, 12)
        axarr = np.asarray([fig.add_subplot(gs[0, :4]),
                            fig.add_subplot(gs[0, 4:8]),
                            fig.add_subplot(gs[0, 8:]),
                            fig.add_subplot(gs[1, :6]),
                            fig.add_subplot(gs[1, 6:]),
                            fig.add_subplot(gs[2, :6]),
                            fig.add_subplot(gs[2, 6:]),
                            ])
        fig.suptitle(' ')
        # FPanal.graphSlice(50, fig, axarr)
        # plt.show()
        nframes = FPanal.time.size
        anim = FuncAnimation(
            fig,
            FPanal.graphSlice,
            frames=np.arange(nframes),
            fargs=(fig, axarr),
            interval=50,
            blit=True)
    t0 = time.time()

    anim.save('{}.mp4'.format(Path.cwd().name), writer=writer)
    t1 = time.time()
    print("Movie saved in: ", t1 - t0)


def makeMinimalAnimation(FPanal, writer=FFMpegWriter):
    """!Make animation of time slices
    @return: TODO

    """
    graph_stl = {
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "font.size": 18
    }
    with plt.style.context(graph_stl):
        plt.style.use(graph_stl)
        fig = plt.figure(figsize=(10, 5), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        axarr = np.asarray([fig.add_subplot(gs[0, 0]),
                            fig.add_subplot(gs[0, 1]), ])
        fig.suptitle(' ')
        nframes = FPanal.time.size
        anim = FuncAnimation(
            fig,
            FPanal.graphReducedSlice,
            frames=np.arange(nframes),
            fargs=(fig, axarr),
            interval=50,
            blit=True)
    t0 = time.time()

    anim.save('{}_min.mp4'.format(Path.cwd().name), writer=writer)
    t1 = time.time()
    print("Movie saved in: ", t1 - t0)


def makeOrientAnimation(FPanal, writer=FFMpegWriter):
    """!Make animation of time slices
    @return: TODO

    """
    fig = plt.figure(constrained_layout=True, figsize=(15, 9))
    graph_stl = {
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "font.size": 15
    }
    with plt.style.context(graph_stl):
        plt.style.use(graph_stl)
        gs = fig.add_gridspec(2, 12)
        axarr = np.asarray([fig.add_subplot(gs[0, :4]),
                            fig.add_subplot(gs[0, 4:8]),
                            fig.add_subplot(gs[0, 8:]),
                            fig.add_subplot(gs[1, :6]),
                            fig.add_subplot(gs[1, 6:]),
                            ])
        fig.suptitle(' ')
        nframes = FPanal.time.size
        anim = FuncAnimation(
            fig,
            FPanal.graphOrientSlice,
            frames=np.arange(nframes),
            fargs=(fig, axarr),
            interval=50,
            blit=True)
    t0 = time.time()

    anim.save('{}_orient.mp4'.format(Path.cwd().name), writer=writer)
    t1 = time.time()
    print("Movie saved in: ", t1 - t0)