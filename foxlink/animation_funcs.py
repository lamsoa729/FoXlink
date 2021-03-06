#!/usr/bin/env python
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

"""@package docstring
File: animation_funcs.py
Author: Adam Lamson
Email: adam.lamson@colorado.edu
Description:
"""


def make_animation(pde_anal, writer='ffmpeg', save_path=Path('./')):
    """!Make animation of time slices, graphing rod diagram, xlink density,
    xlink number, xlink force, xlink torque, moments up to 2nd order,
    distance between rod centers, and angle between rod orientation vectors.

    @param pde_anal: Analyzer object storing PDE data.
    @param writer: Matplotlib Writer object or string dictating how to create
                   and save animation.
    @param save_path: pathlib.Path object for where animation will be saved
    @return: None

    """
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    fig = plt.figure(constrained_layout=True, figsize=(15, 13))
    graph_stl = {
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "font.size": 15,
        "font.sans-serif": 'Helvetica',
        "text.usetex": False,
        'mathtext.fontset': 'cm',
    }
    with plt.style.context(graph_stl):
        plt.style.use(graph_stl)
        gs = fig.add_gridspec(3, 3)
        axarr = np.asarray([fig.add_subplot(gs[0, 0]),
                            fig.add_subplot(gs[0, 1]),
                            fig.add_subplot(gs[0, 2]),
                            fig.add_subplot(gs[1, 0]),
                            fig.add_subplot(gs[1, 1]),
                            fig.add_subplot(gs[1, 2]),
                            fig.add_subplot(gs[2, 0]),
                            fig.add_subplot(gs[2, 1]),
                            fig.add_subplot(gs[2, 2]),
                            ])
        fig.suptitle(' ')
        nframes = pde_anal.time.size
        anim = FuncAnimation(
            fig,
            pde_anal.graph_slice,
            frames=np.arange(nframes),
            fargs=(fig, axarr),
            interval=50,
            blit=True)
    t0 = time.time()

    anim.save(save_path / '{}.mp4'.format(pde_anal.get_name()), writer=writer)
    t1 = time.time()
    print("Movie saved in: ", t1 - t0)


def make_minimal_pde_animation(pde_anal, writer='ffmpeg',
                               save_path=Path('./')):
    """!Make an animation of time slices with only moving rods and xlink
        density distribution.

    @param pde_anal: Analyzer object storing PDE data.
    @param writer: Matplotlib Writer object or string dictating how to create
                   and save animation.
    @param save_path: pathlib.Path object for where animation will be saved
    @return: None

    """
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    graph_stl = {
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "font.size": 18,
        "font.sans-serif": 'Helvetica',
        "text.usetex": False,
        'mathtext.fontset': 'cm',
    }

    with plt.style.context(graph_stl):
        plt.style.use(graph_stl)
        fig = plt.figure(figsize=(10, 5), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        axarr = np.asarray([fig.add_subplot(gs[0, 0]),
                            fig.add_subplot(gs[0, 1]), ])
        fig.suptitle(' ')
        nframes = pde_anal.time.size
        anim = FuncAnimation(
            fig,
            pde_anal.graph_reduced_slice,
            frames=np.arange(nframes),
            fargs=(fig, axarr),
            interval=50,
            blit=True)
    t0 = time.time()

    anim.save(save_path / '{}_min.mp4'.format(pde_anal.get_name()),
              writer=writer)
    t1 = time.time()
    print("Movie saved in: ", t1 - t0)


def make_distr_pde_animation(pde_anal, writer='ffmpeg', save_path=Path('./')):
    """!Make an animation of time slices with moving rods, xlink density
    distribution, and recreated density distribution from moments.

    @param pde_anal: Analyzer object storing PDE data.
    @param writer: Matplotlib Writer object or string dictating how to create
                   and save animation.
    @param save_path: pathlib.Path object for where animation will be saved
    @return: None

    """
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    graph_stl = {
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "font.size": 18,
        "font.sans-serif": 'Helvetica',
        "text.usetex": False,
        'mathtext.fontset': 'cm',
    }
    with plt.style.context(graph_stl):
        plt.style.use(graph_stl)
        fig = plt.figure(figsize=(18, 5), constrained_layout=True)
        gs = fig.add_gridspec(1, 3)
        axarr = np.asarray([fig.add_subplot(gs[0, 0]),
                            fig.add_subplot(gs[0, 1]),
                            fig.add_subplot(gs[0, 2]), ])

        fig.suptitle(' ')
        nframes = pde_anal.time.size
        anim = FuncAnimation(
            fig,
            pde_anal.graph_distr_slice,
            frames=np.arange(nframes),
            fargs=(fig, axarr),
            interval=50,
            blit=True)
    t0 = time.time()

    anim.save(
        save_path /
        '{}_distr.mp4'.format(
            pde_anal.get_name()),
        writer=writer)
    t1 = time.time()
    print("Movie saved in: ", t1 - t0)


def make_stat_pde_animation(
        pde_anal, writer='ffmpeg', save_path=Path('. /')):
    """!Make an animation of time slices for stationary rods. Rod diagram,
    xlink density, xlink number, xlink force, and xlink force are graphed.

    @param pde_anal: Analyzer object storing PDE data.
    @param writer: Matplotlib Writer object or string dictating how to create
                   and save animation.
    @param save_path: pathlib.Path object for where animation will be saved
    @return: None

    """
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    fig = plt.figure(constrained_layout=True, figsize=(15, 9))
    graph_stl = {
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "font.size": 15,
        "font.sans-serif": 'Helvetica',
        "text.usetex": False,
        'mathtext.fontset': 'cm',
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
        nframes = pde_anal.time.size
        anim = FuncAnimation(
            fig,
            pde_anal.graph_orient_slice,
            frames=np.arange(nframes),
            fargs=(fig, axarr),
            interval=50,
            blit=True)
    t0 = time.time()

    anim.save(
        save_path /
        '{}_stat.mp4'.format(
            pde_anal.get_name()),
        writer=writer)
    t1 = time.time()
    print("Movie saved in: ", t1 - t0)


def make_moment_pde_animation(pde_anal, writer='ffmpeg',
                              save_path=Path('. /')):
    """!Make animation time slices with rod diagram, xlink density, xlink force,
    xlink torque, and moments up to 2nd order.

    @param pde_anal: Analyzer object storing PDE data.
    @param writer: Matplotlib Writer object or string dictating how to create
                   and save animation.
    @param save_path: pathlib.Path object for where animation will be saved
    @return: None

    """
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    fig = plt.figure(constrained_layout=True, figsize=(12, 15))
    graph_stl = {
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "font.size": 13,
        "font.sans-serif": 'Helvetica',
        "text.usetex": False,
        'mathtext.fontset': 'cm',
    }
    with plt.style.context(graph_stl):
        plt.style.use(graph_stl)
        gs = fig.add_gridspec(10, 2)
        axarr = np.asarray([fig.add_subplot(gs[0:4, 0]),
                            fig.add_subplot(gs[0:4, 1]),
                            fig.add_subplot(gs[4:6, 0]),
                            fig.add_subplot(gs[6:8, 0]),
                            fig.add_subplot(gs[8:, 0]),
                            fig.add_subplot(gs[4:7, 1]),
                            fig.add_subplot(gs[7:, 1]),
                            ])
        fig.suptitle(' ')
        nframes = pde_anal.time.size
        anim = FuncAnimation(
            fig,
            pde_anal.graph_moment_slice,
            frames=np.arange(nframes),
            fargs=(fig, axarr),
            interval=50,
            blit=True)
    t0 = time.time()

    anim.save(
        save_path /
        '{}_moment.mp4'.format(
            pde_anal.get_name()),
        writer=writer)
    t1 = time.time()
    print("Movie saved in: ", t1 - t0)


def make_moment_expansion_animation(me_anal, writer='ffmpeg',
                                    save_path=Path('./')):
    """!Make animation time slices from moment expansion data, graphing rod
    diagram, xlink force, xlink torque, and moments up to 2nd order.

    @param me_anal: Analyzer object storing ME data.
    @param writer: Matplotlib Writer object or string dictating how to create
                   and save animation.
    @param save_path: pathlib.Path object for where animation will be saved
    @return: None

    """
    if not isinstance(save_path, Path):
        save_path = Path(save_path)
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    graph_stl = {
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "font.size": 13,
        "font.sans-serif": 'Helvetica',
        "text.usetex": False,
        'mathtext.fontset': 'cm',
    }
    with plt.style.context(graph_stl):
        plt.style.use(graph_stl)
        gs = fig.add_gridspec(2, 3)
        axarr = np.asarray([fig.add_subplot(gs[0, 0]),
                            fig.add_subplot(gs[0, 1]),
                            fig.add_subplot(gs[0, 2]),
                            fig.add_subplot(gs[1, 0]),
                            fig.add_subplot(gs[1, 1]),
                            fig.add_subplot(gs[1, 2]),
                            ])
        fig.suptitle(' ')
        nframes = me_anal.time.size
        print(nframes)
        anim = FuncAnimation(
            fig,
            me_anal.graph_slice,
            frames=np.arange(nframes),
            fargs=(fig, axarr),
            interval=50,
            blit=True)
    t0 = time.time()

    anim.save(save_path /
              '{}_ME_{}.mp4'.format(me_anal.get_name(), me_anal.graph_type),
              writer=writer)
    t1 = time.time()
    print("Movie saved in: ", t1 - t0)


def make_moment_distr_animation(
        me_anal, writer='ffmpeg', save_path=Path('./')):
    """!Make animation time slices from moment expansion data, graphing rod
    diagram, recreated xlink density, xlink force, xlink torque, and moments up
    to 2nd order.

    @param me_anal: Analyzer object storing ME data.
    @param writer: Matplotlib Writer object or string dictating how to create
                   and save animation.
    @param save_path: pathlib.Path object for where animation will be saved
    @return: None

    """
    fig = plt.figure(constrained_layout=True, figsize=(12, 10))
    graph_stl = {
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "font.size": 13,
        "font.sans-serif": 'Helvetica',
        "text.usetex": False,
        'mathtext.fontset': 'cm',

    }
    with plt.style.context(graph_stl):
        plt.style.use(graph_stl)
        gs = fig.add_gridspec(3, 3)
        axarr = np.asarray([fig.add_subplot(gs[0, 0]),
                            fig.add_subplot(gs[0, 1]),
                            fig.add_subplot(gs[0, 2]),
                            fig.add_subplot(gs[1, 0]),
                            fig.add_subplot(gs[1, 1]),
                            fig.add_subplot(gs[1, 2]),
                            fig.add_subplot(gs[2, 0]),
                            fig.add_subplot(gs[2, 1]),
                            fig.add_subplot(gs[2, 2]),
                            ])
        fig.suptitle(' ')
        nframes = me_anal.time.size
        print(nframes)
        anim = FuncAnimation(
            fig,
            me_anal.graph_distr_slice,
            frames=np.arange(nframes),
            fargs=(fig, axarr),
            interval=50,
            blit=True)
    t0 = time.time()

    anim.save(save_path /
              '{}_ME_{}.mp4'.format(me_anal.get_name(), me_anal.graph_type),
              writer=writer)
    t1 = time.time()
    print("Movie saved in: ", t1 - t0)


def make_moment_min_animation(me_anal, writer='ffmpeg', save_path=Path('./')):
    """!Make animation time slices from moment expansion data, graphing rod
    diagram and recreated xlink density.

    @param me_anal: Analyzer object storing ME data.
    @param writer: Matplotlib Writer object or string dictating how to create
                   and save animation.
    @param save_path: pathlib.Path object for where animation will be saved
    @return: None

    """
    # fig = plt.figure(constrained_layout=True, figsize=(10, 4))
    graph_stl = {
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "font.size": 18,
        "font.sans-serif": 'Helvetica',
        "text.usetex": False,
        'mathtext.fontset': 'cm',
    }
    # with plt.style.context(graph_stl):
    plt.style.use(graph_stl)
    fig = plt.figure(figsize=(13, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)
    axarr = np.asarray([fig.add_subplot(gs[0, 0]),
                        fig.add_subplot(gs[0, 1]), ])
    fig.suptitle(' ')
    nframes = me_anal.time.size
    print(nframes)
    anim = FuncAnimation(
        fig,
        me_anal.graph_min_slice,
        frames=np.arange(nframes),
        fargs=(fig, axarr),
        interval=50,
        blit=True)
    t0 = time.time()

    anim.save(save_path /
              '{}_ME_{}.mp4'.format(me_anal.get_name(), me_anal.graph_type),
              writer=writer)
    t1 = time.time()
    print("Movie saved in: ", t1 - t0)
