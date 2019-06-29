=======
FoXlink
=======


.. image:: https://img.shields.io/pypi/v/foxlink.svg
        :target: https://pypi.python.org/pypi/foxlink

.. image:: https://img.shields.io/travis/lamsoa729/foxlink.svg
        :target: https://travis-ci.org/lamsoa729/foxlink

.. image:: https://codecov.io/gh/lamsoa729/foxlink/branch/master/graph/badge.svg
        :target: https://codecov.io/gh/lamsoa729/foxlink

.. image:: https://readthedocs.org/projects/foxlink/badge/?version=latest
        :target: https://foxlink.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




FoXlink is an easy to use and modify partial differential equations solving package that describes the binding kinetics and motion of crosslinking proteins on filamentous biopolymers.


* Free software: BSD license
* Documentation: https://foxlink.readthedocs.io.


Features
--------

* Control structure callable from command line once package is installed.
* Object oriented design that allows for rapid prototyping of different system configurations and solving paradigms.
* All output data is written in HDF5 format to reduce storage space requirements while allowing for easy viewing and post-analysis using HDFview and python libraries (e.g. h5py)


Quickstart
----------

TODO

Design layout
-------------

Foxlink revolves around mulit-inheritance  to quickly create and combine PDE solving algorithms. All solving algorithms and specifics inherit from a common base class *Solver*.

.. code-block:: foxlink -f FP_params.yaml


Credits
-------

This package was created with Cookiecutter_ and the `pyOpenSci/cookiecutter-pyopensci`_ project template, based off `audreyr/cookiecutter-pypackage`_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`pyOpenSci/cookiecutter-pyopensci`: https://github.com/pyOpenSci/cookiecutter-pyopensci
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage


.. image:: https://api.codacy.com/project/badge/Grade/f72e009a2ce147a8b8c067fb24c0d6d4
   :alt: Codacy Badge
   :target: https://app.codacy.com/app/lamsoa729/FoXlink?utm_source=github.com&utm_medium=referral&utm_content=lamsoa729/FoXlink&utm_campaign=Badge_Grade_Dashboard
