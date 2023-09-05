Simple Transforms: Tools for creating 3-D geometric transforms as 4x4 numpy matrices
====================================================================================

.. warning::
    This project is under active development - expect bugs

This project does not define any new classes, it creates all transforms as 4x4 numpy
matrices. The goal of this project is to create some tools for the creation and
manipulation of the 4x4 transformation matrices. Historically, this project came from
another project for solid modeling, which needed to work with geometric transformations
that works with a custom data-type. Because it was useful outside of the larger project
both the custom data type and the simple transforms were separated off into their own
projects.

.. toctree::
    :maxdepth: 1

    usage
    api

Geometric transformations
-------------------------

`geometric transforms <https://en.wikipedia.org/wiki/Geometric_transformation>`_

A geometric transformation is a mapping from a set of points in 3-D coordinate space
onto another set of points in the same 3-D coordinate space. All geometric objects, such
as points, lines, surfaces, solids can all be treated as sets of points. So geometric
transformations correspond to manipulations of geometric objects in 3-D space. 

Linear transformation as matrices
---------------------------------

A point can be represented as a 3 component vector of coordinates (vx, vy, vz).
Matrx multiplication is mapping from 

Affine transformations and homogenous coordinates
-------------------------------------------------

Projection transformations
--------------------------

.. todo::
    Write a primer for linear, affine, and projective transformations as matrices.

