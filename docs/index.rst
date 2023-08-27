Simple Transforms: Tools for creating 3-D geometric transforms as 4x4 numpy matrices
====================================================================================

.. warning::
    This project is under active development - expect bugs

A geometric transformation is a mapping from a set of points in 3-D coordinate space
onto another set of points in the same 3-D coordinate space. All geometric objects, such
as points, lines, surfaces, solids can all be treated as sets of points. So geometric
transformations correspond to manipulations of geometric objects in 3-D space. This
project does not describe all possible 3-D transformations, but a subset of
transformations known as 'Projective Transformations' that can be represented by 4x4
matrices.

This project does not define any new classes, it creates all transforms as 4x4 numpy
matrices. The goal of this project is to create some tools for the creation and
manipulation of the 4x4 transformation matrices. Historically, this project came from
another project for solid modeling, which needed to work with geometric transformations
that works with a custom data-type. Because it was useful outside of the larger project
both the custom data type and the simple transforms were separated off into their own
projects.

.. todo::
    Update the custom numpy universal functions to work with built in (non-flint) data 
    types.

.. todo::
    Figure out how to implement optional behavior so that this tool can be use without
    the 'numpy-flint' package.

.. todo::
    Write a primer for linear, affine, and projective transformations as matrices.

.. todo::
    Create visuals demonstraighting each of the transformations supported

.. todo::
    Add projective transformations to matrix creation routines

.. todo::
    Add a 'decompose' function to take a general transformation and break it down into
    'pure' transformations using singular value decomposition

.. todo::
    Add functions to 'invert' most of the creation routines to recover the parameters
    used from the 4x4 matrix.

Simple Transform API
--------------------

Transform Matrix Creation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: simple_transforms.eye

.. autofunction:: simple_transforms.from_mat

.. autofunction:: simple_transforms.trans

.. autofunction:: simple_transforms.scale

.. autofunction:: simple_transforms.rot

.. autofunction:: simple_transforms.refl

.. autofunction:: simple_transforms.skew

Transform Matrix Tools
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: simple_transforms.combine

.. autofunction:: simple_transforms.transform_reduce

.. autofunction:: simple_transforms.apply
