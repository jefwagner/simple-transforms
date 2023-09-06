Simple-Transforms API
=====================


Transform Matrix Creation
-------------------------

.. todo::
    Create visuals demonstrating each of the transformations

.. autofunction:: simple_transforms.eye

.. autofunction:: simple_transforms.from_mat

.. autofunction:: simple_transforms.trans

.. autofunction:: simple_transforms.scale

.. autofunction:: simple_transforms.rot

.. autofunction:: simple_transforms.refl

.. autofunction:: simple_transforms.skew

.. todo::
    Add projective transformations to matrix creation routines


Transform Matrix Tools
----------------------

.. autofunction:: simple_transforms.combine

.. autofunction:: simple_transforms.transform_reduce

.. autofunction:: simple_transforms.apply

.. todo::
    Add a 'decompose' function to take a general transformation and break it down into
    'pure' transformations using singular value decomposition

.. todo::
    Add functions to 'invert' most of the creation routines to recover the parameters
    used from the 4x4 matrix.
