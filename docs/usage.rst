Installation and usage
======================

Installation
------------

Binary packages are available from `PyPI
<https://pypi.org/project/simple-transforms/>`_ and can be installed using pip.

.. prompt:: bash

    pip install simple-transforms


The package was built with support for the ``flint`` datatype from the 
`numpy-flint <https://jefwagner.github.io/flint>`_ package. You can install
that package along with this as an optional dependency.

.. prompt:: bash

    pip install simple-transforms[flint]


Usage
-----

The package is imported as usual.

.. code-block:: python

    import simple_transforms as transforms

You can create transformation matrices using any of the matrix creation
routines. The result is a 4x4 NumPy array.

.. code-block:: python

    a = transforms.rot('z', np.pi/2)
    print(a)

The ``apply`` function can be used to apply a transform at any array-like set
of 3-d vertices or 4-d vertices in homogenous coordinates

.. code-block:: python

    points = [(i, 0, 0) for i in range(10)]
    a = transforms.rot('z', np.pi/2)
    rot_points = transforms.apply(a, points)
    print(rot_points)
