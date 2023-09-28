Primer on Transformation matrices
=================================

* Geometric transforms
    - Create shapes
    - Manipulate them by moving, rotating, stretching, twisting, etc.
    - The manipulations are called geometric transforms
    - Math - shape is a(n infinite) collection of points
    - Geometric transform is a function from points in space to new points in space

One of the primary task with working with shapes is manipulating them by moving,
rotating, stretching, twisting, etc. These different manipulations are called `geometric
transformations <https://en.wikipedia.org/wiki/Geometric_transformation>`_. A
certain very useful subset of those transformations are represented by 4x4 matrices for
3D transformation or 3x3 matrices for 2D transformations. This section attempts to
describe why the transformations are represented by matrices and what the limitations
for that particular representation are.

Lets start by describing 'shapes' and 'transformations' in a more precise notation. A
geometric shape can be considered the set of all points that lie in the shape. A
transformation is then a function that acts on each point of the ball, and transforms it
to another point in 3D space. A transform applied to a geometric object gives a new
transformed geometric object, defined as the set of all transformed points.

Lets go through a quick example with these very general definitions. We start by
describing the object as the set of points. A sphere (or more mathematically precisely,
a ball) is the set of all points whose distance from the center is less than or equal to
the radius, and is given by the definition in equation :eq:`sphere`. A transform is
function that acts on points, lets use an example of 'cubing' the coordinate as in 
equation :eq:`cube_transform`. Finally, the transform applied to the object is simply
the set of all the transformed points as shown in equation :eq:`cubed_sphere`. 

.. math::
    :label: sphere
 
    Sp = \{\vec{v} \,|\, |\vec{v}-\vec{c}| < r\}

.. math:: 
    :label: cube_transform

    T(\vec{v}) = \vec{v}(\vec{v}\cdot\vec{v})

.. math::
    :label: cubed_sphere

    T \circ Sp = \{T(\vec{v}) \,\forall \vec{v} \in Sp\}

.. todo::

    Make a visualization of the 'cube' transformation on a sphere.

The above definitions are very general and very powerful, but not the easiest to
implement with computers, and frequently do not lend themselves to the most efficient
algorithms. It becomes more useful if we both restrict the set of geometric objects we
work on and the set of transformations we can apply. 

For a start, let's examining how we'll restrict the description for the geometric
objects. The definition of the object as the set of all points on the object is clearly
not the most useful for computation; an infinite number of points is too many points. We
want to restrict the definition of an object to things that can be represented with
finite amount of data. The sphere above is a good example, it can be represented with
just the position of the center and a radius. But that definition does not lend itself
well to transformations either. Applying any transformation with any non-uniform
stretching the original sphere in such a way that it the resulting object will no longer
be represented by a center and single radius. 

The question becomes: how to we define shapes in a general enough way that still allows
us to apply transforms in a useful manner? Here is a simple proposal that we can start
with. The geometric object needs to be defined with a finite set of control points that
will transform with the transformation function [#f1]_. The prototypical example of such
an geometric object is line-segment defined by it's two endpoints. However, this clearly
does not work for all transformations. Two points always make a straight line, but
many(most?) transformations will transform straight lines into curves! So in order to
keep our nice useful description for geometric objects we'll have to limit what
geometric transforms we can use.

.. [#f1] This is not quite precise enough a definition. We end up requiring that the
   geometric object is defined as linear combinations of the control points. This covers
   the case of lines, or triangles, as well as some more complex shapes such as various
   kinds of `splines <https://en.wikipedia.org/wiki/Spline_(mathematics)>`_ [#f2]_.

.. [#f2] Splines are traditionally a 1-D curve, and the 2-D and 3-D versions are called 
   spline-surfaces or spline-volumes. This seems like a lost opportunity to make new
   words like 'splurface' or 'spvolume'

Now let's examine how we'll restrict the geometric transformations themselves. We'll
start by requiring the simplest of our restricted geometric objects - line segments
transform as their endpoint. All the points on a line can be defined with as a simple
linear interpolation between the points, as in equation :eq:`line_def`. The transform
applied to the equation should be the same as the equation applied to the transform
of the control points, as in equation :eq:`linear_trans`. This is recognizable as the
definition of a linear function. This gives us the first hint that we can use matrices
as transforms, since matrices are linear operators for vectors under the operation of
matrix-vector multiplication.

.. math::
    :label: line_def

    L = \{\vec{p} \,|\, \vec{p} = (1-t)\vec{p}_A + t \vec{p}_B \,\forall t \in [0,1]\}

.. math::
    :label: linear_trans

    T\big((1-t) \vec{p}_A + t \vec{p}_B\big) = (1-t) T(\vec{p}_A) + t T(\vec{p}_B)

Let's first take a moment to examine matrix vector multiplication and why it defines
linear operations on a vector. A linear function of multiple variables is one where the
maximum power of any variable is 1, and no two variables are multiplied together. A
point can be defined by the cartesian coordinates ([x,y] in 2D, [x,y,z] in 3D), and a
linear function from [x,y] to [x', y'] would take on the form seen in equation
:eq:`2d_linear_eqs`.

.. math::
    :label: 2d_linear_eqs

        x' & = a x + b y \\
        y' & = c x + d y 

* Math for the transformations
    - Points are [x,y,z] coordinates
    - transformed point x' can be linear combinations of x,y, and z
    - Looks like matrix vector multiplication
    - Scalings, Rotations, Reflections, and Skewing(Shearing)
    - All linear transformations leave [0,0,0] unchanged

* Adding in translations
    - Need to add an offset, neat trick : Homogenous coordinates
    - points get 1 extra coordinate (canonically labeled w)
    - All 4-D points along a line going through the 4-D origin correspond to the same 
      point.
    - Represent our 3-D points with [x,y,z,1] = [v,1]
    - Turn 3x3 matrix to 4x4 with [[A,b],[0,1]]
    - Matrix multiplication now gives [A.v + b, 1]
    - These are known as affine transforms

* Still not quite done
    - Two things stand out:
        - Affine transforms preserve straight lines - ALSO preserve parallel lines
        - We have bottom row of the transformation matrix unused
    - What happens when we use last row: look at the math
        - assume identity for affine transform
        - [x,y,z,w'] -> [x/w', y/w', z/w', 1] where w' = p.v + pw.
        - dot product finds the component of v = [x,y,z] in the direction of p
            - planes perpendicular to p will all the the same value of w'
            - shapes in the planes for larger values of w' uniformly scaled down more

