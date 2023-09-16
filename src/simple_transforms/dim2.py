## @file affine.py 
"""
"""
# Copyright (c) 2023, Jef Wagner <jefwagner@gmail.com>
#
# This file is part of simple-transforms.
#
# simple-transforms is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# simple-transforms is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# simple-transforms. If not, see <https://www.gnu.org/licenses/>.
from typing import Union, Optional, List

import numpy as np
import numpy.typing as npt
try:
    from flint import flint
    DEFAULT_DTYPE = flint
except ModuleNotFoundError:
    DEFAULT_DTYPE = np.float64

from ._c_trans import rescale2, apply_vert2

NumLike = Union[int, float, flint]

def eye(dtype: npt.DTypeLike = DEFAULT_DTYPE) -> npt.NDArray:
    """Create an identify affine transform"""
    return np.eye(3, dtype=dtype)

def from_mat(mat: npt.ArrayLike, dtype: npt.DTypeLike = DEFAULT_DTYPE) -> npt.NDArray:
    """Create a new generic affine transform from a 3x3, 2x3 or 2x2 matrix
    
    * A 2x2 matrix will only specify the linear transformation.
    * A 2x3 matrix will specify the linear transformation and translation.
    * A 3x3 will specify the linear transformation, translation, and perspective
        transformation.
    
    :param mat: The input matrix (any properly shaped nested sequence type).
    
    :return: An AffineTransform object corresponding to the matrix"
    """
    mat = np.array(mat, dtype=dtype)
    if mat.shape not in ((3,3),(2,3),(2,2)):
        raise ValueError('Argument must be a 3x3, 2x3, or 2x2 array')
    trans = eye(dtype=dtype)
    I, J = mat.shape
    for i in range(I):
        for j in range(J):
            trans[i,j] = mat[i,j]
    return trans

def _relocate_center(trans: npt.NDArray, center: npt.ArrayLike):
    """Relocate the center of a linear transformation"""
    center = np.array(center)
    if center.shape != (2,):
        raise ValueError('The center should be a 2 length [cx, cy]')
    a = trans[:2,:2]
    d = center - a.dot(center)
    for i in range(2):
        trans[i,2] = d[i]

def trans(d: npt.ArrayLike, 
          center: Optional[npt.NDArray] = None, 
          dtype: npt.DTypeLike = DEFAULT_DTYPE) -> npt.NDArray:
    """Create a new pure translation transformation.
    
    :param d: A 3-length sequence [dx, dy, dz]
    :param center: Ignored
    
    :return: An pure translation AffineTransformation.
    """
    d = np.array(d, dtype=dtype)
    if d.shape != (2,):
        raise ValueError('The translation argument `d` should be a 3 length [dx, dy, dz]')
    trans = eye(dtype=dtype)
    for i in range(2):
        trans[i,2] = d[i]
    return trans

def scale(s: Union[NumLike, npt.ArrayLike], 
          center: Optional[npt.NDArray] = None, 
          dtype: npt.DTypeLike = DEFAULT_DTYPE) -> npt.NDArray:
    """Create a new pure scaling transformation.
    
    :param s: A scalar or 2-length sequence [sx, sy] for scaling along each n
    :param center: Optional 2-length center position [cx, cy] for the scaling
        transform
    
    :return: A scaling if AffineTransformation."""
    s = np.array(s, dtype=dtype)
    # Scalar input
    if s.shape == ():
        s = np.array([s,s], dtype=dtype)
    if s.shape != (2,):
        raise ValueError('The scale argument `s` must be a scalar or a 2 length [sx, sy]')
    trans = eye(dtype=dtype)
    for i in range(2):
        trans[i,i] = s[i]
    if center is not None:
        _relocate_center(trans, center)
    return trans

def rot(angle: NumLike,
        center: Optional[npt.ArrayLike] = None,
        dtype: npt.DTypeLike = DEFAULT_DTYPE) -> npt.NDArray:
    """Create a new pure rotation transformation.
    
    :param angle: The angle in radians to rotate
    :param center: Optional 2-length position [cx, cy] for to specify the center of rotation
    
    :return: A rotation AffineTransformation."""
    angle = np.array(angle, dtype=dtype)
    if angle.shape != ():
        raise ValueError('Angle argument must be a scalar')
    s, c = np.sin(angle), np.cos(angle)
    trans = eye(dtype=dtype)
    trans[0,0] = c
    trans[0,1] = -s
    trans[1,0] = s
    trans[1,1] = c
    if center is not None:
        _relocate_center(trans, center)
    return trans

def refl(n: Union[str, npt.ArrayLike],
         center: Optional[npt.ArrayLike] = None,
         dtype: npt.DTypeLike = DEFAULT_DTYPE) -> npt.NDArray:
    """Create a new pure reflection transformation.
    
    :param normal: The character 'x','y','z' or a 3 length [ux, uy, uz] vector for
        the normal vector for the reflection plane.
    :param center: Optional 3-length center position [cx, cy, cz] a point on the
        plane of reflection operation.
    
    :return: A skew AffineTransformation.
    """
    if isinstance(n, str):
        if len(n) != 1 or n.lower()[0] not in ['x','y']:
            raise ValueError("n must be either the character 'x','y' or a two length vector [ax, ay]")
        n = n.lower()[0]
    else:
        n = np.array(n, dtype=dtype)
        if n.shape != (2,):
            raise ValueError("n must be either he character 'x','y' or a two length vector [ax, ay]")
        n_len = np.sqrt(np.sum(np.dot(n, n)))
        if n_len != 1:
            n = n/n_len
    trans = eye(dtype=dtype)
    if isinstance(n, str):
        if n == 'x':
            trans[0,0] = -1
        elif n == 'y':
            trans[1,1] = -1
        elif n == 'z':
            trans[2,2] = -1
    else:
        for i in range(2):
            for j in range(2):
                trans[i,j] -= 2*n[i]*n[j]
    if center is not None:
        _relocate_center(trans, center)
    return trans    

def skew(sv: Union[str, npt.ArrayLike],
         sa: Optional[NumLike] = None,
         center: Optional[npt.ArrayLike] = None,
         dtype: npt.DTypeLike = DEFAULT_DTYPE) -> npt.NDArray:
    """Create a new pure skew transformation.
    
    :param sv: The character 'x','y' or a 2 length [sx, sy] vector to define the skew 
        (shear) direction.
    :param sa: The skew amount
    :param center: Optional 3-length center position [cx, cy, cz] for the center of
        the skew operation.
    
    :return: A skew AffineTransformation."""
    trans = eye(dtype=dtype)
    if isinstance(sv, str):
        if sa is None:
            raise TypeError("sa required when using axis name")
        sa = np.array(sa, dtype=dtype)
        if sa.shape != ():
            raise ValueError("sa must be a scalar value")
        if len(sv) != 1 or sv.lower()[0] not in ['x','y']:
            raise ValueError("sv must be either the character 'x','y' or a two length vector [ax, ay]")
        if sv == 'x':
            trans[0,1] = sa
        elif sv == 'y':
            trans[1,0] = -sa
    else:
        sv = np.array(sv, dtype=dtype)
        if sv.shape != (2,):
            raise ValueError("sv must be either he character 'x','y' or a two length vector [ax, ay]")
        if sa is not None:
            raise TypeError("Can not specify skew amount if skew vector specified")
        n = np.zeros((2,), dtype=dtype)
        mag = np.sqrt(np.sum(np.dot(sv,sv)))
        n[0] = -sv[1]/mag
        n[1] = sv[0]/mag
        for i in range(2):
            for j in range(2):
                trans[i,j] += sv[i]*n[j]
    if center is not None:
        _relocate_center(trans, center)
    return trans

rescale = rescale2

def combine(lhs: npt.NDArray, rhs: npt.NDArray) -> npt.NDArray:
    """Combine two affine transforms into a single transform. 

    This is simply the matrix multiplication of the two transforms, and so the
        order of the two transforms matters. The resulting transform is the same
        as applying the right-hand-side transform first, then the
        left-hand-side.
    
    :param lhs: The left-hand-side affine transform
    :param rhs: The right-hand-side affine transform

    :return: The resulting combined transform
    """
    return np.dot(lhs, rhs)

def transform_reduce(transforms: List[npt.NDArray]) -> npt.NDArray:
    r"""Reduce a sequence of affine transforms into a single affine transform.

    This is the same as a repeated matrix multiplication, and so order of the
        transforms matter. The result is the same as the first transform applied
        followed by the second, and so on. A transform list `[T0, T1, T2, ...,
        TN]` would reduce to

    $T_{\text{reduced}} = T_N\cdot T_{N-1] \cdot \ldots \cdot T1 \cdot T0.$
    
    :param transforms: The sequence of affine transforms

    :return: The resulting reduced affine transform
    """
    out = eye()
    for tr in transforms:
        np.dot(tr, out, out=out)
    return out

def apply(transform: npt.NDArray, v_in: npt.ArrayLike) -> npt.NDArray:
    """Apply a transform to a single vertex or array of vertices.

    The vertex can either be a 3-length coordinates [x,y,z] or 4-length 
        homogeneous coordinates [x,y,z,1]. For a 3-length vertex the result
        is the same as it would be for the same homogenous coordinate.
    
    :param transform: The affine transform to apply
    :param v_in: The vertex or array of vertices

    :return: A new transformed vertex or array of transformed vertices
    """
    if not isinstance(v_in, np.ndarray):
        v_in = np.array(v_in)
    if len(v_in.shape) == 0:
        raise TypeError('foo')
    if v_in.shape[-1] not in [2,3]:
        raise ValueError('foo')
    out_dtype = flint if transform.dtype == flint or v_in.dtype == flint else np.float64
    v_out = np.empty(v_in.shape, dtype=out_dtype)
    if v_in.shape[-1] == 2:
        # apply for 2-length vertices
        apply_vert2(transform, v_in, v_out)
    else:
        # apply for 3-length homogenous coordinates
        v_out = np.inner(v_in, transform)
        rescale2(v_out, v_out)
    return v_out
