"""
.. module:: deriv
    :synopsis: Computes the derivatives of data fields using compact finite differences.

.. moduleauthor:: Paul Bartholomew <ptb08@ic.ac.uk>
"""

import numpy as np

from py4incompact3d.postprocess.fields import Field
from py4incompact3d.parallel.transpose import transpose

def tdma(a, b, c, rhs, overwrite=True):
    """ The Tri-Diagonal Matrix Algorithm.

    Solves tri-diagonal matrices using TDMA where the matrices are of the form
    [b0 c0
     a1 b1 c1
        a2 b2   c2

           an-2 bn-2 cn-1
                an-1 bn-1]

    :param a: The `left' coefficients.
    :param b: The diagonal coefficients. (All ones?)
    :param c: The 'right' coefficients.
    :param rhs: The right-hand side vector.
    :param overwrite: Should the rhs and diagonal coefficient (b) arrays be overwritten?

    :type a: numpy.ndarray
    :type b: numpy.ndarray
    :type c: numpy.ndarray
    :type rhs: numpy.ndarray
    :type overwrite: bool

    :returns: rhs -- the rhs vector overwritten with derivatives.
    :rtype: numpy.ndarray
    """

    if overwrite:
        bloc = b
        rhsloc = rhs
    else: # Creat local copies
        bloc = np.copy(b)
        rhsloc = np.copy(rhs)

    # # I've written this really dumb - quick fix, loop over first index
    # rhsloc = np.swapaxes(rhsloc, 0, 2)

    ni = rhsloc.shape[0]
    nj = rhsloc.shape[1]
    nk = rhsloc.shape[2]

    # First manipulate the diagonal
    bloc[1:] -= a[1:] * c[0:nk-1] / bloc[0:nk-1]

    for i in range(ni):
        for j in range(nj):
            # Forward elimination
            rhsloc[i,j,1:] -= (a[1:] / bloc[0:nk-1]) * rhsloc[i,j,0:nk-1]

            # Backward substitution
            rhsloc[i,j,-1] /= bloc[-1]
            rhsloc[i,j,nk-2:0:-1] -= c[nk-2:0:-1] * rhsloc[i,j,nk-1:1:-1] / bloc[nk-2:0:-1]
            k = 0
            rhsloc[i,j,k] -= c[k] * rhsloc[i,j,k + 1]
            rhsloc[i,j,k] /= bloc[k]

    # # I've written this really dumb - input expects result in last index
    # rhsloc = np.swapaxes(rhsloc, 0, 2)

    return rhsloc

def tdma_periodic(a, b, c, rhs):
    """ Periodic form of Tri-Diagonal Matrix Algorithm.

    Solves periodic tri-diagonal matrices using TDMA where the matrices are of the form
    [b0   c0           c1
     a1   b1 c1
          a2 b2   c2

             an-2 bn-2 cn-2
     cn-1         an-1 bn-1]

    :param a: The `left' coefficients.
    :param b: The diagonal coefficients. (All ones?)
    :param c: The 'right' coefficients.
    :param rhs: The right-hand side vector.

    :type a: numpy.ndarray
    :type b: numpy.ndarray
    :type c: numpy.ndarray
    :type rhs: numpy.ndarray

    :returns: rhs -- the rhs vector overwritten with derivatives.
    :rtype: numpy.ndarray
    """

    # Setup utility vectors u, v
    u = np.zeros(rhs.shape[2])
    v = np.zeros(rhs.shape[2])
    u[0] = -b[0]
    u[-1] = c[-1]
    v[0] = 1.0
    v[-1] = -a[0] / b[0]

    # Modify the diagonal -> A'
    b[-1] += (a[0] / b[0]) * c[-1]
    b[0] *= 2

    # Solve A'y=rhs, A'q=u
    # XXX don't overwrite the coefficient arrays!
    rhs = tdma(a, b, c, rhs, False)
    u = tdma(a, b, c, np.array([[u]]), False) # TDMA expects a 3D rhs 'vector'
    u = u[0][0]

    # Compute solution x = y - v^T y / (1 + v^T q) q
    vu = np.dot(v, u)
    u /= (1.0 + vu)
    for i in range(rhs.shape[0]):
        for j in range(rhs.shape[1]):
            rhs[i,j] -= np.dot(v, rhs[i,j]) * u

    return rhs

def compute_deriv(rhs, bc, npaire):
    """ Compute the derivative by calling to TDMA.

    :param rhs: The rhs vector.
    :param bc: The boundary condition for the axis.
    :param npaire: Does the field not 'point' in the same direction as the derivative?

    :type rhs: numpy.ndarray
    :type bc: int
    :type npaire: bool

    :returns: The derivative
    :rtype: numpy.ndarray"""

    a = (1.0 / 3.0) * np.ones(rhs.shape[2])
    b = np.ones(rhs.shape[2])
    c = a

    if bc == 0:
        # Periodic
        return tdma_periodic(a, b, c, rhs)
    else:
        if bc == 1:
            # Free slip
            if npaire:
                # 'even'
                a[-1] = 0.0
                c[0] = 0.0
            else:
                # 'odd'
                a[-1] *= 2
                c[0] *= 2
        else:
            #Dirichlet
            c[0] = 2.0
            a[1] = 0.25
            c[1] = 0.25
            a[-2] = 0.25
            c[-2] = 0.25
            a[-1] = 1.0
            b[-1] = 2.0
        return tdma(a, b, c, rhs)

def compute_rhs_0(mesh, field, axis):
    """ Compute the rhs for the derivative for periodic BCs.

    :param mesh: The mesh on which derivatives are taken.
    :param field: The field for the variable who's derivative we want.
    :param axis: A number indicating direction in which to take derivative: 0=x; 1=y; 2=z.

    :type mesh: py4incompact3d.postprocess.mesh.Mesh
    :type axis: int

    :returns: rhs -- the right-hand side vector.
    :rtype: numpy.ndarray
    """

    # Setup
    rhs = np.zeros([field.shape[0], field.shape[1], field.shape[2]])

    if axis == 0:
        invdx = 1.0 / mesh.dx
    elif axis == 1:
        invdx = 1.0 / mesh.dy
    else:
        invdx = 1.0 / mesh.dz

    a = mesh.a * invdx / 2.0
    b = mesh.b * invdx / 4.0

    # Compute RHS
    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            # XXX Due to python's negative indices, BC @ k = 0 automatically applied
            for k in range(field.shape[2] - 2):
                rhs[i,j,k] = a * (field[i,j,k+1] - field[i,j,k-1]) \
                               + b * (field[i,j,k+2] - field[i,j,k-2])

            # BCs @ k = n
            k = field.shape[2] - 2
            rhs[i,j,k] = a * (field[i,j,k+1] - field[i,j,k-1]) \
                           + b * (field[i,j,0] - field[i,j,k-2])
            k = field.shape[2] - 1
            rhs[i,j,k] = a * (field[i,j,0] - field[i,j,k-1]) \
                           + b * (field[i,j,1] - field[i,j,k-2])

    return rhs

def compute_rhs_1(mesh, field, axis, field_direction):
    """ Compute the rhs for the derivative for free slip BCs.

    :param mesh: The mesh on which derivatives are taken.
    :param field: The field for the variable who's derivative we want.
    :param axis: A number indicating direction in which to take derivative: 0=x; 1=y; 2=z.
    :param field_direction: Indicates the direction of the field: -1=scalar; 0=x; 1=y; 2=z.

    :type mesh: py4incompact3d.postprocess.mesh.Mesh
    :type field: np.ndarray
    :type axis: int
    :type field_direction: list of int

    :returns: rhs -- the right-hand side vector.
    :rtype: numpy.ndarray
    """

    # Setup
    rhs = np.zeros([field.shape[0], field.shape[1], field.shape[2]])

    if axis == 0:
        invdx = 1.0 / mesh.dx
    elif axis == 1:
        invdx = 1.0 / mesh.dy
    else:
        invdx = 1.0 / mesh.dz

    a = mesh.a * invdx / 2.0
    b = mesh.b * invdx / 4.0

    #BCs @ k = 0
    if axis not in field_direction:
        # npaire = 1
        k = 0
        rhs[:,:,k] = 0.0
        k = 1
        rhs[:,:,k] = a * (field[:,:,k+1] - field[:,:,k-1]) \
            + b * (field[:,:,k+2] - field[:,:,k])
    else:
        #npaire = 0
        k = 0
        rhs[:,:,k] = 2 * (a * field[:,:,k+1] + b * field[:,:,k+2])
        k = 1
        rhs[:,:,k] = a * (field[:,:,k+1] - field[:,:,k-1]) \
            + b * (field[:,:,k+2] + field[:,:,k])

    # Internal nodes
    n = field.shape[2]
    rhs[:,:,2:n-3] = a * (field[:,:,3:n-2] - field[:,:,1:n-4]) \
        + b * (field[:,:,4:n-1] - field[:,:,0:n-5])

    # BCs @ k = n
    if axis not in field_direction:
        # npaire = 1
        k = field.shape[2] - 2
        rhs[:,:,k] = a * (field[:,:,k+1] - field[:,:,k-1]) \
                               + b * (field[:,:,k] - field[:,:,k-2])
        k = field.shape[2] - 1
        rhs[:,:,k] = 0.0
    else:
        # npaire = 0
        k = field.shape[2] - 2
        rhs[:,:,k] = a * (field[:,:,k+1] - field[:,:,k-1]) \
            - b * (field[:,:,k] + field[:,:,k-2])
        k = field.shape[2] - 1
        rhs[:,:,k] = -2 * (a * field[:,:,k-1] + b * field[:,:,k-2])

    return rhs

def compute_rhs_2(mesh, field, axis):
    """ Compute the rhs for the derivative for Dirichlet BCs.

    :param mesh: The mesh on which derivatives are taken.
    :param field: The field for the variable who's derivative we want.
    :param axis: A number indicating direction in which to take derivative: 0=x; 1=y; 2=z.

    :type mesh: py4incompact3d.postprocess.mesh.Mesh
    :type axis: int

    :returns: rhs -- the right-hand side vector.
    :rtype: numpy.ndarray
    """

    # Setup
    rhs = np.zeros([field.shape[0], field.shape[1], field.shape[2]])

    if axis == 0:
        invdx = 1.0 / mesh.dx
    elif axis == 1:
        invdx = 1.0 / mesh.dy
    else:
        invdx = 1.0 / mesh.dz

    a = mesh.a * invdx / 2.0
    b = mesh.b * invdx / 4.0

    for i in range(field.shape[0]):
        for j in range(field.shape[1]):
            # BCs @ k = 0
            k = 0
            rhs[i,j,k] = -(5.0 * field[i,j,k] - 4.0 * field[i,j,k+1] - field[i,j,k+2]) \
                           * (0.5 * invdx)
            k = 1
            rhs[i,j,k] = 1.5 * (field[i,j,k+1] - field[i,j,k-1]) * (0.5 * invdx)

            # Internal nodes
            for k in range(2, field.shape[2] - 2):
                rhs[i,j,k] = a * (field[i,j,k+1] - field[i,j,k-1]) \
                               + b * (field[i,j,k+2] - field[i,j,k-2])

            # BCs @ k = n
            k = field.shape[2] - 2
            rhs[i,j,k] = 1.5 * (field[i,j,k+1] - field[i,j,k-1]) * (0.5 * invdx)
            k = field.shape[2] - 1
            rhs[i,j,k] = (5.0 * field[i,j,k] - 4.0 * field[i,j,k-1] - field[i,j,k-2]) \
                           * (0.5 * invdx)

    return rhs

def compute_rhs(postproc, arr, axis, direction, bc):
    """ Compute the rhs for the derivative.

    :param postproc: The basic postprocessing object.
    :param arr: An array containing the field's values.
    :param axis: A number indicating direction in which to take derivative: 0=x; 1=y; 2=z.
    :param direction: The field's orientation
    :param bc: The boundary condition: 0=periodic; 1=free-slip; 2=Dirichlet.

    :type mesh: py4incompact3d.postprocess.postproc.Postproc
    :type arr: numpy.ndarray
    :type axis: int
    :type direction: int
    :type bc: int

    :returns: rhs -- the right-hand side vector.
    :rtype: numpy.ndarray
    """

    mesh = postproc.mesh
    if bc == 0:
        return compute_rhs_0(mesh, arr, axis)
    elif bc == 1:
        return compute_rhs_1(mesh, arr, axis, direction)
    else:
        return compute_rhs_2(mesh, arr, axis)

def deriv(postproc, phi, axis, time):
    """ Take the derivative of field 'phi' along axis.

    :param postproc: The basic Postprocess object.
    :param phi: The name of the variable who's derivative we want.
    :param axis: A number indicating direction in which to take derivative: 0=x; 1=y; 2=z.
    :param time: The time stamp to compute derivatives for.

    :type postproc: py4incompact3d.postprocess.postprocess.Postprocess
    :type phi: str
    :type axis: int
    :type time: int

    :returns: dphidx -- the derivative
    :rtype: numpy.ndarray
    """

    # Ensure we have the derivative variables up to date
    postproc.mesh.compute_derivvars()

    # Transpose input pencil, data comes in in X-pencils
    if axis > 0:
        arrY = Field()
        arrY.new(postproc.mesh, pencil=1)
        arrY.data[0] = transpose(postproc.fields[phi].data[time], "xy", arrY.data[0])
        if axis == 2:
            arrZ = Field()
            arrZ.new(postproc.mesh, pencil=2)

            arr = transpose(arrY.data[0], "yz", arrZ.data[0])
        else:
            arr = arrY.data[0]
    else:
        arr = postproc.fields[phi].data[time]
    
    # Transpose the data to make loops more efficient
    arr = np.swapaxes(arr, axis, 2)

    # Get boundary conditions
    if axis == 0:
        bc = postproc.mesh.BCx
    elif axis == 1:
        bc = postproc.mesh.BCy
    else:
        bc = postproc.mesh.BCz

    # Compute RHS->derivative
    rhs = compute_rhs(postproc, arr, axis, [postproc.fields[phi].direction], bc)
    rhs = compute_deriv(rhs, bc, not bool(axis in [postproc.fields[phi].direction]))

    if (axis == 1) and postproc.mesh.stretched:
        rhs[:][:] *= postproc.mesh.ppy # XXX derivative is stored in last axis

    # Transpose output data
    rhs = np.swapaxes(rhs, 2, axis)

    # Transpose output pencil
    if axis > 0:
        if axis == 2:
            arrZ.data[0] = rhs
            arrY.data[0] = transpose(arrZ.data[0], "zy", arrY.data[0])
        else:
            arrY.data[0] = rhs

        arrX = Field()
        arrX.new(postproc.mesh, pencil=0)
        rhs = transpose(arrY.data[0], "yx", arrX.data[0])
        
    return rhs
