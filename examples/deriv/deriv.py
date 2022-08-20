import math

import numpy as np

from py4incompact3d import comm, rank
from py4incompact3d.postprocess.postprocess import Postprocess
from py4incompact3d.deriv.deriv import deriv

import matplotlib.pyplot as plt

N=33
L=math.pi
BC=1

def init(postproc):

    postproc.add_field("ux", "f", direction=0)
    postproc.add_field("uy", "f", direction=1)
    postproc.add_field("uz", "f", direction=2)
    postproc.add_field("phi", "f", direction=-1)

    print(postproc.fields["ux"].direction)

    postproc.fields["ux"].new(postproc.mesh)
    postproc.fields["uy"].new(postproc.mesh)
    postproc.fields["uz"].new(postproc.mesh)
    postproc.fields["phi"].new(postproc.mesh)

    print(postproc.fields["ux"].direction)

    postproc.mesh.compute_derivvars()

    ni = postproc.mesh.NxLocal[0]
    nj = postproc.mesh.NyLocal[0]
    nk = postproc.mesh.NzLocal[0]
    i0 = postproc.mesh.NxStart[0]
    j0 = postproc.mesh.NyStart[0]
    k0 = postproc.mesh.NzStart[0]

    dx = postproc.mesh.dx
    dy = postproc.mesh.dy
    dz = postproc.mesh.dz
    
    for i in range(ni):
        x = (i0 + i) * dx
        for j in range(nj):
            y = (j0 + j) * dy
            for k in range(nk):
                z = (k0 + k) * dz

                postproc.fields["ux"].data[0][i,j,k] = math.sin(x) * math.cos(y) * math.cos(z)
                postproc.fields["uy"].data[0][i,j,k] = math.cos(x) * math.sin(y) * math.cos(z)
                postproc.fields["uz"].data[0][i,j,k] = math.cos(x) * math.cos(y) * math.sin(z)
                postproc.fields["phi"].data[0][i,j,k] = math.cos(x) * math.cos(y) * math.cos(z)

def compute_grad(postproc, fieldname):

    dfdx = deriv(postproc, fieldname, 0, 0)
    dfdy = deriv(postproc, fieldname, 1, 0)
    dfdz = deriv(postproc, fieldname, 2, 0)

    return dfdx, dfdy, dfdz

def write_solution(postproc, dfx, dfy, dfz, field, fnx, fny, fnz):

    ext = ".txt"

    f = []
    with open(field + "dx" + ext, "w") as output:
        for i in range(postproc.mesh.NxLocal[0]):
            j = 0 #(N // 2) + 1
            k = 0 #(N // 2) + 1

            x = i * postproc.mesh.dx
            y = j * postproc.mesh.dy
            z = k * postproc.mesh.dz
            f.append(fnx(x,y,z))
            output.write(f"{dfx[i,j,k]} {f[-1]}\n")
    plt.plot(dfx[:,j,k])
    plt.plot(f, ls="", marker="o")
    plt.savefig("dudx.png")
    plt.close()

    f = []
    with open(field + "dy" + ext, "w") as output:
        i = (N // 2) + 1
        for j in range(postproc.mesh.NyLocal[0]):
            k = 0 #(N // 2) + 1

            x = i * postproc.mesh.dx
            y = j * postproc.mesh.dy
            z = k * postproc.mesh.dz
            f.append(fny(x,y,z))
            output.write(f"{dfy[i,j,k]} {f[-1]}\n")
    plt.plot(dfy[i,:,k])
    plt.plot(f, ls="", marker="o")
    plt.savefig("dudy.png")
    plt.close()

    f= []
    with open(field + "dz" + ext, "w") as output:
        i = (N // 2) + 1
        j = 0 #(N // 2) + 1
        for k in range(postproc.mesh.NzLocal[0]):

            x = i * postproc.mesh.dx
            y = j * postproc.mesh.dy
            z = k * postproc.mesh.dz
            f.append(fnz(x,y,z))
            output.write(f"{dfz[i,j,k]} {f[-1]}\n")
    plt.plot(dfz[i,j,:])
    plt.plot(f, ls="", marker="o")
    plt.savefig("dudz.png")

def main():

    postproc = Postprocess(n=[N, N, N],
                           l=[L, L, L],
                           bc=[BC, BC, BC])
    init(postproc)

    dudx, dudy, dudz = compute_grad(postproc, "ux")
    dvdx, dvdy, dvdz = compute_grad(postproc, "uy")
    dwdx, dwdy, dwdz = compute_grad(postproc, "uz")
    dphidx, dphidy, dphidz = compute_grad(postproc, "phi")

    def ddx(x, y, z):
        return math.cos(x) * math.cos(y) * math.cos(z)
    def ddy(x, y, z):
        return -math.sin(x) * math.sin(y) * math.cos(z)
    def ddz(x, y, z):
        return -math.sin(x) * math.cos(y) * math.sin(z)
    write_solution(postproc, dudx, dudy, dudz, "u", ddx, ddy, ddz)

if __name__ == "__main__":
    main()
