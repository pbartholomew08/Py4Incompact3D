import math

import numpy as np

from py4incompact3d import comm, rank
from py4incompact3d.postprocess.postprocess import Postprocess
from py4incompact3d.deriv.deriv import deriv

#import matplotlib.pyplot as plt

NX=65
NY=65
NZ=65
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
                #postproc.fields["ux"].data[0][i,j,k] = math.sin(x) * math.cos(z)
                #postproc.fields["ux"].data[0][i,j,k] = math.sin(x) * math.cos(y)
                #postproc.fields["ux"].data[0][i,j,k] = math.sin(x)
                
                postproc.fields["uy"].data[0][i,j,k] = math.cos(x) * math.sin(y) * math.cos(z)
                postproc.fields["uz"].data[0][i,j,k] = math.cos(x) * math.cos(y) * math.sin(z)
                postproc.fields["phi"].data[0][i,j,k] = math.cos(x) * math.cos(y) * math.cos(z)

def compute_grad(postproc, fieldname):

    #print(f"{fieldname}: X")
    dfdx = deriv(postproc, fieldname, 0, 0)
    #print(f"{fieldname}: Y")
    dfdy = deriv(postproc, fieldname, 1, 0)
    #print(f"{fieldname}: Z")
    dfdz = deriv(postproc, fieldname, 2, 0)

    return dfdx, dfdy, dfdz

def write_solution(postproc, dfx, dfy, dfz, field, fnx, fny, fnz):

    ext = "-" + str(rank) + ".txt"

    f = []
    with open(field + "dx" + ext, "w") as output:
        for i in range(postproc.mesh.NxLocal[0]):
            j = 0 #(NY // 2) + 1
            k = 0 #(NZ // 2) + 1

            i0 = postproc.mesh.NxStart[0]
            j0 = postproc.mesh.NyStart[0]
            k0 = postproc.mesh.NzStart[0]

            x = (i0 + i) * postproc.mesh.dx
            y = (j0 + j) * postproc.mesh.dy
            z = (k0 + k) * postproc.mesh.dz
            f.append(fnx(x,y,z))
            output.write(f"{dfx[i,j,k]} {f[-1]}\n")
    plt.plot(dfx[:,j,k])
    plt.plot(f, ls="", marker="o")
    figname = "dudx-" + str(rank) + ".png"
    plt.savefig(figname)
    plt.close()

    f = []
    with open(field + "dy" + ext, "w") as output:
        i = (NX // 2) + 1
        for j in range(postproc.mesh.NyLocal[0]):
            k = 0 #(NZ // 2) + 1

            i0 = postproc.mesh.NxStart[1]
            j0 = postproc.mesh.NyStart[1]
            k0 = postproc.mesh.NzStart[1]

            x = (i0 + i) * postproc.mesh.dx
            y = (j0 + j) * postproc.mesh.dy
            z = (k0 + k) * postproc.mesh.dz
            f.append(fny(x,y,z))
            output.write(f"{dfy[i,j,k]} {f[-1]}\n")
    plt.plot(dfy[i,:,k])
    plt.plot(f, ls="", marker="o")
    figname = "dudy-" + str(rank) + ".png"
    plt.savefig(figname)
    plt.close()

    f= []
    with open(field + "dz" + ext, "w") as output:
        i = (NX // 2) + 1
        j = 0 #(NY // 2) + 1
        for k in range(postproc.mesh.NzLocal[0]):

            i0 = postproc.mesh.NxStart[2]
            j0 = postproc.mesh.NyStart[2]
            k0 = postproc.mesh.NzStart[2]

            x = (i0 + i) * postproc.mesh.dx
            y = (i0 + j) * postproc.mesh.dy
            z = (i0 + k) * postproc.mesh.dz
            f.append(fnz(x,y,z))
            output.write(f"{dfz[i,j,k]} {f[-1]}\n")
    plt.plot(dfz[i,j,:])
    plt.plot(f, ls="", marker="o")
    figname = "dudz-" + str(rank) + ".png"
    plt.savefig(figname)

def main():

    postproc = Postprocess(n=[NX, NY, NZ],
                           l=[L, L, L],
                           bc=[BC, BC, BC])
    init(postproc)

    #print(postproc.fields["ux"].data[0].flags)
    for t in range(100):
        dudx, dudy, dudz = compute_grad(postproc, "ux")
        dvdx, dvdy, dvdz = compute_grad(postproc, "uy")
        dwdx, dwdy, dwdz = compute_grad(postproc, "uz")
    #dphidx, dphidy, dphidz = compute_grad(postproc, "phi")

    #def ddx(x, y, z):
    #    #return math.cos(x) * math.cos(y) * math.cos(z)
    #    return math.cos(x)
    #def ddy(x, y, z):
    #    #return -math.sin(x) * math.sin(y) * math.cos(z)
    #    return -math.sin(y)
    #def ddz(x, y, z):
    #    #return -math.sin(x) * math.cos(y) * math.sin(z)
    #    return -math.sin(z)
    #write_solution(postproc, dudx, dudy, dudz, "u", ddx, ddy, ddz)

if __name__ == "__main__":
    main()
