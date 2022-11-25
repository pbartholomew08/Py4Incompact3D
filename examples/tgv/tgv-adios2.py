"""
       FILE: tgv-adios2.py
     AUTHOR: Paul Bartholomew
DESCRIPTION: Tests reading and processing TGV with ADIOS2.
"""

import math

from mpi4py import MPI

import numpy as np

from py4incompact3d import comm, rank
from py4incompact3d.postprocess.postprocess import Postprocess
from py4incompact3d.deriv.deriv import deriv
import decomp2d

HDR="=" * 72
LBR="-" * 72

NX=65
NY=NX
NZ=NX

LX=math.pi
LY=LX
LZ=LX

BCX=1
BCY=BCX
BCZ=BCX

ADIOS2_FILE="/home/paul/src/Xcompact3d/Incompact3d/examples/Taylor-Green-Vortex/data"

def calc_enst(dudy, dudz, dvdx, dvdz, dwdx, dwdy):

    enst = (dwdy - dvdz)**2 + (dudz - dwdx)**2 + (dvdx - dudy)**2
    return enst / 2.0

def calc_ke(u, v, w):
    ke = u**2 + v**2 + w**2
    return ke / 2.0

def integrate_field(phi):
    return comm.allreduce(phi.sum(), op=MPI.SUM)
    
def main():

    print(HDR)
    print("Post-processing TGV.")
    print(LBR)

    io_name = "solution-io"

    postprocess = Postprocess(n=[NX, NY, NZ],
                              l=[LX, LY, LZ],
                              bc=[BCX, BCY, BCZ])

    postprocess.init_io(io_name)

    ADIOS2_FILE = "data.sst"
    postprocess.add_field("ux", ADIOS2_FILE, direction=0)
    postprocess.add_field("uy", ADIOS2_FILE, direction=1)
    postprocess.add_field("uz", ADIOS2_FILE, direction=2)

    io_name = "solution-io"

    postprocess.open_io(io_name, "data")

    ke_hist = []
    enst_hist = []
    
    mesh = postprocess.mesh
    xlast = mesh.NxLocal[0]
    ylast = mesh.NyLocal[0]
    zlast = mesh.NzLocal[0]
    if (mesh.NxStart[0] + mesh.NxLocal[0]) == NX:
        xlast -=1
    if (mesh.NyStart[0] + mesh.NyLocal[0]) == NY:
        ylast -=1
    if (mesh.NzStart[0] + mesh.NzLocal[0]) == NZ:
        zlast -=1

    for t in range(1, 100+1):

        print(f"Timestep: {t}")
        postprocess.load(time=t)

        ux = postprocess.fields["ux"].data[t]
        uy = postprocess.fields["uy"].data[t]
        uz = postprocess.fields["uz"].data[t]

        # Update kinetic energy
        ke = integrate_field(calc_ke(ux[:xlast,:ylast,:zlast],
                                     uy[:xlast,:ylast,:zlast],
                                     uz[:xlast,:ylast,:zlast]))
        ke = ke / (NX - 1) / (NY - 1) / (NZ - 1) # Average
        ke_hist.append(ke)

        # Update enstrophy
        dudy = deriv(postprocess, "ux", 1, t)
        dudz = deriv(postprocess, "ux", 2, t)
        dvdx = deriv(postprocess, "uy", 0, t)
        dvdz = deriv(postprocess, "uy", 2, t)
        dwdx = deriv(postprocess, "uz", 0, t)
        dwdy = deriv(postprocess, "uz", 1, t)
        enst = integrate_field(calc_enst(dudy[:xlast,:ylast,:zlast], dudz[:xlast,:ylast,:zlast],
                                         dvdx[:xlast,:ylast,:zlast], dvdz[:xlast,:ylast,:zlast],
                                         dwdx[:xlast,:ylast,:zlast], dwdy[:xlast,:ylast,:zlast]))
        enst = enst / (NX - 1) / (NY - 1) / (NZ - 1)
        enst_hist.append(enst)

        # Cleanup data
        postprocess.clear_data()

    postprocess.close_io(io_name, "data")

    if rank == 0:
        with open("ke.dat", "w") as output:
            t = 1
            for i in range(len(ke_hist)):
                if (t%10 == 0):
                    output.write(f"{ke_hist[i]},{enst_hist[i]}\n")
                t += 1

if __name__ == "__main__":
    main()
