import numpy as np

from mpi4py import MPI

from py4incompact3d import comm, rank
from py4incompact3d.postprocess.postprocess import Postprocess

NX=9
NY=NX
NZ=NX

LX=1.0
LY=LX
LZ=LX

BCX=1
BCY=BCX
BCZ=BCX

ADIOS2_FILE="/home/paul/src/Xcompact3d/Incompact3d/examples/User/adios2_config.xml"

def main():

    io_name = "solution-io"

    postprocess = Postprocess(n=[NX, NY, NZ],
                              l=[LX, LY, LZ],
                              bc=[BCX, BCY, BCZ])

    postprocess.init_io(io_name)

    ADIOS2_FILE = "data.sst"
    postprocess.add_field("ux", ADIOS2_FILE, direction=0)

    postprocess.open_io(io_name, "data")
    mesh = postprocess.mesh

    for t in range(1, 2):

        postprocess.load(time=t)

        ux = postprocess.fields["ux"].data[t]

        of = "of"+str(rank)
        print(f"{rank}: {mesh.NxLocal[0]},{mesh.NyLocal[0]},{mesh.NzLocal[0]} {ux.shape}")
        with open(of, "w") as outfile:
            for i in range(mesh.NxLocal[0]):
                for j in range(mesh.NyLocal[0]):
                    for k in range(mesh.NzLocal[0]):
                        idx = mesh.NxStart[0] + mesh.NyStart[0] * NX + mesh.NzStart[0] * NX * NY
                        idx += i + j * NX + k * NX * NY
                        outfile.write(f"{i},{j},{k} {idx}: {ux[i,j,k]}\n")
        
        for i in range(mesh.NxLocal[0]):
            for j in range(mesh.NyLocal[0]):
                for k in range(mesh.NzLocal[0]):

                    idx = mesh.NxStart[0] + mesh.NyStart[0] * NX + mesh.NzStart[0] * NX * NY
                    idx += i + j * NX + k * NX * NY

                    if ux[i,j,k] != idx:
                        print(f"Error @ {i},{j},{k}: expected {idx}, got {ux[i,j,k]}")
                        exit(1)

    postprocess.close_io(io_name, "data")

if __name__ == "__main__":
    main()
