# conda create --name torch python=3.8
# conda activate torch
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
# mpirun -np 2 python3 main.py

from mpi4py import MPI
from numba import cuda
import numpy as np

@cuda.jit()
def kernel(array_on_gpu):
    array_on_gpu[0] = 0.5 # FAST!

def rank0():
    input_array = np.zeros((100,), dtype=np.float64)
    gpu_input_array = cuda.to_device(input_array)
    MPI.COMM_WORLD.send(gpu_input_array.get_ipc_handle(), dest=1)

def rank1():
    handle = MPI.COMM_WORLD.recv(source=0)
    received_gpu_input_array = handle.open() # FAST
    # received_gpu_input_array.copy_to_host() # SLOW
    kernel[32, 32](received_gpu_input_array)
    # handle.close() # SLOW
    print("Success!")

def main():
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        rank0()
    else:
        rank1()

if __name__ == "__main__":
    main()