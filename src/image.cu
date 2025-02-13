#include "image.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>

__device__ cufftComplex Image::gpu_at(int x, int y){
	if (device==CPU){
        // throw cuda runtime error throw std::runtime_error("Calling __device__ function to access host data. Check device attribute.");
    }
                
    return d_data[x*ncols+y];

}     
 
__device__ void Image::set_gpu_value(int x, int y, cufftComplex value){
    if (device==CPU){
        // throw cuda run time error throw std::runtime_error("Calling __device__ function to modify host data. Check device attribute.");
    }
        
	d_data[x*ncols+y] = value;
}


__global__ void multiply(cufftComplex* d_output, cufftComplex* d_input1, cufftComplex* d_input2, int nrows, int ncols){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i<nrows && j<ncols) {
        d_output[i*ncols + j].x = d_input1[i*ncols + j].x*d_input2[i*ncols + j].x - d_input1[i*ncols + j].y*d_input2[i*ncols + j].y;
        d_output[i*ncols + j].y = d_input1[i*ncols + j].x*d_input2[i*ncols + j].y + d_input1[i*ncols + j].y*d_input2[i*ncols + j].x;
    }
}

