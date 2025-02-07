#include "image.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

__device__ float Image::gpu_at(int x, int y){
	if (device==CPU){
        	throw std::runtime_error("Calling __device__ function to access host data. Check device attribute.");
        }else{
                return d_data[x*ncols+y];
        }
}     
 
__device__ void Image::set_gpu_value(int x, int y, value){
        if (device==CPU){
                throw std::runtime_error("Calling __device__ function to modify host data. Check device attribute.");
        }else{
                d_data[x*ncols+y] = value;
        }
}



