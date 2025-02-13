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



