#include "image.hpp" 
#include "blur.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <cufft.h>

constexpr double PI = 3.14159265358979323846;
#define GAUSSIAN(i, j, center_row, center_col, sigma_row, sigma_col) \
     (1.0/ (2.0*PI*sigma_row*sigma_col)) * \
     (std::exp(-0.5 * (std::pow(((i) - (center_row)) / (sigma_row), 2) + \
     std::pow(((j) - (center_col)) / (sigma_col), 2))))


// cuda kernel to generate the data of stationary Gaussian psf on GPU
__global__ void generateGaussian(cufftComplex* d_psf_data, 
								 int nrows,
								 int ncols,
								 float center_row,
								 float center_col,
								 float sigma_row, 
								 float sigma_col,
								 float* d_sum){
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i<nrows && j<ncols){
		float x = -1.0f + 2.0f*i/(nrows-1);
		float y = -1.0f + 2.0f*j/(ncols-1);
		
		cufftComplex value;
		value.x = GAUSSIAN(x, y, center_row, center_col, sigma_row, sigma_col);
		value.y = 0.0f;

		d_psf_data[i*ncols + j] = value;
		
		atomicAdd(d_sum, powf(value.x, 2.0) + powf(value.y, 2.0));
	}	
}

// cuda kernel to normalize a 2d mask 
__global__ void normalize(cufftComplex* d_data,
						  int nrows,
						  int ncols,
						  float d_sum){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i<nrows && j<ncols && d_sum>0) {
    	// Normalize to [0,1]	
        d_data[i*ncols + j].x /= d_sum;
		d_data[i*ncols + j].y /= d_sum; 
    }
}


// GaussianBlur constructor 
GaussianBlur::GaussianBlur(float sigma_row,
				 		   float sigma_col,	 
					  	   int nrows, 
					  	   int ncols,
					  	   bool cache_psf): 
	StationaryBlur(nrows, ncols, cache_psf),
	sigma_row(sigma_row),
	sigma_col(sigma_col){

	// allocate memory for the PSF, if needed
	if (cache_psf){
		// allocate the psf data
		psf = new Image(nrows, ncols, GPU);
		
		// allocate d_sum and set it to zero. we'll use this to 
		// store the normalization value 
		float* d_sum; 
		cudaMalloc(&d_sum, sizeof(float));
		cudaMemset(d_sum, 0, sizeof(float));

		// fill the gaussian psf data
		generateGaussian<<<gridSize, blockSize>>>(psf->d_data, 
			         							  nrows, 
												  ncols, 
                                                  center_row, 
                                                  center_col, 
                                                  sigma_row, 
                                                  sigma_col, 
                                                  d_sum);
		cudaDeviceSynchronize();
		
		// normalize
		normalize<<<gridSize, blockSize>>>(psf->d_data,
										   nrows,
										   ncols,
										   *d_sum);
	   cudaDeviceSynchronize();
	   
	   // free d_sum
	   cudaFree(d_sum);

	}
}

// forward method
void GaussianBlur::forward(Image& output, Image& input){
	if (input.nrows != nrows && input.ncols != ncols){
		throw std::runtime_error("input format does not match the GaussianBlur dimensions.");
	}

	// create temporary buffer for fft(psf) 
	cufftComplex* d_data_psf_swap;
	cudaMalloc(&d_data_psf_swap, nrows*ncols*sizeof(cufftComplex));

	if (cache_psf){
		// if psf is cached, is only a matter of copying it to the temporary buffer 
		cudaMemcpy(d_data_psf_swap, psf->d_data, nrows*ncols*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
	}else{
		// otherwise, if psf is not cached, create it on demand 

		// allocate d_sum and set it to zero. we'll use this to 
		// store the normalization value 
		float* d_sum; 
		cudaMalloc(&d_sum, sizeof(float));
		cudaMemset(d_sum, 0, sizeof(float));

		// fill the gaussian psf data
		generateGaussian<<<gridSize, blockSize>>>(d_data_psf_swap, 
			         							  nrows, 
												  ncols, 
                                                  center_row, 
                                                  center_col, 
                                                  sigma_row, 
                                                  sigma_col, 
                                                  d_sum);
		cudaDeviceSynchronize();
		
		// normalize
		normalize<<<gridSize, blockSize>>>(d_data_psf_swap,
										   nrows,
										   ncols,
										   *d_sum);
		cudaDeviceSynchronize();
	   
	   // free d_sum
	   cudaFree(d_sum);
	}

	// compute the fft of the psf and store on the temporary buffer
	if (cufftExecC2C(plan, d_data_psf_swap, d_data_psf_swap, CUFFT_FORWARD) != CUFFT_SUCCESS){
		throw std::runtime_error("CUFFT forward transform failed.");
	}

	// compute the fft of the input data and store on the output data 
	if (cufftExecC2C(plan, input.d_data, output.d_data, CUFFT_FORWARD) != CUFFT_SUCCESS){
		throw std::runtime_error("CUFFT forward transform failed.");
	}

	// multiply 
	multiply<<<gridSize, blockSize>>>(output.d_data, 
									  output.d_data,
									  d_data_psf_swap,
									  nrows,
									  ncols);

	// compute ifft of the product 
	if (cufftExecC2C(plan, output.d_data, output.d_data, CUFFT_INVERSE) != CUFFT_SUCCESS){
		throw std::runtime_error("CUFFT inverse transform failed.");
	}
}




// // stationary blur 
// StationaryBlur::StationaryBlur(Image& kernel): kernel(kernel){}

// void StationaryBlur::forward(Image& image){
//     for (int x=0; x<image.nrows; x++){
//         for (int y=0; y<image.ncols; y++){
//             image(x,y) *= x-y;
//         }
//     }
// }
