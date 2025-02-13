#ifndef BLUR_HPP
#define BLUR_HPP

#include <cmath>
#include "image.hpp" 
#include <cuda_runtime.h>
#include <cufft.h>

// Abstract blur definition
class Blur{
public:
	int nrows, ncols;
	dim3 blockSize;
	dim3 gridSize;
	cufftHandle plan;

 	Blur(int nrows, int ncols):
 		nrows(nrows),
 		ncols(ncols),
 		blockSize(16,16){

		// set gridsize
    	dim3 gridSize((nrows + blockSize.x - 1) / blockSize.x, 
       	          (ncols + blockSize.y - 1) / blockSize.y);

		// create cufft c2c Plan
		if (cufftPlan2d(&plan, nrows, ncols, CUFFT_C2C) == CUFFT_SUCCESS){
			std::cout << "blurtool2: Successfully created CUFFT_C2C plan." << std::endl;
		} 			
	}

 	// forward method
 	virtual void forward(Image& output, Image& input) = 0;

 	// destructor 
 	virtual ~Blur(){
		cufftDestroy(plan);
	}
 };

// Abstract stationary blur definition
class StationaryBlur : public Blur{
public: 
	Image* psf;
 	bool cache_psf;
 	 
 	StationaryBlur(int nrows, 
 				   int ncols,  
 				   bool cache_psf = true):
 		Blur(nrows, ncols),	
 		cache_psf(cache_psf) {}


 	// destructor
 	virtual ~StationaryBlur() = default;

};

// Concrete class for a Gaussian blur
class GaussianBlur : public StationaryBlur{
public:
	float sigma_row, sigma_col;
	float center_row = 0.0f;
	float center_col = 0.0f;

	// constructor
 	explicit GaussianBlur(float sigma_row,
 						  float sigma_col,	 
 						  int nrows, 
 						  int ncols,
 						  bool cache_psf = true);
		

	// forward method
	void forward(Image& output, Image& input) override;


// 	// forward method
// 	void forward(Image& output, Image& input) override{
// 		// check if the input image is on the same device as the PSF
// 		if (input.device != device){
// 			throw std::runtime_error("Input image and PSF are not on the same device.");
// 		}

// 		// check if the output image is on the same device as the input image
// 		if (output.device != input.device){
// 			throw std::runtime_error("Output image and input image are not on the same device.");
// 		}

// 		// check shapes

// 		int center_row = nrows/2;
// 		int center_col = ncols/2;

// 		// cached psf case
// 		if (cache_psf){
// 			// convolve the input image with the PSF
// 			for (int i=0; i<input.nrows; i++){
// 				for (int j=0; j<input.ncols; j++){
// 					float sum = 0.0f;
// 					for (int k=0; k<nrows; k++){
// 						for (int l=0; l<ncols; l++){
// 							int x = i + k - center_row;
// 							int y = j + l - center_col;
// 							if (x >= 0 && x < input.nrows && y >= 0 && y < input.ncols){
// 								sum += input(x, y)*(*psf)(k, l);
// 							}
// 						}
// 					}
// 					output(i, j) = sum;
// 				}
// 			}

		
// 		// non-cached psf case
// 		}else{
// 			// compute the PSF on the fly
// 			// Image psf(nrows, ncols, device);
// 			// for (int i=0; i<nrows; i++){
// 			// 	for (j=0; j<ncols; j++){
// 			// 		psf(i, j) = exp(-0.5*(pow((i-center_row)/sigma_row, 2) + pow((j-center_col)/sigma_col, 2)));
// 			// 	}
// 			// }


// 			// deallocate psf
// 			// delete psf;
		
// 		}
// 	}
};

#endif
