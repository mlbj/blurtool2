#ifndef BLUR_HPP
#define BLUR_HPP

#include <cmath>
#include "image.hpp" 

constexpr double PI = 3.14159265358979323846;
#define GAUSSIAN(i, j, center_row, center_col, sigma_row, sigma_col) \
    (1.0/ (2.0*PI*sigma_row*sigma_col)) * \
	(std::exp(-0.5 * (std::pow(((i) - (center_row)) / (sigma_row), 2) + \
    std::pow(((j) - (center_col)) / (sigma_col), 2))))

// Abstract blur definition
struct Blur{
public:
	Device device;

	Blur(Device device = CPU):
		device(device){}

	// forward method
	virtual void forward(Image& output, Image& input) = 0;

	// destructor 
	virtual ~Blur() = default;
};

// Abstract stationary blur definition
struct StationaryBlur : public Blur{
public: 
	Image* psf;
	int nrows, ncols;
	bool cache_psf;

	StationaryBlur(int nrows, 
				   int ncols, 
				   Device device, 
				   bool cache_psf = true):
		Blur(device),
		nrows(nrows),
		ncols(ncols),
		cache_psf(cache_psf) {}


	// destructor
	virtual ~StationaryBlur() = default;

};

// Concrete class for a Gaussian blur
struct GaussianBlur : public StationaryBlur{
public:
	float sigma_row, sigma_col;

	// constructor
	explicit GaussianBlur(float sigma_row,
						  float sigma_col,	 
						  int nrows, 
						  int ncols,
						  Device device = CPU,
						  bool cache_psf = true): 
		StationaryBlur(nrows, ncols, device, cache_psf),
		sigma_row(sigma_row),
		sigma_col(sigma_col){
		

		// compute the center of the PSF
		float center_row = 0.0;
		float center_col = 0.0;

		// allocate memory for the PSF, if needed
		if (cache_psf){
			// allocate the psf data
			psf = new Image(nrows, ncols, device);
			double sum = 0.0f;

			// CPU device
			if (device == CPU){
				// create psf
				for (int i=0; i<nrows; i++){
					float x = -1.0 + 2.0*i/(nrows-1);
					for (int j=0; j<ncols; j++){
						float y = -1.0 + 2.0*j/(ncols-1);
						(*psf)(i, j) = GAUSSIAN(x, y, 
											    center_row, center_col, 
											    sigma_row, sigma_col); // exp(-0.5*(pow((i-center_row)/sigma_row, 2) + pow((j-center_col)/sigma_col, 2)));

						// accumulate 
						sum += (*psf)(i, j);
					}
				}

				// normalize area
				for (int i=0; i<nrows; i++){
					for (int j=0; j<ncols; j++){
						(*psf)(i,j) /= sum;
					}
				}


			// GPU device
			}else if(device == GPU){
				// non implemented 
				// 
			}else{
				throw std::runtime_error("Invalid device.");
			}
		}


	}

	// forward method
	void forward(Image& output, Image& input) override{
		// check if the input image is on the same device as the PSF
		if (input.device != device){
			throw std::runtime_error("Input image and PSF are not on the same device.");
		}

		// check if the output image is on the same device as the input image
		if (output.device != input.device){
			throw std::runtime_error("Output image and input image are not on the same device.");
		}

		// check shapes

		int center_row = nrows/2;
		int center_col = ncols/2;

		// cached psf case
		if (cache_psf){
			// convolve the input image with the PSF
			for (int i=0; i<input.nrows; i++){
				for (int j=0; j<input.ncols; j++){
					float sum = 0.0f;
					for (int k=0; k<nrows; k++){
						for (int l=0; l<ncols; l++){
							int x = i + k - center_row;
							int y = j + l - center_col;
							if (x >= 0 && x < input.nrows && y >= 0 && y < input.ncols){
								sum += input(x, y)*(*psf)(k, l);
							}
						}
					}
					output(i, j) = sum;
				}
			}

		
		// non-cached psf case
		}else{
			// compute the PSF on the fly
			// Image psf(nrows, ncols, device);
			// for (int i=0; i<nrows; i++){
			// 	for (j=0; j<ncols; j++){
			// 		psf(i, j) = exp(-0.5*(pow((i-center_row)/sigma_row, 2) + pow((j-center_col)/sigma_col, 2)));
			// 	}
			// }


			// deallocate psf
			// delete psf;
		
		}
	}
};

#endif
