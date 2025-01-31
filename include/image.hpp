#ifndef IMAGE_H
#define IMAGE_H

#include <vector>


// this is our main Image definition 
struct Image{
	int ncols, nrows;
	std::vector<std::vector<float>> data;

	// zero constructor 
	Image(int h, int w): nrows(h),
						 ncols(w),
			     		 data(h, std::vector<float>(w, 0.0f)){}

	// accessor
	float& operator()(int x, int y){
		return data[x][y];
	}
	const float& operator()(int x, int y) const{
		return data[x][y];
	}
};

#endif
