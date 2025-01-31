#include "blur.h"
#include "image.h"
#include <iostream>
#include <algorithm>

// stationary blur 
StationaryBlur::StationaryBlur(Image& kernel): kernel(kernel){}

void StationaryBlur::forward(Image& image){
    for (int x=0; x<image.nrows; x++){
        for (int y=0; y<image.ncols; y++){
            image(x,y) *= x-y;
        }
    }
}