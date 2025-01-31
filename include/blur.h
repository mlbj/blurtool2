#ifndef BLUR_H
#define BLUR_H

#include "image.h"

// this is our main Blur definition
struct Blur{
	virtual void forward(Image& image) = 0;
	virtual ~Blur() = default;
};

// Stationary blur definition
struct StationaryBlur : public Blur{
	Image kernel;
	StationaryBlur(Image& kernel);
	void forward(Image& image) override;
};

#endif
