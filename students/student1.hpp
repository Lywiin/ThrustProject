#pragma once

#include "ppm.hpp"

// Conversion from RGB (inRGB) to HSV (outH, outS, outV)
// Launch with 2D grid
__global__
void rgb2hsv(	const uchar4 *const inRGB, const int width, const int height, float *const outH, float *const outS, float *const outV );

// Conversion from HSV (inH, inS, inV) to RGB (outRGB)
// Launch with 2D grid
__global__
void hsv2rgb(	const float *const inH, const float *const inS, const float *const inV, const int width, const int height, uchar4 *const outRGB );

float student1(const PPMBitmap& in, PPMBitmap& out, const int size);

// You may export your own class or functions to communicate data between the exercises ...
