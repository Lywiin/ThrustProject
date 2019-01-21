#pragma once

#include "ppm.hpp"

// Conversion from RGB (inRGB) to HSV (outH, outS, outV)
// Launch with 2D grid
__global__
void rgb2hsv( const uchar3 *const inRGB, float3 *outHSV, const int width, const int height );

// Conversion from HSV (inH, inS, inV) to RGB (outRGB)
// Launch with 2D grid
__global__
void hsv2rgb( const float3 *inHSV, uchar3 *const outRGB, const int width, const int height );

float student2(const PPMBitmap& in, PPMBitmap& out, const int size);

// You may export your own class or functions to communicate data between the exercises ...
