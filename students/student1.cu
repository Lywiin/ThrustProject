#include "student1.hpp"

#include <iostream>

// converts a RGB color to a HSV one ...
__device__
float3 RGB2HSV( const uchar4 inRGB ) {
	const float R = (float)( inRGB.x ) / 256.f;
	const float G = (float)( inRGB.y ) / 256.f;
	const float B = (float)( inRGB.z ) / 256.f;

	const float min		= fminf( R, fminf( G, B ) );
	const float max		= fmaxf( R, fmaxf( G, B ) );
	const float delta	= max - min;

	// H
	float H;
	if( delta < 0 )
		H = 0.f;
	else if	( max == R )
		H = 60.f * ( G - B ) / delta + 360.f;
	else if ( max == G )
		H = 60.f * ( B - R ) / delta + 120.f;
	else
		H = 60.f * ( R - G ) / delta + 240.f;
	while	( H >= 360.f )
		H -= 360.f ;

	// S
	float S = max < 0 ? 0.f : 1.f - min / max;

	// V
	float V = max;

	return make_float3(H, S, V);
}


// converts a HSV color to a RGB one ...
__device__
uchar4 HSV2RGB( const float H, const float S, const float V )
{
	const float d = H / 60.f;
	const int hi = (int)d % 6;
	const float f = d - (float)hi;

	const float l = V * ( 1.f - S );
	const float m = V * ( 1.f - f * S );
	const float n = V * ( 1.f - ( 1.f - f ) * S );

	float R, G, B;

	if	( hi == 0 )
	{ R = V; G = n;	B = l; }
	else if ( hi == 1 )
	{ R = m; G = V;	B = l; }
	else if ( hi == 2 )
	{ R = l; G = V;	B = n; }
	else if ( hi == 3 )
	{ R = l; G = m;	B = V; }
	else if ( hi == 4 )
	{ R = n; G = l;	B = V; }
	else
	{ R = V; G = l;	B = m; }

	return make_uchar4( R*256.f, G*256.f, B*256.f, 255 );
}

// Conversion from RGB (inRGB) to HSV (outH, outS, outV)
// Launched with 2D grid
__global__
void rgb2hsv(	const uchar4 *const inRGB, const int width, const int height, float *const outH, float *const outS, float *const outV ) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx > width) return;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy > height) return;
	int tid = tidx + tidy * width;

	float3 resultHSV = RGB2HSV(inRGB[tid]);
	outH[tid] = resultHSV.x;
	outS[tid] = resultHSV.y;
	outV[tid] = resultHSV.z;
}

// Conversion from HSV (inH, inS, inV) to RGB (outRGB)
// Launched with 2D grid
__global__
void hsv2rgb(	const float *const inH, const float *const inS, const float *const inV, const int width, const int height, uchar4 *const outRGB ) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx > width) return;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy > height) return;
	int tid = tidx + tidy * width;

	uchar4 outRGBtid = HSV2RGB(inH[tid], inS[tid], inV[tid]);
	outRGB[tid] = outRGBtid;
}

float student1(const PPMBitmap &in, PPMBitmap &out, const int size) {
	return 0.f;
}

