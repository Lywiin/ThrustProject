#include "student1.hpp"
#include "student2.hpp"
#include "../utils/common.hpp"
//#include "../utils/chronoGPU.hpp"
#include "../utils/utils.cuh"

#include <iostream>
#include <math.h>
/*
// converts a RGB color to a HSV one ...
__device__
float3 RGB2HSV( const uchar3 inRGB ) {
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
uchar3 HSV2RGB( const float H, const float S, const float V )
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

	return make_uchar3( R*256.f, G*256.f, B*256.f );
}

// Conversion from RGB (inRGB) to HSV (outH, outS, outV)
// Launched with 2D grid
__global__
void rgb2hsv( const uchar3 *const inRGB, float3 *outHSV, const int width, const int height ) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx > width) return;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy > height) return;
	int tid = tidx + tidy * width;

	float3 resultHSV = RGB2HSV(inRGB[tid]);
	outHSV[tid] = resultHSV;
}

// Conversion from HSV (inH, inS, inV) to RGB (outRGB)
// Launched with 2D grid
__global__
void hsv2rgb( const float3 *inHSV, uchar3 *const outRGB, const int width, const int height ) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx > width) return;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy > height) return;
	int tid = tidx + tidy * width;

	uchar3 outRGBtid = HSV2RGB(inHSV[tid].x, inHSV[tid].y, inHSV[tid].z);
	outRGB[tid] = outRGBtid;
}
*/
/*
// Sort an array by ascending order using bubble sort method
__device__
void sort( float3* inTab, int tabSize )
{
	for (int i = 0; i < tabSize / 2; i++)
	{
		int min = i;
		for (int l = i + 1; l < tabSize; ++l)
			if (inTab[l].z < inTab[min].z)
				min = l;
		float3 temp = inTab[i];
		inTab[i] = inTab[min];
		inTab[min] = temp;
	}
}

// Apply median filter on HSV image
// Launched with 2D grid
__global__
void medianFilter( const float3 *inHSV, float3 *outHSV, const int width, const int height, const int windowSize ) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx >= width) return;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy >= height) return;
	int tid = tidx + tidy * width;

	int halfSize = windowSize / 2;

	if(tid == 0)
		printf("windowSize: %d, halfSize: %d\n", windowSize, halfSize);

	// Borders do not change from the input
	if (	tid % height < halfSize ||
		height - (tid % height) - 1 < halfSize ||
		tid / height < halfSize ||
		width - (tid / height) - 1 < halfSize)
	{
		outHSV[tid] = inHSV[tid];
	}
	else
	{
		// Allocate memory for array of size windowSize*windowSize that will be sorted
		float3 *sortTab = new float3[windowSize * windowSize];

		// Double for loop to fill the array with pixel around the center pixel
		for (int x = -halfSize; x <= halfSize; x++)
		{
			for (int y = -halfSize; y <= halfSize; y++)
			{
				// Compute temp tid of pixel that will be added
				int tempTid = tid - (y * height + x);
				// Add the pixel to the array
				sortTab[(x + halfSize) * windowSize + (y + halfSize)] = inHSV[tempTid];
			}
		}

		// Function that sort the array
		sort(sortTab, windowSize * windowSize);

		// The output is the median value of the array
		outHSV[tid] = sortTab[windowSize + 1];

		// Free the sorting tab
		free(sortTab);
	}
}
*/


float student1(const PPMBitmap &in, PPMBitmap &out, const int size) {
    //ChronoGPU chrUP, chrDOWN, chrGPU;

    // Setup
    //chrUP.start();

    // Get input dimensions
    int width = in.getWidth(); int height = in.getHeight();

    // Compute number of pixels
    int pixelCount = width * height;

    uchar3 *devRGB;
    uchar3 *devRGBOutput;
    float3 *devHSV;
    float3 *devHSVOutput;

    // Allocate device memory
    cudaMalloc(&devRGB, pixelCount * sizeof(uchar3));
    cudaMalloc(&devRGBOutput, pixelCount * sizeof(uchar3));
    cudaMalloc(&devHSV, pixelCount * sizeof(float3));
    cudaMalloc(&devHSVOutput, pixelCount * sizeof(float3));

    // Convert input from PPMBitmap to uchar3
    uchar3 hostImage[pixelCount];
    int i = 0;
    for (int w = 0; w < width; w ++) {
    	for (int h = 0; h < height; h ++) {
    		PPMBitmap::RGBcol pixel = in.getPixel( w, h );
    		hostImage[i++] = make_uchar3(pixel.r, pixel.g, pixel.b);
    	}
    }

    // Copy memory from host to device
    cudaMemcpy(devRGB, hostImage, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	//chrUP.stop();

    // Processing
    //chrGPU.start();

    // Setup kernel block and grid size
    dim3 blockSize = dim3(16, 16);
    dim3 gridSize = dim3(ceilf(static_cast<float>(width) / blockSize.x),
     			 ceilf(static_cast<float>(height) / blockSize.y));
	printf("blockSize:%d %d, gridSize:%d %d\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);

    // Convertion from RGB to HSV
    rgb2hsv<<<gridSize, blockSize>>>(devRGB, devHSV, width, height);

	// Median Filter
	printf("width: %d, height: %d\n", width, height);
//    medianFilter<<<gridSize, blockSize>>>(devHSV, devHSVOutput, width, height, size);

	// Convertion from HSV to RGB
//    hsv2rgb<<<gridSize, blockSize>>>(devHSVOutput, devRGBOutput, width, height);
    hsv2rgb<<<gridSize, blockSize>>>(devHSV, devRGB, width, height);

	//chrGPU.stop();

    // Cleaning
    //======================
    //chrDOWN.start();
    // Copy memory from device to host
//    cudaMemcpy(hostImage, devRGBOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostImage, devRGB, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

    // Convert output from uchar3 to PPMBitmap
    i = 0;
    for (int w = 0; w < width; w ++) {
    	for (int h = 0; h < height; h ++) {
    		out.setPixel( w, h, PPMBitmap::RGBcol(hostImage[i].x, hostImage[i].y, hostImage[i].z) );
    		i++;
    	}
    }

    // Free device Memory
	cudaFree(&devRGB);
	cudaFree(&devRGBOutput);
	cudaFree(&devHSV);
	cudaFree(&devHSVOutput);

	//chrDOWN.stop();


    // Return
    //======================
    //return chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime(); 
    return 0.f;
}
