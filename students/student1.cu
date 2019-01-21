#include "student1.hpp"
#include "student2.hpp"
#include "../utils/common.hpp"
#include "../utils/chronoGPU.hpp"
#include "../utils/utils.cuh"

#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <math.h>

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


class MedianFilter : public thrust::unary_function<float3, float3> {
	public:
	__device__ float3 operator()(float3 &inHSV) {
		//return int( color.color == ColoredObject::BLUE );
		return inHSV;
	}
};

float student1(const PPMBitmap &in, PPMBitmap &out, const int size) {
	ChronoGPU chrUP, chrDOWN, chrGPU;

	//*************
	// SETUP
	//*************
	chrUP.start();

	// Get input dimensions
	int width = in.getWidth(); int height = in.getHeight();

	// Setup kernel block and grid size
	dim3 blockSize = dim3(16, 16);
	dim3 gridSize = dim3(ceilf(static_cast<float>(width) / blockSize.x),
	 			 ceilf(static_cast<float>(height) / blockSize.y));
	printf("blockSize:%d %d, gridSize:%d %d\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);

	// Compute number of pixels
	int pixelCount = width * height;

	uchar3 *devRGB;
	float3 *devHSV;

	// Allocate device memory
	cudaMalloc(&devRGB, pixelCount * sizeof(uchar3));
	cudaMalloc(&devHSV, pixelCount * sizeof(float3));

	float3 hostImageHSV[pixelCount];

	// Convert input from PPMBitmap to uchar3
	uchar3 hostImage[pixelCount];
	int i = 0;
	for (int w = 0; w < width; w ++) {
		for (int h = 0; h < height; h ++) {
			PPMBitmap::RGBcol pixel = in.getPixel( w, h );
			hostImage[i++] = make_uchar3(pixel.r, pixel.g, pixel.b);
		}
	}

	chrUP.stop();



	//*************
	// PROCESSING
	//*************
	chrGPU.start();

	// CONVERTION RGB TO HSV
	//======================
	// Copy memory from host to device
	cudaMemcpy(devRGB, hostImage, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	// Convertion from RGB to HSV
	rgb2hsv<<<gridSize, blockSize>>>(devRGB, devHSV, width, height);
	// Copy memory from device to host
	cudaMemcpy(hostImageHSV, devHSV, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);


	// MEDIAN FILTER WITH THRUST
	//======================
	// Setup Thrust
	thrust::host_vector<float3> HSVt(pixelCount);
	thrust::host_vector<float3> HSVtOutput(pixelCount);
	for(int i = pixelCount; i--; ) {
		HSVt[i] = hostImageHSV[i];
		HSVtOutput[i] = make_float3(-1, -1, -1);
	}
	thrust::device_vector<float3> d_HSVt(HSVt);
	thrust::device_vector<float3> d_HSVtOutput(HSVt.size());

	// Process Thrust
	thrust::transform(
		d_HSVt.begin(), d_HSVt.end(),
		d_HSVtOutput.begin(),
		MedianFilter()
	);

	// Result Thrust
	HSVtOutput = d_HSVtOutput;
	for(int i = pixelCount; i--; ) {
		hostImageHSV[i] = HSVtOutput[i];
	}


	// CONVERTION HSV TO RGB
	//======================
	// Copy memory from host to device
	cudaMemcpy(devHSV, hostImageHSV, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	// Convertion from HSV to RGB
	hsv2rgb<<<gridSize, blockSize>>>(devHSV, devRGB, width, height);
	// Copy memory from device to host
	cudaMemcpy(hostImage, devRGB, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

	chrGPU.stop();



	//*************
	// CLEANING
	//*************
	chrDOWN.start();
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
	cudaFree(&devHSV);

	chrDOWN.stop();



	//*************
	// RETURN
	//*************
	return chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime();
}
