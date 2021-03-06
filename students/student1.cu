#include "student1.hpp"
#include "student2.hpp"
#include "../utils/common.hpp"
#include "../utils/chronoGPU.hpp"
#include "../utils/utils.cuh"

#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <math.h>

class MedianFunctor : public thrust::unary_function<float3, float3> {
	const thrust::device_ptr<float3> m_d_HSVt;
	const int m_width;
	const int m_height;
	const int m_size;
	const int m_halfSize;

	thrust::device_ptr<float3> m_sortTab;

public:
	__host__ __device__ MedianFunctor() = delete;
	__host__ __device__ MedianFunctor(thrust::device_ptr<float3> d_HSVt, const int width, const int height, const int size, const int halfSize)  
					: m_width(width), m_height(height), m_d_HSVt(d_HSVt), m_size(size), m_halfSize(halfSize) {}
	MedianFunctor(const MedianFunctor&) = default;

	// Fill an array in preparation for sorting
	__host__ __device__
	void fill( float3* inTab, int tid, int size )
	{
		int halfSize = size / 2;

		// Double for loop to fill the array with pixel around the center pixel
		for (int x = -halfSize; x <= halfSize; x++)
		{
			for (int y = -halfSize; y <= halfSize; y++)
			{
				// Compute temp tid of pixel that will be added
				int tempTid = tid - (y * m_height + x);
				// Add the pixel to the array
				inTab[(x + halfSize) * size + (y + halfSize)] = m_d_HSVt[tempTid];
			}
		}
	}

	// Sort an array by ascending order using bubble sort method
	__host__ __device__
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

	__host__ __device__ float3 operator()(int tid) {

		float3 sortTab[225];
		int s = m_size;

		int leftB = tid % m_height;
		int rightB = m_height - (tid % m_height) - 1;
		int upB = tid / m_height;
		int downB = m_width - (tid / m_height) - 1;

		int minLR = (leftB < rightB ? leftB : rightB);
		int minUP = (upB < downB ? upB : downB);
		int min = (minLR < minUP ? minLR : minUP);

		if ( leftB < m_halfSize || rightB < m_halfSize || upB < m_halfSize || downB < m_halfSize )
			s = min * 2 + 1;

		// Filter of size 1x1, no need to sort
		if (s == 1)
			return m_d_HSVt[tid];

		// Fill the array with window's values
		fill(sortTab, tid, s);

		// Sort the array in ascending order
		sort(sortTab, s * s);

		// The output is the median value of the array
		return sortTab[s + 1];

	}
};

float student1(const PPMBitmap &in, PPMBitmap &out, const int size) {
	ChronoGPU chrUP, chrDOWN, chrGPU;

	//*************/
	// SETUP
	//*************/
	chrUP.start();

	// Get input dimensions
	int width = in.getWidth(); int height = in.getHeight();
	printf("width:%d, height:%d\n", width, height);

	// Setup kernel block and grid size
	dim3 blockSize = dim3(16, 16);
	dim3 gridSize = dim3(	ceilf(static_cast<float>(width) / blockSize.x),
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


	//*************/
	// PROCESSING
	//*************/
	chrGPU.start();

	// CONVERTION RGB TO HSV
	//======================
	// Copy memory from host to device
	cudaMemcpy(devRGB, hostImage, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);
	// Convertion from RGB to HSV
	rgb2hsv<<<gridSize, blockSize>>>(devRGB, devHSV, width, height);
	// Copy memory from device to host
	cudaMemcpy(hostImageHSV, devHSV, pixelCount * sizeof(float3), cudaMemcpyDeviceToHost);


	// MEDIAN FILTER WITH THRUST
	//======================
	// Setup Thrust
	thrust::host_vector<float3> HSVt(pixelCount);
	thrust::host_vector<float3> HSVtOutput(pixelCount);
	for(int i = pixelCount; i--; ) {
		HSVt[i] = hostImageHSV[i];
		HSVtOutput[i] = make_float3(0, 0, 0);
	}
	thrust::device_vector<float3> d_HSVt(HSVt);
	thrust::device_vector<float3> d_HSVtOutput(HSVt.size());

	thrust::device_ptr<float3> d_HSVt_val = d_HSVt.data();


	// Process Thrust
	thrust::copy_n(
		thrust::make_transform_iterator(
			thrust::make_counting_iterator(static_cast<int>(0)),
			MedianFunctor(d_HSVt_val, width, height, size, size / 2)),
		pixelCount,
		d_HSVtOutput.begin()
	);



	// Result Thrust
	HSVtOutput = d_HSVtOutput;
	for(int i = pixelCount; i--; ) {
		hostImageHSV[i] = HSVtOutput[i];
	}


	// CONVERTION HSV TO RGB
	//======================
	// Copy memory from host to device
	cudaMemcpy(devHSV, hostImageHSV, pixelCount * sizeof(float3), cudaMemcpyHostToDevice);
	// Convertion from HSV to RGB
	hsv2rgb<<<gridSize, blockSize>>>(devHSV, devRGB, width, height);
	// Copy memory from device to host
	cudaMemcpy(hostImage, devRGB, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

	chrGPU.stop();



	//*************/
	// CLEANING
	//*************/
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



	//*************/
	// RETURN
	//*************/
	return chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime();
}
