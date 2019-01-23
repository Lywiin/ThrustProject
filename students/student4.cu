#include "student4.hpp"
#include "../utils/common.hpp"
#include "../utils/chronoGPU.hpp"
#include "../utils/utils.cuh"

struct Edge{
	int tidSrc;
	int tidDest;
	int weight;
};

struct Graph{
	int V, E;
	Edge* edge;
}

/*
* You have here to compute the segmented image from the filtered one.
* Calculations have to be done using Cuda.
*
* @param in: input (filtered) image
* @param out: output (segmented) image
* @param threshold: thresholding value (remove the edges greater than it)
*/
float student4(const PPMBitmap& in, PPMBitmap& out, const int threshold) {
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
	uchar3 *devRGBOutput;

	// Allocate device memory
	cudaMalloc(&devRGB, pixelCount * sizeof(uchar3));
	cudaMalloc(&devRGBOutput, pixelCount * sizeof(uchar3));

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
	chrUP.stop();



	//*************
	// PROCESSING
	//*************
	chrGPU.start();

	// Create Graph
//	medianFilter<<<gridSize, blockSize>>>(devHSV, devHSVOutput, width, height, size);

	chrGPU.stop();



	//*************
	// CLEANING
	//*************
	chrDOWN.start();
	// Copy memory from device to host
//	cudaMemcpy(hostImage, devRGBOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
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

	chrDOWN.stop();



	//*************
	// RETURN
	//*************
	return chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime(); 
    return 0.f;
}
