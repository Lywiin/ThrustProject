#include "student2.hpp"
#include "../utils/common.hpp"
//#include "../utils/chronoGPU.hpp"
#include "../utils/utils.cuh"

#include <iostream>

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

__device__
void sort( float3* inTab, int tabSize )
{
	int i = 0;
	while (i < tabSize - 1)
	{
		if (inTab[i].z > inTab[i + 1].z)
		{
			float3 temp = inTab[i];
			inTab[i] = inTab[i + 1];
			inTab[i + 1] = temp;

			if (i > 0){ i--; }else{ i++; }
		}else
		{
			i++;
		}
	}
}

// Apply median filter on HSV image
// Launched with 2D grid
__global__
void medianFilter( const float3 *inHSV, float3 *const outHSV, const int width, const int height, const int windowSize ) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx > width) return;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy > height) return;
	int tid = tidx + tidy * width;

	int halfSize = windowSize / 2;

	if (	tid % width < halfSize ||
		width - (tid % width) - 1 < halfSize ||
		tid / height < halfSize ||
		height - (tid / height) - 1 < halfSize)
	{
		outHSV[tid] = inHSV[tid];
	}
	else
	{

		if (tid == width + 1)
		{
			printf("%d ", tid); 
			printf("%d ", tidx - halfSize); 
			printf("%d ", tidx + halfSize); 
			printf("%d \n", halfSize);
		}

		float3 *sortTab = static_cast<float3 *>(malloc(windowSize * windowSize * sizeof(float3)));
		int index = 0;

		for (int y = tidy - halfSize; y <= tidy + halfSize; y++)
		{
			for (int x = tidx - halfSize; x <= tidx + halfSize; x++)
			{
				int tempTid = x + y * width;
				sortTab[index] = inHSV[tempTid];
				index++;
			}

		}

		if (tid == width + 1)
		{
			for (int i = 0; i < 9; i++)
				printf("%f ", sortTab[i].z); printf("\n");
		}

		sort(sortTab, windowSize * windowSize);

		if (tid == width + 1)
		{
			for (int i = 0; i < 9; i++)
				printf("%f ", sortTab[i].z); printf("\n");
		}


		outHSV[tid] = sortTab[windowSize + 1];
		free(sortTab);
	}

//	outHSV[tid] = inHSV[tid];

}


/* Exercice 2.
* Here, you have to apply the Median Filter on the input image.
* Calculations have to be done using CUDA. 
* You have to measure the computation times, and to return the number of ms 
* your calculation took (do not include the memcpy).
*
* @param in: input image
* @param out: output (filtered) image
* @param size: width of the kernel
*/
float student2(const PPMBitmap &in, PPMBitmap &out, const int size) {
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
    dim3 blockSize = dim3(32, 32);
    dim3 gridSize = dim3((width  + (blockSize.x-1))/blockSize.x,
     			 (height + (blockSize.y-1))/blockSize.y );

    // Convertion from RGB to HSV
    rgb2hsv<<<gridSize, blockSize>>>(devRGB, devHSV, width, height);

	// Median Filter
    medianFilter<<<gridSize, blockSize>>>(devHSV, devHSVOutput, width, height, 3);
	printf("ALLOW\n");

	// Convertion from HSV to RGB
    hsv2rgb<<<gridSize, blockSize>>>(devHSVOutput, devRGBOutput, width, height);

	//chrGPU.stop();

    // Cleaning
    //======================
    //chrDOWN.start();
    // Copy memory from device to host
    cudaMemcpy(hostImage, devRGBOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
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
	cudaFree(&devHSVOutput);

	//chrDOWN.stop();


    // Return
    //======================
    //return chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime(); 
    return 0.f;
}
