#include "student4.hpp"
#include "../utils/common.hpp"
#include "../utils/chronoGPU.hpp"
#include "../utils/utils.cuh"

struct Edge{
	int src;
	int dest;
	int weight;
};

struct Graph{
	int V, E;
	Edge* edge;
};

// Creates a graph with V vertices and E edges
struct Graph* createGraph(int V, int E)
{
	Graph* graph = new Graph;
	graph->V = V;
	graph->E = E;
	graph->edge = new Edge[E];
	return graph;
}

// Destroy a graph
void destroyGraph(struct Graph* graph)
{
	free(graph->edge);
	free(graph);
}

__device__
int computeWeight(const uchar3* inHSV, const int srcTid, const int destTid)
{
	int x = inHSV[srcTid].x - inHSV[destTid].x >= 0 ? inHSV[srcTid].x - inHSV[destTid].x : (inHSV[srcTid].x - inHSV[destTid].x) * -1;
	int y = inHSV[srcTid].y - inHSV[destTid].y >= 0 ? inHSV[srcTid].y - inHSV[destTid].y : (inHSV[srcTid].y - inHSV[destTid].y) * -1;
	int z = inHSV[srcTid].z - inHSV[destTid].z >= 0 ? inHSV[srcTid].z - inHSV[destTid].z : (inHSV[srcTid].z - inHSV[destTid].z) * -1;
	return x + y + z;
}

__global__
void InitializeEdges(const uchar3* inHSV, Edge* edge, const int E, const int width, const int height)
{
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	if (tidx >= width) return;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y;
	if (tidy >= height) return;
	int tid = tidx + tidy * width;

	//corner
	if (tid != 0) {
		// not up
		if (!(tid / height == 0))
		{
			int destTid = tid - height;
			edge[tid - height].src = tid;
			edge[tid - height].dest = destTid;
			edge[tid - height].weight = computeWeight(inHSV, tid, destTid);
		}
		// not left
		if (!(tid % height == 0))
		{
			int destTid = tid - 1;
			edge[E / 2 + tid - 1].src = tid;
			edge[E / 2 + tid - 1].dest = destTid;
			edge[E / 2 + tid - 1].weight = computeWeight(inHSV, tid, destTid);
		}
	}
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
//	ChronoGPU chrUP, chrDOWN, chrGPU;

	//*************/
	// SETUP
	//*************/

//	chrUP.start();

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

	// Prepare Graph
	int E = (width - 1) * (height - 1);
	struct Graph* graph = createGraph(pixelCount, E);

	Edge *devEdge;
	cudaMalloc(&devEdge, E * sizeof(Edge));

	// Copy memory from host to device
	cudaMemcpy(devRGB, hostImage, pixelCount * sizeof(uchar3), cudaMemcpyHostToDevice);

	// Compute every edge (not working)
/*
	cudaMemcpy(devEdge, graph->edge, E * sizeof(Edge), cudaMemcpyHostToDevice);
	InitializeEdges<<<gridSize, blockSize>>>(devRGB, devEdge, E, width, height);
	cudaMemcpy(graph->edge, devEdge, E * sizeof(Edge), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 10; i++)
	{
		printf("%d %d %d\n", graph->edge[i].src, graph->edge[i].dest, graph->edge[i].weight);
	}
*/
//	chrUP.stop();



	//*************/
	// PROCESSING
	//*************/

//	chrGPU.start();


//	chrGPU.stop();



	//*************/
	// CLEANING
	//*************/

	// Destroy allocated graph
	destroyGraph(graph);

//	chrDOWN.start();
	// Copy memory from device to host
	cudaMemcpy(hostImage, devRGBOutput, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);
//	cudaMemcpy(hostImage, devRGB, pixelCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

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
	cudaFree(&devEdge);

//	chrDOWN.stop();

//	cudaError_t err = cudaGetLastError();
//	if (err != cudaSuccess)
//		printf("Error: %s\n", cudaGetErrorString(err));


	//*************/
	// RETURN
	//*************/
//	return chrUP.elapsedTime() + chrDOWN.elapsedTime() + chrGPU.elapsedTime();
	return 0.f;
}
