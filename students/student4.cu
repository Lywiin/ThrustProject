#include "student4.hpp"

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
    return 0.f;
}
