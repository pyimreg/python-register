#include <iostream>
#include "ndarray.h"
#include "math.h"
#include "stdio.h"

#define MAX_POINTS_PER_BLOCK 4
#include "SaliencyForest.h"

#define VBLOCKS 10
#define HBLOCKS 10
#define THRESHOLD 0.8
#define CHESSBOARD false

using namespace std;

// The public API follows:

extern "C" {

int haar(   numpyArray<float> array0,
            numpyArray<int> array1,
            int levels
        )
{
    Ndarray<float,2> image(array0);
    Ndarray<int,2> result(array1);

    int rows = image.getShape(0);
    int cols = image.getShape(1);
    int maxpoints = result.getShape(0);

    // Do Haar tranform on image
    Matrix imageData((float*) &(image[0][0]), rows, cols);
    WaveletTransform<HaarLifter> transform(levels);
    Matrix haarData = transform.DoForward(imageData);
    
    // Find salient features
    SaliencyForest forest(haarData, levels);
    forest.CalcRoots();
    int numPoints = 0;
    SalientPoint** points = forest.GetSaliencyPoints(VBLOCKS, HBLOCKS, THRESHOLD, CHESSBOARD, numPoints);
    for (int i = 0; i < numPoints; i++)
    {
        if (i < maxpoints)
        {
            result[i][0] = i;
            result[i][1] = points[i]->X;
            result[i][2] = points[i]->Y;
            printf("%d : (%d, %d)\n", i, points[i]->X, points[i]->Y);
        }
    }
    delete[] points;
    return 0;

}

} // end extern "C"
