#include <iostream>
#include "ndarray.h"
#include "math.h"
#include "stdio.h"
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
    printf("Matrix\n");
    //Matrix imageData((float*) &(image[0][0]), rows, cols);
    Matrix* imageData = new Matrix(new float[rows*cols], rows, cols);
    printf("Transform\n");
    WaveletTransform transform(levels);
    printf("Forward\n");
    Matrix* haarData = transform.DoForward(imageData);
    
    // Find salient features
    printf("Forest\n");
    SaliencyForest forest(*haarData, levels);
    printf("Roots\n");
    //forest.CalcRoots();
    int numPoints = 0;
    printf("Points\n");
    //SalientPoint** points = forest.GetSaliencyPoints(VBLOCKS, HBLOCKS, THRESHOLD, CHESSBOARD, numPoints);
    printf("for\n");
    for (int i = 0; i < numPoints; i++)
    {
        if (i < maxpoints)
        {
    //        result[i][0] = i;
    //        result[i][1] = points[i]->X;
    //        result[i][2] = points[i]->Y;
        }
    }
    //delete[] points;
    return 0;

}

} // end extern "C"
