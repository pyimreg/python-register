#include <iostream>
#include "ndarray.h"
#include "math.h"
#include "stdio.h"

#define MAX_POINTS_PER_BLOCK 4
#include "SaliencyForest.h"

#define VBLOCKS 10
#define HBLOCKS 10
#define THRESHOLD 0.1
#define CHESSBOARD false

using namespace std;

// The public API follows:

extern "C" {

int haar(   numpyArray<double> array0,
            numpyArray<int> array1,
            int levels
        )
{
    Ndarray<double,2> image(array0);
    Ndarray<int,2> result(array1);

    int rows = image.getShape(0);
    int cols = image.getShape(1);
    int maxpoints = result.getShape(0);

    // Do Haar tranform on image
    Matrix imageData(rows, cols);
    for (int i = 0; i < cols; i++)
        for (int j = 0; j < rows; j++)
            imageData.data[i * rows + j] = (float)(image[j][i]);
    WaveletTransform<HaarLifter> transform(levels);
    Matrix haarData = transform.DoForward(imageData);
    
    // Find salient features
    SaliencyForest forest(haarData, levels);
    forest.CalcRoots();
    int numPoints = 0;
    SalientPoint** points = forest.GetSaliencyPoints(VBLOCKS, HBLOCKS, THRESHOLD, CHESSBOARD, numPoints);
    int cnt = 0;
    for (int i = 0; i < numPoints; i++)
    {
        if ((points[i]->X < cols) && (points[i]->Y < rows))
        {
            if (cnt < maxpoints)
            {
                cnt++;
                result[i][0] = i;
                result[i][1] = points[i]->X;
                result[i][2] = points[i]->Y;
                printf("%d : (%d, %d)\n", i, points[i]->X, points[i]->Y);
            }
        }
    }
    delete[] points;
    return 0;

}

} // end extern "C"
