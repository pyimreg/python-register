#include <iostream>
#include "ndarray.h"
#include "math.h"

using namespace std;

extern "C" {


int nearest(numpyArray<double> array0,
            numpyArray<double> array1,
            numpyArray<double> array2
        )
{
    Ndarray<double,3> warp(array0);
    Ndarray<double,2> image(array1);
    Ndarray<double,2> result(array2);

    int di = 0;
    int dj = 0;

    int rows = image.getShape(0);
    int cols = image.getShape(1);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
        	di = (int)warp[0][i][j];
        	dj = (int)warp[1][i][j];

        	if ( ( di < rows && di >= 0 ) &&
        		 ( dj < cols && dj >= 0 ) )
        	{
        		result[i][j] = image[di][dj];
        	}
        	else
        	{
        		result[i][j] = 0.0;
        	}
        }
    }

    return 0;
}


int bilinear(numpyArray<double> array0,
             numpyArray<double> array1,
             numpyArray<double> array2
        )
{
    Ndarray<double,3> warp(array0);
    Ndarray<double,2> image(array1);
    Ndarray<double,2> result(array2);

    int u = 0;
    int v = 0;
    int iu = 0;
    int iv = 0;

    double w0 = 0.0;
    double w1 = 0.0;
    double w2 = 0.0;
    double w3 = 0.0;

    int rows = image.getShape(0);
    int cols = image.getShape(1);

    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
        	iu = (int)warp[1][y][x];
        	iv = (int)warp[0][y][x];

        	u = floor(warp[1][y][x]);
        	v = floor(warp[0][y][x]);

        	w0 = (u+1-x)*(v+1-y);
        	w1 = (x-u)*(v+1-y);
        	w2 = (u+1-x)*(y-v);
        	w3 = (x-u)*(y-v);

        	if ( ( iv < rows-1 && iv >= 1 ) &&
                 ( iu < cols-1 && iu >= 1 ) )
            {
        		result[y][x] = (w0*image[iv][iu]) +
        					   (w1*image[iv][iu+1]) +
        					   (w2*image[iv+1][iu]) +
        					   (w3*image[iv+1][iu+1]);
        	}
        	else
        	{
        		result[y][x] = 0.0;
        	}
        }
    }

    return 0;
}


int cubicConvolution(numpyArray<double> array0,
            numpyArray<double> array1,
            numpyArray<double> array2
        )
{
    Ndarray<double,3> warp(array0);
    Ndarray<double,2> image(array1);
    Ndarray<double,2> result(array2);

    int di = 0;
    int dj = 0;

    int rows = image.getShape(0);
    int cols = image.getShape(1);

    double xShift;
    double yShift;
    double xArray0;
    double xArray1;
    double xArray2;
    double xArray3;
    double yArray0;
    double yArray1;
    double yArray2;
    double yArray3;
    double c0;
    double c1;
    double c2;
    double c3;
    
    for (int i = 0; i < image.getShape(0); i++)
    {
        for (int j = 0; j < image.getShape(1); j++)
        {
            di = (int)floor(warp[0][i][j]);
            dj = (int)floor(warp[1][i][j]);

            if ( ( di < rows-2 && di >= 2 ) &&
                 ( dj < cols-2 && dj >= 2 ) )
            {
                xShift = warp[1][i][j] - dj;
                yShift = warp[0][i][j] - di; 
                xArray0 = -(1/2.0)*pow(xShift, 3) + pow(xShift, 2) - (1/2.0)*xShift;
                xArray1 = (3/2.0)*pow(xShift, 3) - (5/2.0)*pow(xShift, 2) + 1;
                xArray2 = -(3/2.0)*pow(xShift, 3) + 2*pow(xShift, 2) + (1/2.0)*xShift;
                xArray3 = (1/2.0)*pow(xShift, 3) - (1/2.0)*pow(xShift, 2);
                yArray0 = -(1/2.0)*pow(yShift, 3) + pow(yShift, 2) - (1/2.0)*yShift;
                yArray1 = (3/2.0)*pow(yShift, 3) - (5/2.0)*pow(yShift, 2) + 1;
                yArray2 = -(3/2.0)*pow(yShift, 3) + 2*pow(yShift, 2) + (1/2.0)*yShift;
                yArray3 = (1/2.0)*pow(yShift, 3) - (1/2.0)*pow(yShift, 2);                
                c0 = xArray0 * image[di-1][dj-1] + xArray1 * image[di-1][dj+0] + xArray2 * image[di-1][dj+1] + xArray3 * image[di-1][dj+2];
                c1 = xArray0 * image[di+0][dj-1] + xArray1 * image[di+0][dj+0] + xArray2 * image[di+0][dj+1] + xArray3 * image[di+0][dj+2];
                c2 = xArray0 * image[di+1][dj-1] + xArray1 * image[di+1][dj+0] + xArray2 * image[di+1][dj+1] + xArray3 * image[di+1][dj+2];
                c3 = xArray0 * image[di+2][dj-1] + xArray1 * image[di+2][dj+0] + xArray2 * image[di+2][dj+1] + xArray3 * image[di+2][dj+2];
                result[i][j] = c0 * yArray0 + c1 * yArray1 + c2 * yArray2 + c3 * yArray3;
            }
            else
            {
                result[i][j] = 0.0;
            }
        }
    }

    return 0;
}

} // end extern "C"
