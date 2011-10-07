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

    double di = 0.0;
    double dj = 0.0;

    double fi = 0;
    double fj = 0;
    
    int ii = 0;
    int jj = 0;

    double w0 = 0.0;
    double w1 = 0.0;
    double w2 = 0.0;
    double w3 = 0.0;

    int rows = image.getShape(0);
    int cols = image.getShape(1);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
	{
	        // Floating point coordinates
	        fi = warp[0][i][j];
        	fj = warp[1][i][j];
		
		// Integer component
		di = (double)((int)(warp[0][i][j]));
  	        dj = (double)((int)(warp[1][i][j]));
		
		ii = (int)di;
		jj = (int)dj;
		
        	if ( ( ii < rows && ii >= 0 ) &&
                     ( jj < cols && jj >= 0 ) )
		{
		  
		  // Bilinear interpolation weights
		  w0 = ((dj+1-fj)*(di+1-fi))*image[ii][jj];
		  
		  w1 = 0.0;
		  if (jj+1 < cols)
		    w1 = ((fj-dj)*(di+1-fi))*image[ii][jj+1];  
		  
		  w2 = 0.0;
		  if (ii+1 < rows)
		    w2 = ((dj+1-fj)*(fi-di))*image[ii+1][jj];  
		  
		  w3 = 0.0;
		  if ((ii+1 < rows) && (jj+1 < cols))
		    w3 = ((fj-dj)*(fi-di))*image[ii+1][jj+1];  
		  
		  result[i][j] = w0 + w1 + w2 + w3;
		
        	}
        	else
        	{
		  result[i][j] = 0.0;

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
