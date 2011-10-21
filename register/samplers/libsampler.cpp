#include <iostream>
#include "ndarray.h"
#include "math.h"

using namespace std;

extern "C" {

#include <Python.h>

/*

A mapping function which adjusts coordinates outside the domain in the following way:
   
   'n' nearest : sampler the nearest valid coordinate.

   'r' reflect : reflect the coordinate into the boundary.

   'w' wrap : wrap the coordinate around the appropriate axis.
*/

int map(int *coords, int rows, int cols, char mode)
  {
    switch (mode) 
      {
	
      case 'c': /* constant */
	
	if (coords[0] < 0) return 0;
	if (coords[1] < 0) return 0;

	if (coords[0] == rows) return 0;
	if (coords[1] == cols) return 0;

	if (coords[0] > rows) return 0;
	if (coords[1] > cols) return 0;

	break;

      case 'n': /* nearest */

	if (coords[0] < 0) coords[0] = 0;
	if (coords[1] < 0) coords[1] = 0;

	if (coords[0] == rows) coords[0] = rows-1;
	if (coords[1] == cols) coords[1] = cols-1;

	if (coords[0] > rows) coords[0] = rows;
	if (coords[1] > cols) coords[1] = cols;

	break;

      case 'r': /* reflect */

	if (coords[0] < 0) coords[0] = fmod(-coords[0],rows);
	if (coords[1] < 0) coords[1] = fmod(-coords[1],cols);

	if (coords[0] == rows) coords[0] = rows-1;
	if (coords[1] == cols) coords[1] = cols-1;

	if (coords[0] > rows) coords[0] = rows;
	if (coords[1] > cols) coords[1] = cols;

	break;

      case 'w': /* wrap */

	if (coords[0] < 0) coords[0] = rows - fmod(-coords[0],rows);
	if (coords[1] < 0) coords[1] = cols - fmod(-coords[1],cols);

	if (coords[0] == rows) coords[0] = 0;
	if (coords[1] == cols) coords[1] = 0;

	if (coords[0] > rows) coords[0] = fmod(coords[0],rows);
	if (coords[1] > cols) coords[1] = fmod(coords[1],cols);

	break;
    }
    
    return 1;
  }

/*
  nearest neighbour interpolator.
 */

int nearest(numpyArray<double> array0,
            numpyArray<double> array1,
            numpyArray<double> array2,
	    char mode,
	    double cvalue
           )
{
    Ndarray<double,3> warp(array0);
    Ndarray<double,2> image(array1);
    Ndarray<double,2> result(array2);

    int coords[2] = {0, 0};

    int rows = image.getShape(0);
    int cols = image.getShape(1);

    for (int i = 0; i < warp.getShape(1); i++)
    {
      for (int j = 0; j < warp.getShape(2); j++)
        {
        	coords[0] = (int)warp[0][i][j];
        	coords[1] = (int)warp[1][i][j];

		if ( not map(coords, rows, cols, mode) ) 
		  {
		    result[i][j] = cvalue;
		  }
		else
		  {
		    result[i][j] = image[coords[0]][coords[1]];
		  }
        }
    }

    return 0;
}


/* 
   bilinear interpolator 
 */

int bilinear(numpyArray<double> array0,
             numpyArray<double> array1,
             numpyArray<double> array2,
	     char mode,
	     double cvalue
        )
{
    Ndarray<double,3> warp(array0);
    Ndarray<double,2> image(array1);
    Ndarray<double,2> result(array2);

    double di = 0.0;
    double dj = 0.0;

    double fi = 0;
    double fj = 0;

    double w0 = 0.0;
    double w1 = 0.0;
    double w2 = 0.0;
    double w3 = 0.0;
    
    int tl[2] = {0, 0};
    int tr[2] = {0, 0};
    int ll[2] = {0, 0};
    int lr[2] = {0, 0};

    int rows = image.getShape(0);
    int cols = image.getShape(1);

    for (int i = 0; i < warp.getShape(1); i++)
    {
      for (int j = 0; j < warp.getShape(2); j++)
	{
	  /* Floating point coordinates */
	  fi = warp[0][i][j];
	  fj = warp[1][i][j];
		
	  /* Integer component */
	  di = (double)((int)(warp[0][i][j]));
	  dj = (double)((int)(warp[1][i][j]));
	
	  /* Defined sampling coordinates */
	  
	  tl[0] = (int)fi;
	  tl[1] = (int)fj;
	
	  tr[0] = tl[0];
	  tr[1] = tl[1] + 1;

	  ll[0] = tl[0] + 1;
	  ll[1] = tl[1];

	  lr[0] = tl[0] + 1;
	  lr[1] = tl[1] + 1;
	  
	  w0 = 0.0;
	  if ( map(tl, rows, cols, mode) ) 
	    w0 = ((dj+1-fj)*(di+1-fi))*image[tl[0]][tl[1]];
	  			  
	  w1 = 0.0;
	  if ( map(tr, rows, cols, mode) )
	    w1 = ((fj-dj)*(di+1-fi))*image[tr[0]][tr[1]]; 
		  
	  w2 = 0.0;
	  if ( map(ll, rows, cols, mode) )
	    w2 = ((dj+1-fj)*(fi-di))*image[ll[0]][ll[1]];  
		  
	  w3 = 0.0;
	  if ( map(lr, rows, cols, mode) )
	    w3 = ((fj-dj)*(fi-di))*image[lr[0]][lr[1]];  
		  
	  result[i][j] = w0 + w1 + w2 + w3;
	        	
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

static PyMethodDef sampler_methods[] = {
    {NULL, NULL}
};

void initlibsampler()
{
    (void) Py_InitModule("libsampler", sampler_methods);
}

} // end extern "C"
