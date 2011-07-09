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

    for (int i = 0; i < image.getShape(0); i++)
    {
        for (int j = 0; j < image.getShape(1); j++)
        {
        	di = (int)warp[0][i][j];
        	dj = (int)warp[1][i][j];

        	if ( ( di < rows && di > 0 ) &&
        		 ( dj < cols && dj > 0 ) )
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

} // end extern "C"
