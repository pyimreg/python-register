#include "HaarLifter.h"

class WaveletTransform {
private:
	HaarLifter _lifter;
	int _levels;

public:
	WaveletTransform(int levels) : 
	  _lifter()
	{
		_levels = levels;
	}

	Matrix* DoForward(Matrix* data) 
	{
		printf("data2\n");
		Matrix* data2 = Enlarge(data, _levels); //add padding
		

		printf("copy\n");
		/*
		for (int level = 1; level <= _levels; level++) {
			TransformRows(data, level, Lifting::forward);
			TransformCols(data, level, Lifting::forward);
		}
		*/
		return data2;
	}

	Matrix* Enlarge(Matrix* data, int levels) 
	{
		int extraRows = 0;
		int extraCols = 0;
		printf("while1\n");
		while (((data->rows + extraRows) >> levels) << levels != (data->rows
				+ extraRows))
			extraRows++;
		printf("while2\n");
		while (((data->cols + extraCols) >> levels) << levels != (data->cols
				+ extraCols))
			extraCols++;

		printf("result %d %d\n", extraRows, extraCols);
		Matrix* result = new Matrix(data->rows + extraRows, data->cols + extraCols);
		for (int j = 0; j < data->rows; j++)
		{
			for (int i = 0; i < data->cols; i++)
			{
				printf("(%d,%d)\n", j, i);
				printf("(%d,%d)\n", result->cols, data->cols);
				 
				result->data[j * result->cols + i] = data->data[j * data->cols + i]; 
			}
		}
		
		printf("return\n");
		return result;
	}

	void TransformCols(Matrix* data, int level, Lifting::Direction direction) {
		int n = data->cols / (int) pow(2.0, level - 1);
		for (int i = 0; i < n; i++) {
			data->SelectCol(i);
			if (direction == Lifting::forward)
				_lifter.ForwardTrans(*data, level);
			else if (direction == Lifting::reverse)
				_lifter.InverseTrans(*data, level);
			else
				;
		}
	}

	void TransformRows(Matrix* data, int level, Lifting::Direction direction) {
		int n = data->rows / (int) pow(2.0, level - 1);
		for (int i = 0; i < n; i++) {
			data->SelectRow(i);
			if (direction == Lifting::forward)
				_lifter.ForwardTrans(*data, level);
			else if (direction == Lifting::reverse)
				_lifter.InverseTrans(*data, level);
			else
				;
		}
	}

	void DoInverse(Matrix* data) {
		for (int level = 0; level < _levels; level++) {
			TransformRows(data, level, Lifting::reverse);
			TransformCols(data, level, Lifting::reverse);
		}
	}

};
