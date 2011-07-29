#include "HaarLifter.h"

class WaveletTransform {
private:
	BaseLifter &_lifter;
	int _levels;

public:
	WaveletTransform(BaseLifter &waveletLifter, int levels) :
		_lifter(waveletLifter) {
		_levels = levels;
	}

	Matrix DoForward(Matrix &data) {
		Matrix data2 = Enlarge(data, _levels); //add padding

		for (int level = 1; level <= _levels; level++) {
			TransformRows(data2, level, Lifting::forward);
			TransformCols(data2, level, Lifting::forward);
		}

		return data2;
	}

	Matrix Enlarge(Matrix &data, int levels) {
		int extraRows = 0;
		int extraCols = 0;
		while (((data.rows + extraRows) >> levels) << levels != (data.rows
				+ extraRows))
			extraRows++;
		while (((data.cols + extraCols) >> levels) << levels != (data.cols
				+ extraCols))
			extraCols++;

		Matrix result(data.rows + extraRows, data.cols + extraCols);
		for (int j = 0; j < data.cols; j++)
			for (int i = 0; i < data.rows; i++)
				result.data[i + j * result.rows] = data.data[i + j
						* result.rows];

		return result;
	}

	void TransformCols(Matrix &data, int level, Lifting::Direction direction) {
		int n = data.cols / (int) pow(2.0, level - 1);
		for (int i = 0; i < n; i++) {
			data.SelectCol(i);
			if (direction == Lifting::forward)
				_lifter.ForwardTrans(data, level);
			else if (direction == Lifting::reverse)
				_lifter.InverseTrans(data, level);
			else
				;
		}
	}

	void TransformRows(Matrix &data, int level, Lifting::Direction direction) {
		int n = data.rows / (int) pow(2.0, level - 1);
		for (int i = 0; i < n; i++) {
			data.SelectRow(i);
			if (direction == Lifting::forward)
				_lifter.ForwardTrans(data, level);
			else if (direction == Lifting::reverse)
				_lifter.InverseTrans(data, level);
			else
				;
		}
	}

	void DoInverse(Matrix &data) {
		for (int level = 0; level < _levels; level++) {
			TransformRows(data, level, Lifting::reverse);
			TransformCols(data, level, Lifting::reverse);
		}
	}

};
