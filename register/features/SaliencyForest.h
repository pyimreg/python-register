#include "SalientPoint.h"
#include "WaveletTransform.h"

#define MAX_POINTS_PER_BLOCK 1

class SaliencyForest {
private:
	Matrix data;
	int levels;
	bool ready;

public:
	SaliencyForest(Matrix &aData, int aLevels) :	data(aData.data, aData.rows, aData.cols) 
    {
		levels = aLevels;
		ready = false;
	}

	SaliencyForest(float *fdata, int rows, int cols, int aLevels) 
    {
        data.data = fdata;
        data.rows = rows;
        data.cols = cols;
		levels = aLevels;
		ready = false;
	}

	void CalcRoots() {
		if (ready)
			return;

		//printf("\tCalculating saliency roots...\n");

		float max;

		int rOffset, cOffset, rOffset2, cOffset2, n, m, r, c;
            
		for (int level = 1; level <= levels; level++) 
        {
			n = data.rows / (int) pow(2.0, level);
			m = data.cols / (int) pow(2.0, level);

			rOffset = n;
			cOffset = m;
			rOffset2 = n / 2;
			cOffset2 = m / 2;
			for (int i = 0; i < m / 2; i++) // cols
			{
				for (int j = 0; j < n / 2; j++) // rows
				{
					max = -HUGE;

					// coeef TopLeft
					r = rOffset + j * 2;
					c = cOffset + i * 2;
                    /*printf("\tr,c = %d,%d\n", r,c);
                    printf("\tmax,index = %d,%d\n", data.rows*data.cols, r + c * data.rows);*/
					if (abs(data.data[r + c * data.rows]) > max)
						max = abs(data.data[r + c * data.rows]);

					// coeef BottomLeft
					r = rOffset + j * 2 + 1;
                    //printf("\tr,c = %d,%d\n", r,c);
					if (abs(data.data[r + c * data.rows]) > max)
						max = abs(data.data[r + c * data.rows]);

					// coeef BottomRight
					c = cOffset + i * 2 + 1;
                   // printf("\tr,c = %d,%d\n", r,c);
					if (abs(data.data[r + c * data.rows]) > max)
						max = abs(data.data[r + c * data.rows]);

					// coeef TopRight
					r = rOffset + j * 2;
                    //printf("\tr,c = %d,%d\n", r,c);
					if (abs(data.data[r + c * data.rows]) > max)
						max = abs(data.data[r + c * data.rows]);

					if (level == levels) {
					} else {
						// Add max abs coeff to next level abs coeff
						r = rOffset2 + j;
						c = cOffset2 + i;
						data.data[r + c * data.rows] = abs(data.data[r + c * data.rows]) + max;
					}

				}
			}

			//                #region H Block
			rOffset = 0;
			cOffset = m;
			rOffset2 = 0;
			cOffset2 = m/2;
			for (int i = 0; i < m / 2; i++) // cols
			{
				for (int j = 0; j < n / 2; j++) // rows
				{
					max = -HUGE;

					// coeef TopLeft
					r = rOffset + j * 2;
					c = cOffset + i * 2;
					if (abs(data.data[r+ c * data.rows]) > max)
					max = abs(data.data[r+ c * data.rows]);

					// coeef BottomLeft
					r = rOffset + j * 2 + 1;
					if (abs(data.data[r+ c * data.rows]) > max)
					max = abs(data.data[r+ c * data.rows]);

					// coeef BottomRight
					c = cOffset + i * 2 + 1;
					if (abs(data.data[r+ c * data.rows]) > max)
					max = abs(data.data[r+ c * data.rows]);

					// coeef TopRight
					r = rOffset + j * 2;
					if (abs(data.data[r+ c * data.rows]) > max)
					max = abs(data.data[r+ c * data.rows]);

					if (level == levels)
					{
					}
					else
					{
						// Add max abs coeff to next level abs coeff
						r = rOffset2 + j;
						c = cOffset2 + i;
						data.data[r+ c * data.rows] = abs(data.data[r+ c * data.rows]) + max;
					}
				}
			}

			//#endregion

			//#region V Block
			rOffset = n;
			cOffset = 0;
			rOffset2 = n / 2;
			cOffset2 = 0;

			for (int i = 0; i < m / 2; i++) // cols
			{
				for (int j = 0; j < n / 2; j++) // rows
				{
					max = -HUGE;

					// coeef TopLeft
					r = rOffset + j * 2;
					c = cOffset + i * 2;
					if (abs(data.data[r+ c * data.rows]) > max)
					max = abs(data.data[r+ c * data.rows]);

					// coeef BottomLeft
					r = rOffset + j * 2 + 1;
					if (abs(data.data[r+ c * data.rows]) > max)
					max = abs(data.data[r+ c * data.rows]);

					// coeef BottomRight
					c = cOffset + i * 2 + 1;
					if (abs(data.data[r+ c * data.rows]) > max)
					max = abs(data.data[r+ c * data.rows]);

					// coeef TopRight
					r = rOffset + j * 2;
					if (abs(data.data[r+ c * data.rows]) > max)
					max = abs(data.data[r+ c * data.rows]);

					if (level == levels)
					{
					}
					else
					{
						// Add max abs coeff to next level abs coeff
						r = rOffset2 + j;
						c = cOffset2 + i;
						data.data[r+ c * data.rows] = abs(data.data[r+ c * data.rows]) + max;
					}
				}
			}


			//                #region Determine root saliency values

			float sum;
			if (level == levels) {
				for (int i = 0; i < m; i++) // cols
				{
					for (int j = 0; j < n; j++) // rows
					{
						max = -HUGE;
						sum = 0;

						//#region H Block
						rOffset = 0;
						cOffset = m;
						r = rOffset + j;
						c = cOffset + i;
						if (data.data[r + c * data.rows] > max)
							max = data.data[r + c * data.rows];
						sum += data.data[r + c * data.rows];
						//#endregion

						//#region D Block
						rOffset = n;
						cOffset = m;
						r = rOffset + j;
						c = cOffset + i;
						if (data.data[r + c * data.rows] > max)
							max = data.data[r + c * data.rows];
						sum += data.data[r + c * data.rows];
						//#endregion

						//#region V Block
						rOffset = n;
						cOffset = 0;
						r = rOffset + j;
						c = cOffset + i;
						if (data.data[r + c * data.rows] > max)
							max = data.data[r + c * data.rows];
						sum += data.data[r + c * data.rows];
						//#endregion

						data.data[j + i * data.rows] = sum;
					}
				}
			}

			//#endregion
		}

		//#endregion
	}

	Matrix Spy() {
		return data;
	}

	SalientPoint** GetSaliencyPoints(int vBlocks, int hBlocks, float threshold,
			bool chessBoard, int &numPoints) {
		SalientPoint** points = new SalientPoint*[vBlocks * hBlocks
				* MAX_POINTS_PER_BLOCK];

		int n = data.rows / (int) pow(2.0, levels);
		int m = data.cols / (int) pow(2.0, levels);

		SalientPoint** blockPoints = new SalientPoint*[n * m];

		int pointIndex = 0;
		int blockPointIndex = 0;

		int cOffset;
		int rOffset;

		float max = -HUGE;

		for (int c = 0; c < m; c++) {
			for (int r = 0; r < n; r++) {
				if (max < data.data[r + c * data.rows])
					max = data.data[r + c * data.rows];
			}
		}

		//printf("\tMax salient root = %f\n",max);

		threshold *= max;

		for (int h = 0; h < hBlocks; h++) {
			for (int v = 0; v < vBlocks; v++) {
				if (chessBoard)
					if ((((h + v) >> 1) << 1) == (h + v))
						continue;
				rOffset = v * n / vBlocks;
				cOffset = h * n / hBlocks;
				for (int c = cOffset; c < cOffset + m / hBlocks; c++) {
					for (int r = rOffset; r < rOffset + n / vBlocks; r++) {
						blockPoints[blockPointIndex++] = new SalientPoint(c, r,
								data.data[r + c * data.rows]);
					}
				}
				SortSaliencyPoints(blockPoints, blockPointIndex);
				int cnt = 0;
				for (int i = blockPointIndex - 1; i >= 0; i--) {
					if (cnt >= MAX_POINTS_PER_BLOCK)
						break;
					SalientPoint* blockPoint = blockPoints[i];
					if ((blockPoint->Saliency > threshold) && NoPointsCloseBy(
							points, blockPoint, m / hBlocks, n / vBlocks,  pointIndex)) {
						points[pointIndex++] = new SalientPoint(blockPoint->X,
								blockPoint->Y, blockPoint->Saliency);
						cnt++;
					}
				}
				for (int i = 0; i < blockPointIndex; i++) {
					delete blockPoints[i];
				}
				blockPointIndex = 0;
			}
		}

		for (int i = 0; i < pointIndex; i++) {
			ResolveSaliencyPoint(points[i]);
		}

		numPoints = pointIndex;
        SortSaliencyPoints(points, numPoints);

		return points;
	}

	void SortSaliencyPoints(SalientPoint** points, int numPoints) {
		Quick_sort(0, numPoints - 1, points);
	}

	/*Function for partitioning the array*/

	int Partition(int left, int right, int pivotIndex, SalientPoint** points) {
		SalientPoint* pivotValue = points[pivotIndex];
		swap(points[pivotIndex], points[right]); // Move pivot to end
		int storeIndex = left;
		for (int i = left; i < right; i++) // left ≤ i < right
			if (points[i]->Saliency > pivotValue->Saliency) {
				swap(points[i], points[storeIndex]);
				storeIndex++;
			}
		swap(points[storeIndex], points[right]); // Move pivot to its final place
		return storeIndex;
	}

	void swap(SalientPoint* &a, SalientPoint* &b) {
		SalientPoint* t;
		t = a;
		a = b;
		b = t;
	}

	void Quick_sort(int low, int hi, SalientPoint** points) {
		int Piv_index = low + (hi - low) / 2;
		if (low < hi) {
			Piv_index = Partition(low, hi, Piv_index, points);
			Quick_sort(low, Piv_index - 1, points);
			Quick_sort(Piv_index + 1, hi, points);
		}
	}

	bool NoPointsCloseBy(SalientPoint** points, SalientPoint* s, int hSpace,
			int vSpace, int numPoints) {
		int cnt = 0;
		for (int i = 0; i < numPoints; i++) {
			SalientPoint* point = points[i];
			if ((abs(point->X - s->X) <= hSpace) && (abs(point->Y - s->Y)
					<= vSpace))
				cnt++;
		}
		return true;//cnt == 0;
	}

	void ResolveSaliencyPoint(SalientPoint *point) {
		int rOffset, rOffset2 = 0, r;
		int cOffset, cOffset2 = 0, c;
		int n = data.rows / (int) pow(2.0, levels);
		int m = data.cols / (int) pow(2.0, levels);

		//printf("%d, %d\t", point->X, point->Y);
		//printf("%d, %d\t", n, m);

		float max = -HUGE;

		rOffset = n;
		cOffset = 0;
		r = rOffset + point->Y;
		c = cOffset + point->X;
		if (data.data[r + c * data.rows] > max) {
			max = data.data[r + c * data.rows];
			rOffset2 = rOffset;
			cOffset2 = cOffset;
		}

		rOffset = n;
		cOffset = m;
		r = rOffset + point->Y;
		c = cOffset + point->X;
		if (data.data[r + c * data.rows] > max) {
			max = data.data[r + c * data.rows];
			rOffset2 = rOffset;
			cOffset2 = cOffset;
		}

		rOffset = 0;
		cOffset = m;
		r = rOffset + point->Y;
		c = cOffset + point->X;
		if (data.data[r + c * data.rows] > max) {
			max = data.data[r + c * data.rows];
			rOffset2 = rOffset;
			cOffset2 = cOffset;
		}

		for (int level = levels; level > 0; level--) {
			if (level == levels) {
				point->Y += rOffset2;
				point->X += cOffset2;
			} else {
				point->Y *= 2;
				point->X *= 2;

				max = -HUGE;

				r = point->Y + 0;
				c = point->X + 0;
				if (data.data[r + c * data.rows] > max) {
					max = data.data[r + c * data.rows];
					rOffset = 0;
					cOffset = 0;
				}

				r = point->Y + 1;
				if (data.data[r + c * data.rows] > max) {
					max = data.data[r + c * data.rows];
					rOffset = 1;
					cOffset = 0;
				}

				c = point->X + 1;
				if (data.data[r + c * data.rows] > max) {
					max = data.data[r + c * data.rows];
					rOffset = 1;
					cOffset = 1;
				}

				r = point->Y + 0;
				if (data.data[r + c * data.rows] > max) {
					max = data.data[r + c * data.rows];
					rOffset = 0;
					cOffset = 1;
				}

				point->Y += rOffset;
				point->X += cOffset;
			}
		}

		point->Y -= rOffset2 * (int) pow(2.0, levels - 1);
		point->X -= cOffset2 * (int) pow(2.0, levels - 1);

		point->Y *= 2;
		point->X *= 2;

		//printf("%d, %d\n", point->X, point->Y);
	}

};

