#include "BaseLifter.h"

class HaarLifter: public BaseLifter {

	virtual void Predict(Matrix &data, int N, Lifting::Direction direction) {
		int half = N >> 1;

		for (int i = 0; i < half; i++) {
			float predictVal = data.GetVectorElement(i);
			int j = i + half;

			if (direction == Lifting::forward) {
				data.SetVectorElement(j, data.GetVectorElement(j) - predictVal);
			} else if (direction == Lifting::reverse) {
				data.SetVectorElement(j, data.GetVectorElement(j) + predictVal);
			} else {

			}
		}

	}

	virtual void Update(Matrix &data, int N, Lifting::Direction direction) {
		int half = N >> 1;

		for (int i = 0; i < half; i++) {
			int j = i + half;
			float updateVal = data.GetVectorElement(j) / 2.0;

			if (direction == Lifting::forward) {
				data.SetVectorElement(i, data.GetVectorElement(i) + updateVal);
			} else if (direction == Lifting::reverse) {
				data.SetVectorElement(i, data.GetVectorElement(i) - updateVal);
			} else {
			}
		}
	}
};
