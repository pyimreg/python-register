#include "Matrix.h"
#include <stdlib.h>
#include <math.h>

namespace Lifting {
enum Direction {
	forward, reverse
};
}

class BaseLifter {

private:
	void Split(Matrix &data, int N);
	void Merge(Matrix &data, int N);

protected:
	virtual void Predict(Matrix &data, int N, Lifting::Direction direction);
	virtual void Update(Matrix &data, int N, Lifting::Direction direction);

public:
	virtual void ForwardTrans(Matrix &data, int level);
	virtual void InverseTrans(Matrix &data, int level);

};

void BaseLifter::Split(Matrix &data, int N) {

	float* tmp = new float[N / 2];

	for (int i = 1; i < N; i++) {
		if (i >> 1 << 1 != i)
			// Odd
			tmp[i >> 1] = data.GetVectorElement(i);
		else
			// Even
			data.SetVectorElement(i >> 1, data.GetVectorElement(i));
	}

	for (int i = 0; i < N / 2; i++) {
		data.SetVectorElement((N >> 1) + i, tmp[i]);
	}

}

void BaseLifter::Merge(Matrix &data, int N) {
	// Not implemented
}

void BaseLifter::Predict(Matrix &data, int N, Lifting::Direction direction) {
}

void BaseLifter::Update(Matrix &data, int N, Lifting::Direction direction) {
}

void BaseLifter::ForwardTrans(Matrix &data, int level) {
	int N = data.GetSelecedVectorLength();

	if (((N >> level) << level) != N)
		return;

	int n = N / (int) pow(2.0, level - 1);
	Split(data, n);
	Predict(data, n, Lifting::forward);
	Update(data, n, Lifting::forward);
}

void BaseLifter::InverseTrans(Matrix &data, int level) {
	int N = data.GetSelecedVectorLength();

	if (((N >> level) << level) != N)
		return;

	int n = N / (int) pow(2.0, level - 1);
	Update(data, n, Lifting::reverse);
	Predict(data, n, Lifting::reverse);
	Merge(data, n);
}
