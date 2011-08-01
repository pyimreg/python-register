class Matrix {
private:
	int _row;
	int _col;

public:
	int rows;
	int cols;
	float* data;

	Matrix();
	Matrix(int noRows, int noCols);
	Matrix(float* data, int noRows, int noCols);

	void SelectRow(int row);
	void SelectCol(int col);

	int GetSelecedVectorLength();
	float GetVectorElement(int index);
	void SetVectorElement(int index, float value);
};

Matrix::Matrix() {
	rows = 0;
	cols = 0;
	data = NULL;
}

Matrix::Matrix(int noRows, int noCols) {
	rows = noRows;
	cols = noCols;
	this->data = new float[rows * cols];
}

Matrix::Matrix(float* data, int noRows, int noCols) {
	rows = noRows;
	cols = noCols;
	this->data = data;
}

void Matrix::SelectRow(int row) {
	_row = row;
	_col = -1;
}

void Matrix::SelectCol(int col) {
	_col = col;
	_row = -1;
}

int Matrix::GetSelecedVectorLength() {
	if (_row != -1)
		return cols;
	else if (_col != -1)
		return rows;
	else
		return -1;
}

float Matrix::GetVectorElement(int index) {
	if (_row != -1)
		return data[_row + index * rows];
	else if (_col != -1)
		return data[index + _col * rows];
	else
		return -1;
}

void Matrix::SetVectorElement(int index, float value) {
	if (_row != -1)
		data[_row + index * rows] = value;
	else if (_col != -1)
		data[index + _col * rows] = value;
}
