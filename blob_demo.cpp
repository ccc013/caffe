#include<vector>
#include<iostream>
#include<caffe/blob.hpp>

using namespace caffe;
using namespace std;

int main(void) {

	Blob<float> a;
	cout << "Size : " << a.shape_string() << endl;
	a.Reshape(1, 2, 3, 4);
	cout << "Size : " << a.shape_string() << "\n";

	return 0;
}
