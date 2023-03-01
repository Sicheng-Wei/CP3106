#include <opencv2\opencv.hpp>
#include <iostream>
#include <algorithm>
#include <random>
#include <lbfgs.h>

using namespace std;
using namespace cv;

Mat sign(Mat input) {
	Mat output;
	threshold(input, output, 0, 1, THRESH_BINARY);
	output = output * 2 - 1;
	return output;
}

// Define the transform operator (e.g. Fourier or wavelet transform)
void transform(Mat& input, Mat& output) {
	// Perform the transform using OpenCV's DFT function
	dft(input, output, DFT_SCALE | DFT_COMPLEX_OUTPUT);
}

// Define the inverse transform operator
void inverse_transform(Mat& input, Mat& output) {
	// Perform the inverse transform using OpenCV's IDFT function
	idft(input, output, DFT_SCALE | DFT_REAL_OUTPUT);
}

// Define the objective function for the optimization problem
lbfgsfloatval_t objective(void* instance, const lbfgsfloatval_t* x, lbfgsfloatval_t* g, const int n, const lbfgsfloatval_t step) {
	// Cast the data pointer to the correct type
	Mat* measurement = static_cast<Mat*>(instance);

	// Transform x back to the spatial domain
	Mat signal = Mat(n, 1, CV_64FC1, const_cast<lbfgsfloatval_t*>(x)).clone();
	Mat spatial_signal;
	inverse_transform(signal, spatial_signal);

	// Calculate the difference between the signal and the measurements
	Mat residual;
	absdiff(*measurement, spatial_signal, residual);

	// Calculate the objective function value
	double fval = norm(residual) * norm(residual);
	fval += norm(signal, NORM_L1);

	// Calculate the gradient of the objective function
	Mat gradient = Mat::zeros(n, 1, CV_64FC1);
	Mat residual_grad;
	subtract(spatial_signal, *measurement, residual_grad);
	Mat transform_grad;
	transform(signal, transform_grad);
	gradient = residual_grad.mul(transform_grad);
	gradient += Mat::ones(n, 1, CV_64FC1).mul(sign(signal));
	gradient = gradient.reshape(0, 1);

	// Copy the gradient to the output array
	copy(gradient.begin<double>(), gradient.end<double>(), g);

	return fval;
}



int main()
{
	Mat img;
	// img = imread("./monalisa.png", IMREAD_GRAYSCALE);
	img = imread("./pixcat.png", IMREAD_GRAYSCALE);
	int nx = img.cols;
	int ny = img.rows;
	imshow("Origin Grayscale", img);

	// image flatten
	Mat img_flat = img.reshape(1, 1);
	int length = img_flat.size[1];

	// image random params
	int rand_ampl = 16;
	vector<int> rnd_base;
	for (int i = 0; i < length; i++) rnd_base.push_back(i);
	shuffle(rnd_base.begin(), rnd_base.end(), std::default_random_engine(42));
	int flt_len = length / rand_ampl;

	// image random filter
	vector<int> flt_base(rnd_base.begin(), rnd_base.begin() + flt_len);
	sort(flt_base.begin(), flt_base.end());

	Mat img_fltr = img_flat.clone();
	vector<int>::iterator ptr = flt_base.begin();
	for (int i = 0; i < length; i++)
	{
		if (i == *ptr) { ptr += 1; continue; }
		else img_fltr.data[i] = 255;
	}
	imshow("Filter Grayscale", img_fltr.reshape(0, ny));

	// optimization using L - BFGS algorithm
		const int n = length;
	lbfgsfloatval_t* x = lbfgs_malloc(n);
	for (int i = 0; i < n; i++) x[i] = 0.0;

	lbfgs_parameter_t param;
	lbfgs_parameter_init(&param);
	param.max_iterations = 1000;

	Mat filtered_image;
	img_fltr.reshape(0, ny).convertTo(filtered_image, CV_64FC1);
	transform(filtered_image, filtered_image);

	void* data = &filtered_image;
	lbfgs(n, x, NULL, objective, NULL, data, &param);

	Mat restored_image;
	Mat signal = Mat(n, 1, CV_64FC1, x).clone();
	Mat spatial_signal;
	inverse_transform(signal, spatial_signal);
	spatial_signal.convertTo(restored_image, CV_8UC1);
	imshow("Restored Grayscale", restored_image);

	// release memory
	lbfgs_free(x);

	waitKey(0);
	return 0;
}
