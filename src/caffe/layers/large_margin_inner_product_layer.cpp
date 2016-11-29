#include <vector>
#include <cmath>

#include "caffe/filler.hpp"
#include "caffe/layers/large_margin_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define PI 3.14159265

namespace caffe{

int factorial(int n){
	if (0 == n)
		return 1;
	int f = 1;
	while(n){
		f *= n;
		--n;
	}
	return f;
}

template <typename Dtype>
void LargeMarginInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	const int axis = bottom[0]->CanonicalAxisIndex(
		this->layer_param_.large_margin_inner_product_param().axis());
	// large margin param
	std::vector<int> wx_shape(1);
	wx_shape[0] = bottom[0]->shape(0);
	LOG(FATAL) << "wx_shape = " << wx_shape[0] <<"\n";

	this->wx_.Reshape(wx_shape);
	this->abs_w_.Reshape(wx_shape);
	this->abs_x_.Reshape(wx_shape);
	this->k_.Reshape(wx_shape);
	this->cos_t_.Reshape(wx_shape);
	this->cos_mt_.Reshape(wx_shape);

	std::vector<int> cos_theta_bound_shape(1);
	this->margin = static_cast<unsigned int>(this->layer_param_.large_margin_inner_product_param().margin());
	cos_theta_bound_shape[0] = this->margin + 1;
	LOG(FATAL) << "cos_theta_bound_shape = " << cos_theta_bound_shape[0] << "\n";
	this->cos_theta_bound_.Reshape(cos_theta_bound_shape);
	for(int k=0; k <= this->margin; ++k){
		this->cos_theta_bound_.mutable_cpu_data()[k] = std::cos(PI * k / this->margin);
	}
	this->C_M_N_.Reshape(cos_theta_bound_shape);
	for(int n = 0; n <= this->margin; ++n){
		this-> C_M_N_.mutable_cpu_data()[n] = factorial(this->margin) / factorial(this->margin-n) / factorial(n);
	}

	// d size
	std::vector<int> d_shape(2);
	d_shape[0] = bottom[0]->shape(0);
	d_shape[1] = bottom[0]->count(axis);
	this->dydw_.Reshape(d_shape);
	this->dydx_.Reshape(d_shape);
	LOG(FATAL) << "d_shape: " << d_shape[0] << ", " << d_shape[1] << "\n";

	this->lambda = this->layer_param_.large_margin_inner_product_param().lambda();

	transpose_ = false;
	// original innerProducyLayer setup
	const int num_output = this->layer_param_.large_margin_inner_product_param().num_output();
    bias_term_ = this->layer_param_.large_margin_inner_product_param().bias_term();
    N_ = num_output;

    // Dimensions starting from "axis" are "flattened" into a single
    // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
    // and axis == 1, N inner products with dimension CHW are performed.
  	K_ = bottom[0]->count(axis);
  	LOG(FATAL) << "num_output = " << N_ << ", K_ = " << K_ << "\n";

  	// Check if we need to set up the weights
  	if (this->blobs_.size() > 0) {
    	LOG(INFO) << "Skipping parameter initialization";
  	} else {
  		if (bias_term_) {
  			this->blobs_.resize(2);
  		} else {
  			this->blobs_.resize(1);
  		}

  		// Initialize the weights
   	 	vector<int> weight_shape(2);
    	if (transpose_) {
    	  weight_shape[0] = K_;
      	  weight_shape[1] = N_;
    	} else {
    	  weight_shape[0] = N_;
    	  weight_shape[1] = K_;
    	}
    	this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
	    // fill the weights
	    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
	        this->layer_param_.large_margin_inner_product_param().weight_filler()));
	    weight_filler->Fill(this->blobs_[0].get());
	    // If necessary, intiialize and fill the bias term
	    if (bias_term_) {
	      vector<int> bias_shape(1, N_);
	      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
	      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
	          this->layer_param_.large_margin_inner_product_param().bias_filler()));
	      bias_filler->Fill(this->blobs_[1].get());
	    }

  	} // parameter initialization
  	this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void LargeMarginInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// Figure out the dimensions
	const int axis = bottom[0]->CanonicalAxisIndex(
	    this->layer_param_.large_margin_inner_product_param().axis());
	const int new_K = bottom[0]->count(axis);
	CHECK_EQ(K_, new_K)
	    << "Input size incompatible with inner product parameters.";
	// The first "axis" dimensions are independent inner products; the total
	// number of these is M_, the product over these dimensions.
	M_ = bottom[0]->count(0, axis);
	// The top shape will be the bottom shape with the flattened axes dropped,
	// and replaced by a single axis with dimension num_output (N_).
	vector<int> top_shape = bottom[0]->shape();
	top_shape.resize(axis + 1);
	top_shape[axis] = N_;
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void LargeMarginInnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// not implement
}

template <typename Dtype>
void LargeMarginInnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// not implement
}

#ifdef CPU_ONLY
STUB_GPU(LargeMarginInnerProductLayer);
#endif

INSTANTIATE_CLASS(LargeMarginInnerProductLayer);
REGISTER_LAYER_CLASS(LargeMarginInnerProduct);

}