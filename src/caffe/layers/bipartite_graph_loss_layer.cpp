#include <vector>

#include "caffe/layers/bipartite_graph_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

template <typename Dtype>
void BipartiteGraphLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
 		const vector<Blob<Dtype>*>& top){

	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
	 << "The data and label should have the same number.";
	CHECK_EQ(bottom[1]->num(), bottom[2]->num())
	 << "The data and label should have the same number.";
	CHECK_EQ(bottom[2]->num(), bottom[3]->num())
	 << "The data and label should have the same number.";
	// fine-gained predictions and labels channels must equal
	CHECK_EQ(bottom[0]->channel(), bottom[2]->channel());
	// coarse-class predictions and labels channels must equal
	CHECK_EQ(bottom[1]->channel(), bottom[3]->channel());

	CHECK_EQ(bottom[0]->height(), 1);
	CHECK_EQ(bottom[0]->width(), 1);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);
	CHECK_EQ(bottom[2]->height(), 1);
	CHECK_EQ(bottom[2]->width(), 1);
	CHECK_EQ(bottom[3]->height(), 1);
	CHECK_EQ(bottom[3]->width(), 1);

	has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
	if(has_ignore_label_){
		ignore_label_ = this->layer_param_.loss_param().ignore_label();
	}

	// LossLayers have a non-zero (1) loss by default.
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
}

template <typename Dtype>
void BipartiteGraphLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
 		const vector<Blob<Dtype>*>& top){

	softmax_axis_ = 
		bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
	top[0]->ReshapeLike(*bottom[0]);
	vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
    // let output has the same size as input
    sum_multiplier_.Reshape(mult_dims);
    Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
    // inialization
    caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  	// batch nums 
  	outer_num_ = bottom[0]->count(0, softmax_axis_);
  	// fine-gained class nums
  	inner_num_fine = bottom[0]->count(softmax_axis_+1);
  	// coarse-class nums
  	inner_num_coarse = bottom[1]->count(softmax_axis_+1);
  	CHECK_EQ(outer_num_ * inner_num_fine, bottom[2]->count())
  		<< "Number of labels must match number of predictions; "
        << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
        << "label count (number of labels) must be N*H*W, "
        << "with integer values in {0, 1, ..., C-1}.";
    
    // inialize output blobs
    vector<int> loss_shape(0);
    top[0]->Reshape(loss_shape);
    if(top.size() >= 2){
    	top[1]->ReshapeLike(*bottom[0]);
    }

  	// get the shape from fine-gained and coarse-class predictions
  	vector<int> scale_fine_dims = bottom[0]->shape();
  	vector<int> scale_coarse_dims = bottom[1]->shape();
  	scale_fine_dims[softmax_axis_] = 1;
  	scale_coarse_dims[softmax_axis_] = 1;
  	scale_fine.Reshape(scale_fine_dims);
  	scale_coarse.Reshape(scale_coarse_dims);
}

template <typename Dtype>
void BipartiteGraphLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
 		const vector<Blob<Dtype>*>& top){

}

}	 // namespace caffe