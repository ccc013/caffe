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
	// CHECK_EQ(bottom[1]->height(), 1);
	// CHECK_EQ(bottom[1]->width(), 1);
	CHECK_EQ(bottom[2]->height(), 1);
	CHECK_EQ(bottom[2]->width(), 1);
	// CHECK_EQ(bottom[3]->height(), 1);
	// CHECK_EQ(bottom[3]->width(), 1);

	prob_f_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
	prob_f_temp_.Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
	prob_c_.Reshape(bottom[1]->num(), bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());

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
	vector<int> mult_dims_coarse(1, bottom[1]->shape(softmax_axis_));
    // let output has the same size as input
    sum_multiplier_fine_.Reshape(mult_dims);
    sum_multiplier_coarse_.Reshape(mult_dims_coarse);
    Dtype* multiplier_data = sum_multiplier_fine_.mutable_cpu_data();
    Dtype* multiplier_data_coarse = sum_multiplier_coarse_.mutable_cpu_data();
    // inialization
    caffe_set(sum_multiplier_fine_.count(), Dtype(1), multiplier_data);
    caffe_set(sum_multiplier_coarse_.count(), Dtype(1), multiplier_data_coarse);

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
  	scale_fine_.Reshape(scale_fine_dims);
  	scale_coarse_.Reshape(scale_coarse_dims);
}

template <typename Dtype>
void BipartiteGraphLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
 		const vector<Blob<Dtype>*>& top){

	const Dtype* bottom_data_fine = bottom[0]->cpu_data();
	const Dtype* bottom_data_coarse = bottom[1]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	Dtype* fine_data = prob_f_[0]->mutable_cpu_data();
	Dtype* coarse_data = prob_f_temp_[0]->mutable_cpu_data();
	Dtype* prob_c_data = prob_c_[0]->mutable_cpu_data();

	Dtype* scale_fine_data = scale_fine_.mutable_cpu_data();
	Dtype* scale_coarse_data = scale_coarse_.mutable_cpu_data();

	int channels_fine = bottom[0]->shape(softmax_axis_);
	int channels_coarse = bottom[1]->shape(softmax_axis_);
	int dim_fine = bottom[0]->count() / outer_num_;
	int dim_coarse = bottom[1]->count() / outer_num_;

	// label value
	const Dtype* fine_label = bottom[2]->cpu_data();
	const Dtype* coarse_label = bottom[3]->cpu_data();

	caffe_copy(bottom[0]->count(), bottom_data_fine, fine_data);
	caffe_copy(bottom[1]->count(), bottom_data_coarse, coarse_data);

	Dtype loss_fine = 0, loss_coarse = 0;

	// We need to subtract the max to avoid numerical issues, compute the exp,
    // and then normalize.	
	for (int i = 0; i < outer_num_; i++) {
		caffe_copy(inner_num_fine, bottom_data_fine + i * dim_fine, scale_fine_data);
		caffe_copy(inner_num_coarse, bottom_data_coarse + i * dim_coarse, scale_coarse_data);
		
		// firstly, compute the coarse-class
    	for(int j = 0; j < channels_coarse; j++){
    		for(int k = 0; k < inner_num_coarse; k++){
    			scale_coarse_data[k] = std::max(scale_coarse_data[k],
    				bottom_data_coarse[i * dim_coarse + j * inner_num_coarse + k]);
    			// const int label_value = static_cast<int>(coarse_label[i * channels_coarse + j * inner_num_coarse + k]);
    		}
    	}
    	// subtraction, f_j = f_j - max(f_j)
    	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_coarse, inner_num_coarse,
    		1, -1., sum_multiplier_coarse_.cpu_data(), scale_coarse_data, 1., coarse_data);
    	// exponentiation, exp(f_j)
    	caffe_exp<Dtype>(dim_coarse, coarse_data, coarse_data);
    	// compute temp = \prod_{l=1}^m f_l
    	Dtype temp = 1;
    	for (int j = 0; j < channels_coarse; j++){
    		temp *= coarse_data[i * dim_coarse + j];
    	}

    	// compute f_j = g_cj * exp(f_j)
    	caffe_mul<Dtype>(dim_coarse, coarse_label, coarse_data, coarse_data);
    	// compute temp_ = \prod_{j=0}^m f_j
    	Dtype temp_ = 1;
    	for(int j = 0; j < channels_coarse; j++){
    		temp_ *= coarse_data[i * dim_coarse + j];
    	}
    	
    	// secondly compute softmax and final loss
    	for(int j = 0; j < channels_fine; j++){
    		for(int k = 0; k < inner_num_fine; k++){
    			scale_fine_data[k] = std::max(scale_fine_data[k],
    				bottom_data_fine[i * dim_fine + j * inner_num_fine + k]);
    		}
    	}   	

    	// subtraction, f_i = f_i - max(f_i)
    	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_fine, inner_num_fine,
    		1, -1., sum_multiplier_fine_.cpu_data(), scale_fine_data, 1., fine_data);
    	
    	// exponentiation, exp(f_i)
    	caffe_exp<Dtype>(dim_fine, fine_data, fine_data);
    	// compute temp_ * exp(f_i)
    	caffe_scal<Dtype>(dim_fine, temp_, fine_data);


    	// sum after exp, z = 1 + exp(f_i) * temp_
    	caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_fine, inner_num_fine, temp_,
    		fine_data, sum_multiplier_fine_.cpu_data(), 0., scale_fine_data);


    	// division, 
    	for(int j=0; j < channels_fine; j++){
    		caffe_div(inner_num_fine, fine_data, scale_fine_data, fine_data);
    		fine_data += inner_num_fine;
    	}

	}

	// compute loss


	
    
}

template <typename Dtype>
void BipartiteGraphLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

}

#ifdef CPU_ONLY
STUB_GPU(BipartiteGraphLossLayer);
#endif

INSTANTIATE_CLASS(BipartiteGraphLossLayer);
REGISTER_LAYER_CLASS(BipartiteGraphLoss);


}	 // namespace caffe