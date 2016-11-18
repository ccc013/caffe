#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/filler.hpp"
#include "caffe/layers/bipartite_graph_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

template <typename Dtype>
__global__ void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_channel_mul(const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* channel_dot) {
	CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot *= (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}

template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}

template <typename Dtype>
void BipartiteGraphLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
 		const vector<Blob<Dtype>*>& top) {
	
	const Dtype* bottom_data_fine = bottom[0]->gpu_data();
	const Dtype* bottom_data_coarse = bottom[1]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();

	Dtype* scale_fine_data = scale_fine_.mutable_gpu_data();
	Dtype* scale_coarse_data = scale_coarse_.mutable_gpu_data();
	Dtype* fine_data = prob_f_.mutable_gpu_data();
	Dtype* coarse_data = prob_f_temp_.mutable_gpu_data();
	Dtype* prob_c_data = prob_c_.mutable_gpu_data();

	int count = bottom[0]->count();
	int count_coarse = bottom[1]->count();
    int channels = top[0]->shape(softmax_axis_);
    int channels_coarse = bottom[1]->shape(softmax_axis_);
    // coarse-class label
    const Dtype* coarse_label = bottom[3]->gpu_data();

    LOG(FATAL) << "count = " << count << ", channels = " << channels
    		   << ", count_coarse = " << count_coarse << ", channels_coarse = " << channels_coarse;
    caffe_copy(count, bottom_data_fine, fine_data);
    caffe_copy(count_coarse, bottom_data_coarse, coarse_data);
	
	// We need to subtract the max to avoid numerical issues, compute the exp,
    // and then normalize.
    // compute max
    kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_fine),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_fine, fine_data,
      scale_fine_data);
    kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_coarse),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels_coarse, inner_num_coarse, coarse_data,
      scale_coarse_data);

    // subtract
    kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_fine,
      scale_fine_data, fine_data);
    kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count_coarse),
      CAFFE_CUDA_NUM_THREADS>>>(count_coarse, outer_num_, channels_coarse, inner_num_coarse,
      scale_coarse_data, coarse_data);

    // exponentiation
    kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, fine_data, fine_data);
    kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(count_coarse), CAFFE_CUDA_NUM_THREADS>>>(
      count_coarse, coarse_data, coarse_data);
	
	// compute \prod_{j=0}^m f_j = g_cj * exp(f_j), m is nums of coarse classes
	kernel_channel_mul<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_coarse),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels_coarse, inner_num_coarse,
      coarse_label, coarse_data, coarse_data);

    // sum after exp
    

}

template <typename Dtype>
void BipartiteGraphLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
}


INSTANTIATE_LAYER_GPU_FUNCS(BipartiteGraphLossLayer);	
}