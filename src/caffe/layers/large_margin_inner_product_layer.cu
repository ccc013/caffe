#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/large_margin_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void copy_label_score(const int M, const int N, const Dtype *label_data,
	const Dtype *top_data, Dtype *wx_data) {
	CUDA_KERNEL_LOOP(index, M) {
		wx_data[index] = top_data[index * N + static_cast<int>(label_data[index])];
	}
}

template <typename Dtype>
__global__ void cal_cos_mt(const int count, const unsigned int margin, const int *C_M_N, const Dtype *cos_t_data,
	Dtype *cos_mt_data) {
	CUDA_KERNEL_LOOP(index, count) {
		Dtype cos_t = cos_t_data[index];
		Dtype sin_t_2 = 1 - cos_t * cos_t;
		Dtype cos_mt = 0.;
		int flag = -1;
		for(int n = 0; n <= (margin / 2); ++n) {
			flag *= -1;
			cos_mt += flag * C_M_N[2 * n] * powf(cos_t, (margin - 2 * n)) * powf(sin_t_2, n);
		}
		cos_mt_data[index] = cos_mt;
	}
}

template <typename Dtype>
__global__ void LMForward(
  const int M, const int N, const float lambda,
  const Dtype *label_data, const Dtype *cos_mt_data, const int *k_data,
  const Dtype *abs_w_data, const Dtype *abs_x_data, Dtype *top_data) {
 
  CUDA_KERNEL_LOOP(index, M) {
    Dtype cos_mt = cos_mt_data[index];
    int k = k_data[index];
    int label = static_cast<int>(label_data[index]);
    Dtype abs_w = abs_w_data[index];
    Dtype abs_x = abs_x_data[index];
    top_data[N * index + label] =  (lambda * top_data[N * index + label] + 
    	abs_w * abs_x * ( powf(-1, k) * cos_mt - 2 * k )) / (1 + lambda);
  }
}

template <typename Dtype>
void LargeMarginInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* label_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
 
  // 普通fc层的计算
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
  }
 
  const Dtype* label_cpu_data = bottom[1]->cpu_data();
 
  // w * x
  // 直接从前馈的结果中复制
  Dtype *wx_data = this->wx_.mutable_gpu_data();
  copy_label_score<Dtype><<<CAFFE_GET_BLOCKS(M_), CAFFE_CUDA_NUM_THREADS>>>(M_, N_, label_data, top_data, wx_data);
 
  // w * w
  Dtype *abs_w_data = this->abs_w_.mutable_cpu_data();
  for (int m = 0; m < M_; ++ m) {
    abs_w_data[m] = caffe_cpu_dot<Dtype>(
      K_,
      this->blobs_[0]->cpu_data() + static_cast<int>(label_cpu_data[m]) * K_,
      this->blobs_[0]->cpu_data() + static_cast<int>(label_cpu_data[m]) * K_
      );
  }
   // x * x
  Dtype *abs_x_data = this->abs_x_.mutable_cpu_data();
  for (int m = 0; m < M_; ++ m) {
    abs_x_data[m] = caffe_cpu_dot<Dtype>(
      K_, 
      bottom[0]->cpu_data() + m * K_,
      bottom[0]->cpu_data() + m * K_
      );
  }
 
  // abs_w, abs_x
  caffe_gpu_powx<Dtype>(M_, this->abs_w_.mutable_gpu_data(), 0.5, this->abs_w_.mutable_gpu_data());
  caffe_gpu_powx<Dtype>(M_, this->abs_x_.mutable_gpu_data(), 0.5, this->abs_x_.mutable_gpu_data());
 
  // cos_t = wx / (|x| * |w|)
  Dtype *cos_t_data = this->cos_t_.mutable_gpu_data();
  caffe_gpu_div<Dtype>(M_, wx_data, this->abs_x_.gpu_data(), cos_t_data);
  caffe_gpu_div<Dtype>(M_, cos_t_data, this->abs_w_.gpu_data(), cos_t_data);
 
  // cos(mt)
  cal_cos_mt<Dtype><<<CAFFE_GET_BLOCKS(M_), CAFFE_CUDA_NUM_THREADS>>>(
    M_, this->margin, 
    this->C_M_N_.gpu_data(), 
    this->cos_t_.gpu_data(),
    this->cos_mt_.mutable_gpu_data()
    );
  
  // k
  int *k_cpu_data = this->k_.mutable_cpu_data();
  const Dtype *cos_t_cpu_data = this->cos_t_.cpu_data();
  for (int m = 0; m < M_; ++ m) {
    for (int _k = 0; _k < this->cos_theta_bound_.count(); ++ _k) {
      if (this->cos_theta_bound_.cpu_data()[_k] < cos_t_cpu_data[m]) {
        k_cpu_data[m] = _k - 1;
        break;
      }
    }
  }
 
  // y
  LMForward<Dtype><<<CAFFE_GET_BLOCKS(M_), CAFFE_CUDA_NUM_THREADS>>>(
    M_, N_, this->lambda,
    label_data, this->cos_mt_.gpu_data(), this->k_.gpu_data(),
    this->abs_w_.gpu_data(), this->abs_x_.gpu_data(), top[0]->mutable_gpu_data());
}

template <typename Dtype>
void LargeMarginInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(LargeMarginInnerProductLayer);

} // namespace caffe