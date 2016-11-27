#ifndef CAFFE_LARGE_MARGIN_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_LARGE_MARGIN_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class LargeMarginInnerProductLayer : public Layer<Dtype> {
 public:
  explicit LargeMarginInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LargeMarginInnerProduct"; }
  // edited by miao
  // LM_FC层有两个bottom
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  // end edited
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights


  // params for largeMargin
  // 一些常数
  Blob<Dtype> cos_theta_bound_;   // 区间边界的cos值
  Blob<int> k_;                   // 当前角度theta所在的区间的位置
  Blob<int> C_M_N_;               // 组合数
  unsigned int margin;            // margin
  float lambda;                   // lambda

  Blob<Dtype> wx_;                // wjT * xi
  Blob<Dtype> abs_w_;             // ||wj|| 
  Blob<Dtype> abs_x_;             // ||xi||
  Blob<Dtype> cos_t_;             // cos(theta)
  Blob<Dtype> cos_mt_;            // cos(margin * theta)

  Blob<Dtype> dydw_;              // 输出对w的导数
  Blob<Dtype> dydx_;              // 输出对x的导数
  // end added
};

}  // namespace caffe

#endif  // CAFFE_LARGE_MARGIN_INNER_PRODUCT_LAYER_HPP_