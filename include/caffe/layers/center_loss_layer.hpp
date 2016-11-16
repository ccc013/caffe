#ifndef CAFFE_CENTER_LOSS_LAYER_HPP_
#define CAFFE_CENTER_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class CenterLossLayer : public LossLayer<Dtype> {
 public:
  /*
  * @brief compute centerLoss, if set withSoftmax = true, also plus SoftmaxWithLoss.
  */
  explicit CenterLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CenterLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return -1; }

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
  
  Blob<Dtype> distance_;
  Blob<Dtype> variation_sum_;

  // // lambda
  // float lambda_;
  // bool withSoftmax_;

  // // SoftmaxWithLossLayer to compute SoftmaxWithLoss
  // shared_ptr<Layer<Dtype> > softmax_loss_layer_;
  // // loss stores the output from the SoftmaxWithLossLayer
  // Blob<Dtype> softmax_loss_;
  // /// bottom vector holder used in call to the underlying SoftmaxWithLossLayer::Forward
  // vector<Blob<Dtype>*> softmax_bottom_vec_;
  // /// top vector holder used in call to the underlying SoftmaxWithLossLayer::Forward
  // vector<Blob<Dtype>*> softmax_top_vec_;
};

}  // namespace caffe

#endif  // CAFFE_CENTER_LOSS_LAYER_HPP_