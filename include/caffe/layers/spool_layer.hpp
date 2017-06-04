/*
 * spp_layer.h
 *
 *  Created on: 21 Dec, 2014
 *      Author: june
 */

#ifndef INCLUDE_LAYERS_SPOOL_LAYER_HPP_
#define INCLUDE_LAYERS_SPOOL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class SpoolLayer : public Layer<Dtype> {
 public:
  explicit SpoolLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Spool"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // virtual void Forward(const std::vector<Blob*> &bottom,
  //                      std::vector<Blob *> *top);
  // virtual void Init(const caffe::LayerParameter &layer_param);
  // virtual void Reshape(const std::vector<Blob*> &bottom,
  //                      std::vector<Blob *> *top);

  std::vector<int> binSize_;
  std::vector<int> poolHeiht_;
  std::vector<int> poolWidth_;
  std::vector<int> strideX_;
  std::vector<int> strideY_;
  Blob<int> max_idx_;
};

} /* namespace caffe */

#endif /* INCLUDE_LAYERS_SPOOL_LAYER_HPP_ */
