#ifndef CAFFE_BIPARTITE_GRAPH_LOSS_HPP_
#define CAFFE_BIPARTITE_GRAPH_LOSS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe{

/*
* @brief compute the bipartite-graph loss, take 4 Blobs as input,usually (1) fine-gained labels predictions,
*		 (2) coarse_classes/attributes labels predictions, (3) fine-gained labels and (4) coarse_classes/attributes labels
* 
*/

template <typename Dtype>
class BipartiteGraphLossLayer : public LossLayer<Dtype>{
 public:
 	explicit BipartiteGraphLossLayer(const LayerParameter& param)
 		: LossLayer<Dtype>(param) {}
 	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
 		const vector<Blob<Dtype>*>& top);
 	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
 		const vector<Blob<Dtype>*>& top);

 	virtual inline const char* type() const{ return "BipartiteGraphLoss";}
 	virtual inline int ExactNumBottomBlobs() const { return 4;}
 	virtual inline int ExactNumTopBlobs() const { return -1;}
 	// cannot backpropagate to the fine-gained labels and coarse_classes/attributes labels
 	virtual inline bool AllowForceBackward(const int bottom_index) const{
 		return bottom_index != 2 && bottom_index != 3;
 	}

 protected:
 	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
 		const vector<Blob<Dtype>*>& top);
 	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


    

};


}	// namespace caffe

#endif   // CAFFE_BIPARTITE_GRAPH_LOSS_HPP_
