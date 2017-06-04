/*
 * spp_layer.cpp
 *
 *  Created on: Oct 28, 2014
 *      Author: june
 */
#include <algorithm>
#include <vector>
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
// #include "caffe/vision_layers.hpp"
#include "caffe/layers/spool_layer.hpp"
// #include "caffe/layers/pooling_layer.hpp"

namespace caffe {

template<typename Dtype>
void SpoolLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	SpoolParameter spoolParam = this->layer_param_.spool_param();
	float height = bottom[0]->height();
	float width = bottom[0]->width();
	for (int i = 0; i < spoolParam.bin_size_size(); ++i) {
		binSize_.push_back(spoolParam.bin_size(i));
		poolHeiht_.push_back(ceil(height / spoolParam.bin_size(i)));
		poolWidth_.push_back(ceil(width / spoolParam.bin_size(i)));
		strideY_.push_back(floor(height / spoolParam.bin_size(i)));
		strideX_.push_back(floor(width / spoolParam.bin_size(i)));
	}
}

template<typename Dtype>
void SpoolLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	SpoolParameter spoolParam = this->layer_param_.spool_param();
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = 0, width = 1;
	for (int i = 0; i < binSize_.size(); ++i) {
		height += binSize_[i] * binSize_[i];
	}
	(top)[0]->Reshape(num, channels, height, width);
	max_idx_.Reshape(num, channels, height, width);
}

template<typename Dtype>
void SpoolLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	SpoolParameter spoolParam = this->layer_param_.spool_param();
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	const Dtype *bottom_data = bottom[0]->cpu_data();
	Dtype *top_data = (top)[0]->mutable_cpu_data();
	caffe_set((top)[0]->count(), Dtype(-FLT_MAX), top_data);
	int *mask = max_idx_.mutable_cpu_data();
	for (int n = 0, top_index = 0; n < num; ++n) {
		for (int c = 0; c < channels; ++c) {
			for (int l = 0; l < spoolParam.bin_size_size(); ++l) {
				for (int poolY_start = 0; poolY_start + poolHeiht_[l] <= height;
						poolY_start += strideY_[l]) {
					for (int poolX_start = 0;
							poolX_start + poolWidth_[l] <= width; poolX_start +=
									strideX_[l]) {
						for (int w = poolX_start;
								w < poolX_start + poolWidth_[l]; ++w) {
							for (int h = poolY_start;
									h < poolY_start + poolHeiht_[l]; ++h) {
								int data_index = n * channels * height * width
										+ c * height * width + h * width + w;
								CHECK_LT(data_index, bottom[0]->count());
								CHECK_LT(top_index, (top)[0]->count());
								if (bottom_data[data_index]
										> top_data[top_index]) {
									top_data[top_index] =
											bottom_data[data_index];
									mask[top_index] = data_index;
								}
							}
						}
						++top_index;
					}
				}
			}
		}
	}
}

template<typename Dtype>
void SpoolLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const int *mask = max_idx_.cpu_data();
	Dtype *diff = (bottom)[0]->mutable_cpu_diff();
	for (int i = 0; i < top[0]->count(); ++i) {
		int diff_index = mask[i];
		diff[diff_index] += top[0]->cpu_diff()[i];
	}
}

INSTANTIATE_CLASS(SpoolLayer);
REGISTER_LAYER_CLASS(Spool);

}

