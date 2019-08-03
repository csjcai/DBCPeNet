#ifndef CAFFE_WBCPRIOR_LAYER_HPP_
#define CAFFE_WBCPRIOR_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Extract white and black channel of the input image.
   white channel (1 channel): the maximum value of the patch (3 channels)
   black channel (1 channel): the minimum value of the patch (3 channels)  
 */
template <typename Dtype>
class WbcpriorLayer : public Layer<Dtype> {
 public:
  explicit WbcpriorLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Wbcprior"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int kernel_size, kernel_h_, kernel_w_;
  int pad_h_, pad_w_;
  int channels_, height_, width_;
  int pooled_height_, pooled_width_;

  Blob<int> black_mask_;
  Blob<int> white_mask_;
};

}  // namespace caffe

#endif  // CAFFE_WBCPRIOR_LAYER_HPP_
