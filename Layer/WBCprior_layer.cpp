#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/WBCprior_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void WbcpriorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  WbcPriorParameter wbcprior_param = this->layer_param_.wbcprior_param();
  
  kernel_size = kernel_h_ = kernel_w_ = wbcprior_param.kernel_size();
  pad_h_ = pad_w_ = static_cast<int>(floor(static_cast<float>(kernel_size) / 2));
}

template <typename Dtype>
void WbcpriorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  
  pooled_height_ = height_;
  pooled_width_ = width_;

  top[0]->Reshape(bottom[0]->num(), 1, pooled_height_, pooled_width_);
  if (this->layer_param_.wbcprior_param().typeprior() ==
      WbcPriorParameter_PriorMethod_DARK) {
      black_mask_.Reshape(bottom[0]->num(), 1, pooled_height_, pooled_width_);
    } else {
      white_mask_.Reshape(bottom[0]->num(), 1, pooled_height_, pooled_width_);
    }
}

template <typename Dtype>
void WbcpriorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();

  int* mask = NULL;  // suppress warnings about uninitialized variables

  switch (this->layer_param_.wbcprior_param().typeprior()) {
  case WbcPriorParameter_PriorMethod_DARK: // Output the dark channel prior
    mask = black_mask_.mutable_cpu_data();
    caffe_set(top_count, -1, mask);
    caffe_set(top_count, Dtype(FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          int hstart = ph - pad_h_;
          int wstart = pw - pad_w_;
          int hend = min(hstart + kernel_h_, height_);
          int wend = min(wstart + kernel_w_, width_);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);            
          const int pool_index = ph * pooled_width_ + pw;
          for (int c = 0; c < channels_; ++c) {
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = c * height_ * width_ + h * width_ + w;
                if (bottom_data[index] < top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  mask[pool_index] = index;
                }
              }
            }
          }
        }
      }
      // compute offset
      bottom_data += bottom[0]->offset(1);
      top_data += top[0]->offset(1);
      mask += top[0]->offset(1);
    }
    break;
  case WbcPriorParameter_PriorMethod_WHITE: // Output the white channel prior
    mask = white_mask_.mutable_cpu_data();
    caffe_set(top_count, -1, mask);
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          int hstart = ph - pad_h_;
          int wstart = pw - pad_w_;
          int hend = min(hstart + kernel_h_, height_);
          int wend = min(wstart + kernel_w_, width_);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);            
          const int pool_index = ph * pooled_width_ + pw;
          for (int c = 0; c < channels_; ++c) {
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = c * height_ * width_ + h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  mask[pool_index] = index;
                }
              }
            }
          }
        }
      }
      // compute offset
      bottom_data += bottom[0]->offset(1);
      top_data += top[0]->offset(1);
      mask += top[0]->offset(1);
    }
    break;
  default:
    LOG(FATAL) << "Unknown prior method.";
  }
}

template <typename Dtype>
void WbcpriorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  const int* mask = NULL;  // suppress warnings about uninitialized variables
  
  switch (this->layer_param_.wbcprior_param().typeprior()) {
  case WbcPriorParameter_PriorMethod_DARK:
    mask = black_mask_.cpu_data();
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          const int index = ph * pooled_width_ + pw;
          const int bottom_index = mask[index];
          bottom_diff[bottom_index] += top_diff[index];
        }
      }
      bottom_diff += bottom[0]->offset(1);
      top_diff += top[0]->offset(1);
      mask += top[0]->offset(1);
    }
    break;
  case WbcPriorParameter_PriorMethod_WHITE:
    mask = white_mask_.cpu_data();
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          const int index = ph * pooled_width_ + pw;
          const int bottom_index = mask[index];
          bottom_diff[bottom_index] += top_diff[index];
        }
      }
      bottom_diff += bottom[0]->offset(1);
      top_diff += top[0]->offset(1);
      mask += top[0]->offset(1);
    }
    break; 
  default:
    LOG(FATAL) << "Unknown prior method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(WbcpriorLayer);
#endif
    INSTANTIATE_CLASS(WbcpriorLayer);
    REGISTER_LAYER_CLASS(Wbcprior);

}  // namespace caffe
