#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/WBCprior_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DCPForward(const int nthreads, const int channels,
    const Dtype* const bottom_data, const int height, const int width, 
    const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, Dtype* const top_data, int* mask) {
  
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int n = index / pooled_width / pooled_height;  
    int hstart = ph - pad_h;
    int wstart = pw - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype minval = FLT_MAX;
    int minidx = -1;
    const Dtype* const bottom_slice = bottom_data + (n * channels * height * width);
    for (int c = 0; c < channels; ++c) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          if (bottom_slice[c * height * width + h * width + w] < minval) {
            minidx = c * height * width + h * width + w;
            minval = bottom_slice[minidx];
          }
        }
      }
    }
    top_data[index] = minval;
    mask[index] = minidx;
  }
}

template <typename Dtype>
__global__ void WCPForward(const int nthreads, const int channels,
    const Dtype* const bottom_data, const int height, const int width, 
    const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, Dtype* const top_data, int* mask) {
  
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int n = index / pooled_width / pooled_height;  
    int hstart = ph - pad_h;
    int wstart = pw - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice = bottom_data + (n * channels * height * width);
    for (int c = 0; c < channels; ++c) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          if (bottom_slice[c * height * width + h * width + w] > maxval) {
            maxidx = c * height * width + h * width + w;
            maxval = bottom_slice[maxidx];
          }
        }
      }
    }
    top_data[index] = maxval;
    mask[index] = maxidx;
  }
}


template <typename Dtype>
void WbcpriorLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();

  int* mask = NULL;

  switch (this->layer_param_.wbcprior_param().typeprior()) {
  case WbcPriorParameter_PriorMethod_DARK:
    mask = black_mask_.mutable_gpu_data();
    DCPForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, channels_, bottom_data, height_, width_, pooled_height_, pooled_width_, 
        kernel_h_, kernel_w_, pad_h_, pad_w_, top_data, mask);
    break;
  case WbcPriorParameter_PriorMethod_WHITE:
    mask = white_mask_.mutable_gpu_data();
    WCPForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, channels_, bottom_data, height_, width_, pooled_height_, pooled_width_, 
        kernel_h_, kernel_w_, pad_h_, pad_w_, top_data, mask);
    break;
  default:
    LOG(FATAL) << "Unknown prior method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void DCPBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, 
    const int pad_h, const int pad_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) + 1;
    const int phend = min((h + pad_h) + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) + 1;
    const int pwend = min((w + pad_w) + 1, pooled_width);
    
    Dtype gradient = 0;
    const int offset = n * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    const int* const mask_slice = mask + offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (mask_slice[ph * pooled_width + pw] == c * height * width + h * width + w) {
           gradient += top_diff_slice[ph * pooled_width + pw];
         }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void WCPBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, 
    const int pad_h, const int pad_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) + 1;
    const int phend = min((h + pad_h) + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) + 1;
    const int pwend = min((w + pad_w) + 1, pooled_width);
    
    Dtype gradient = 0;
    const int offset = n * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    const int* const mask_slice = mask + offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (mask_slice[ph * pooled_width + pw] == c * height * width + h * width + w) {
           gradient += top_diff_slice[ph * pooled_width + pw];
         }
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void WbcpriorLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);

  const int* mask = NULL;

  switch (this->layer_param_.wbcprior_param().typeprior()) {
  case WbcPriorParameter_PriorMethod_DARK:
    mask = black_mask_.gpu_data();
    DCPBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, channels_, height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, pad_h_, pad_w_, bottom_diff);
    break;
  case WbcPriorParameter_PriorMethod_WHITE:
    mask = white_mask_.gpu_data();
    WCPBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, channels_, height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, pad_h_, pad_w_, bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown prior method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(WbcpriorLayer);


}  // namespace caffe
