#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// template <typename Dtype>
// Dtype abs(Dtype x) {
//   if (x >= 0) return x;else return -x;
// }

template <typename Dtype>
void RescaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const RescaleParameter& rescale_param = this->layer_param_.rescale_param();
  rf_ = Dtype(rescale_param.rf());
  /*
  CHECK_GE(quantize_point_.size(), 1)<<"at least one quantize point required!";
  std::copy(quantize_param.quantize_point().begin(),
      quantize_param.quantize_point().end(),
      std::back_inserter(quantize_point_));
  std::sort(quantize_point_.begin(), quantize_point_.end());
  */
}

template <typename Dtype>
void RescaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  if (top.size() == 2) top[1]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RescaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < bottom[0]->count(); ++i) {
    if (bottom[0]->cpu_data()[i] >= rf_ || abs(bottom[0]->cpu_data()[i]) <= Dtype(0.0001)) {
      top[0]->mutable_cpu_data()[i] = Dtype(0.0);
      if (top.size() == 2) top[1]->mutable_cpu_data()[i] = Dtype(0.0);
    } else {
      if (top.size() == 2) top[1]->mutable_cpu_data()[i] = Dtype(1.0);
      // linear projection: $y = \frac{2}{rf_}*x - 1$
      top[0]->mutable_cpu_data()[i] = 2 * bottom[0]->cpu_data()[i] / rf_ - 1 ;
    }
  }


}

template <typename Dtype>
void RescaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //do nothing
  return;
}

#ifdef CPU_ONLY
STUB_GPU(RescaleLayer);
#endif

INSTANTIATE_CLASS(RescaleLayer);
REGISTER_LAYER_CLASS(Rescale);

}  // namespace caffe
