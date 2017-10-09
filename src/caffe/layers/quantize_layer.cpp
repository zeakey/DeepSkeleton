#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void QuantizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //const QuantizeParameter& quantize_param = this->layer_param_.quantize_param();
  //quantize_point_.clear();
  /*
  CHECK_GE(quantize_point_.size(), 1)<<"at least one quantize point required!";
  std::copy(quantize_param.quantize_point().begin(),
      quantize_param.quantize_point().end(),
      std::back_inserter(quantize_point_));
  std::sort(quantize_point_.begin(), quantize_point_.end());
  */
}

template <typename Dtype>
void QuantizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void QuantizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bot_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  for (int i = 0; i < bottom[0]->count(); ++i) {
    if (bot_data[i] < 1 || bot_data[i] >=200)
      top_data[i] = Dtype(0);
    else if (bot_data[i] >= 1 && bot_data[i] < 12)
      top_data[i] = Dtype(1);
    else if (bot_data[i] >= 12 && bot_data[i] < 36)
      top_data[i] = Dtype(2);
    else if (bot_data[i] >= 36 && bot_data[i] < 80)
      top_data[i] = Dtype(3);
    else if (bot_data[i] >= 80 && bot_data[i] < 200)          
      top_data[i] = Dtype(4);
    else 
      LOG(FATAL)<<"unknown scale"<<bot_data[i];
  }
}

template <typename Dtype>
void QuantizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //do nothing
  return;
}

#ifdef CPU_ONLY
STUB_GPU(QuantizeLayer);
#endif

INSTANTIATE_CLASS(QuantizeLayer);
REGISTER_LAYER_CLASS(Quantize);

}  // namespace caffe
