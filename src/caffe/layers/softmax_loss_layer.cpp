#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  nch = bottom[0]->channels();
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  //if(bottom[0]->has_nan())LOG(WARNING)<<"SOFTMAX LOSS BOTTOM NAN!";

  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  Dtype count_pos = 0,count_neg = 0;
  Dtype neg_loss = 0,pos_loss = 0;
  Dtype temp_neg_loss = 0, temp_pos_loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    temp_pos_loss = 0; temp_neg_loss = 0;
    for (int j = 0; j < inner_num_; j++) {
       int label_value = static_cast<int>(label[i * inner_num_ + j]);
       label_value *= (label_value < nch);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      //
      if(label_value  != 0){
        count_pos++;
        temp_pos_loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],Dtype(FLT_MIN)));
      }else if(label_value ==0){
        count_neg++;
        temp_neg_loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],Dtype(FLT_MIN)));
      }else {
        LOG(FATAL)<<"UNKNOWN TARGET TYPE!";
      }
      ++count;
    }
    pos_loss += temp_pos_loss;
    neg_loss += temp_neg_loss;
  }
  pos_loss = pos_loss * count_neg / (count_pos + count_neg);
  neg_loss = neg_loss * count_pos / (count_pos + count_neg);
  loss = pos_loss + neg_loss;
  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / count;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    Dtype count_pos = 0,count_neg = 0;
    int dim = prob_.count() / outer_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
    //count net/pos
        for(int j = 0; j < inner_num_;++j){
            if(label[i * inner_num_ + j] && label[i * inner_num_ + j] < nch)
                ++count_pos;
            else
                ++count_neg;
        }
        //compute weight 
        Dtype b[2] = {  count_pos / (count_pos + count_neg ),
        count_neg / (count_pos + count_neg ) };

        for (int j = 0; j < inner_num_; ++j) {
            int label_value = static_cast<int>(label[i * inner_num_ + j]) ;
            label_value *= (label_value < nch);
            if (has_ignore_label_ && label_value == ignore_label_) {
                for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
                    bottom_diff[i * dim + c * inner_num_ + j] = 0;
                }
            }else {
                for(int ch = 0;ch < nch;ch++){
                    // formulation of loss partial-derivative computation are here: http://zhaok.xyz/docs/biased_softmax.pdf
                    if (label_value == ch)
                    	bottom_diff[i * dim + ch * inner_num_ + j] = -b[label_value != 0] * (1 - bottom_diff[i * dim + ch * inner_num_ + j]);
                    else
                    	bottom_diff[i * dim + ch * inner_num_ + j] = +b[label_value != 0] * bottom_diff[i * dim + ch * inner_num_ + j];
                }
            }
        }
    }

    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
        caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
        caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
//STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe