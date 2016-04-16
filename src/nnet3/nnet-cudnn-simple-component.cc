// nnet3/nnet-simple-component.cc

// Copyright      2016  Daniel Galvez

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "nnet3/nnet-cudnn-simple-component.h"
#include "nnet3/nnet-parse.h"
#include "cudamatrix/cudnn-utils.h"
#include <cudnn.h>
#include <numeric>
#include <functional>

namespace kaldi {
namespace nnet3 {

  /*
   * TODO: Constructors
CuDNN3DConvolutionComponent::CudnnConvolutionComponent():
  UpdatableComponent(),
  {}
  */
CuDNN3DConvolutionComponent::~CuDNN3DConvolutionComponent() {
  cudnnDestroyFilterDescriptor(filter_desc_);
  cudnnDestroyConvolutionDescriptor(conv_desc_);
}
void CuDNN3DConvolutionComponent::Init(
    int32 input_x_dim, int32 input_y_dim, int32 input_z_dim,
    int32 filt_x_dim, int32 filt_y_dim, int32 filt_z_dim,
    int32 filt_x_stride, int32 filt_y_stride, int32 filt_z_stride,
    int32 num_filters,
    int32 pad_x_dim, int32 pad_y_dim, int32 pad_z_dim,
    int32 upscale_x_dim, int32 upscale_y_dim, int32 upscale_z_dim,
    TensorVectorizationType input_vectorization,
    BaseFloat param_stddev, BaseFloat bias_stddev) {
  input_x_dim_ = input_x_dim;
  input_y_dim_ = input_y_dim;
  input_z_dim_ = input_z_dim;
  int32 *filters = new int32[kConvolutionDimension_];
  filters[0] = filt_x_dim;
  filters[1] = filt_y_dim;
  filters[2] = filt_z_dim;
  input_vectorization_ = input_vectorization;
  int32 filter_dim = filt_x_dim * filt_y_dim * filt_z_dim;
  filter_params_.Resize(num_filters, filter_dim);
  bias_params_.Resize(num_filters);
  KALDI_ASSERT(param_stddev >= 0.0 && bias_stddev >= 0.0);
  filter_params_.SetRandn();
  filter_params_.Scale(param_stddev);
  bias_params_.SetRandn();
  bias_params_.Scale(bias_stddev);

  int32 *strides = new int32[kConvolutionDimension_];
  strides[0] = filt_x_stride;
  strides[1] = filt_y_stride;
  strides[2] = filt_y_stride;

  int32 *upscales = new int[kConvolutionDimension_];
  upscales[0] = upscale_x_dim;
  upscales[1] = upscale_y_dim;
  upscales[2] = upscale_z_dim;

  int32 *padding = new int[kConvolutionDimension_];
  padding[0] = pad_x_dim;
  padding[1] = pad_y_dim;
  padding[2] = pad_z_dim;

  CUDNN_SAFE_CALL(cudnnCreateConvolutionDescriptor(&conv_desc_));
  CUDNN_SAFE_CALL(
    cudnnSetConvolutionNdDescriptor(conv_desc_,
                                    kConvolutionDimension_,
                                    padding,
                                    strides,
                                    upscales,
                                    CUDNN_CROSS_CORRELATION,
                                    cudnn::GetDataType())
  );

  CUDNN_SAFE_CALL(cudnnCreateFilterDescriptor(&filter_desc_));
  CUDNN_SAFE_CALL(
    cudnnSetFilterNdDescriptor(filter_desc_,
                               cudnn::GetDataType(),
                               kConvolutionDimension_,
                               filters
                               )
  );

  delete[] strides; delete[] upscales; delete[] padding; delete[] filters;
}

void CuDNN3DConvolutionComponent::InitFromConfig(ConfigLine *cfl) {
  bool ok = true;
  int32 input_x_dim = -1, input_y_dim = -1, input_z_dim = -1,
    filt_x_dim = -1, filt_y_dim = -1, filt_z_dim = -1,
    filt_x_stride = -1, filt_y_stride = -1, filt_z_stride = -1,
    num_filters = -1,
    upscale_x_dim = -1, upscale_y_dim = -1, upscale_z_dim = -1,
    pad_x_dim = -1, pad_y_dim = -1, pad_z_dim = -1;

  std::string input_vectorization_order = "zyx";
  TensorVectorizationType input_vectorization = kZyx;
  // TODO: Figure our whether we need to allow input_vectorization_order
  // to be configurable.

  InitLearningRatesFromConfig(cfl);
  ok = ok && cfl->GetValue("input-x-dim", &input_x_dim);
  ok = ok && cfl->GetValue("input-y-dim", &input_y_dim);
  ok = ok && cfl->GetValue("input-z-dim", &input_z_dim);
  ok = ok && cfl->GetValue("filt-x-dim", &filt_x_dim);
  ok = ok && cfl->GetValue("filt-y-dim", &filt_y_dim);
  ok = ok && cfl->GetValue("filt-z-dim", &filt_z_dim);
  ok = ok && cfl->GetValue("filt-x-stride", &filt_x_stride);
  ok = ok && cfl->GetValue("filt-y-stride", &filt_y_stride);
  ok = ok && cfl->GetValue("filt-z-stride", &filt_z_stride);

  // upscale_<k>_dim is how many times to
  // repeat each output in the <k>th dimension. This is
  // usually used to do image synthesis. I think
  // this will not be useful to change for most
  // people. By default, it is set to all ones.
  if(!cfl->GetValue("upscale-x-dim", &upscale_x_dim)) {
    upscale_x_dim = 1;
  }
  if(!cfl->GetValue("upscale-y-dim", &upscale_y_dim)) {
    upscale_y_dim = 1;
  }
  if(!cfl->GetValue("upscale-z-dim", &upscale_z_dim)) {
    upscale_z_dim = 1;
  }

  // If padding is not explicitly given, this code chooses padding so that the input dimension 
  // will equal the output dimension.
  // For a justification of this, search for "The effect of zero padding on network size" in 
  // Chapter 9: Convolutional Networks, of the Deep Learning book by Goodfellow et al.
  // TODO: Make a private function for this.
  // ALSO: I'm not sure whether I should be rounding up or down. Right now,
  // I am rounding down.
  if(!cfl->GetValue("pad-x-dim", &pad_x_dim)) {
    pad_x_dim = ((filt_x_stride - upscale_x_dim)*input_x_dim + filt_x_dim - filt_x_stride)/2;
  }
  if(!cfl->GetValue("pad-y-dim", &pad_y_dim)) {
    pad_y_dim = ((filt_y_stride - upscale_y_dim)*input_y_dim + filt_y_dim - filt_y_stride)/2;
  }
  if(!cfl->GetValue("pad-z-dim", &pad_z_dim)) {
    pad_z_dim = ((filt_z_stride - upscale_z_dim)*input_z_dim + filt_z_dim - filt_z_stride)/2;
  }

  int32 filter_input_dim = filt_x_dim * filt_y_dim * input_z_dim;
  BaseFloat param_stddev = 1.0 / std::sqrt(filter_input_dim), bias_stddev = 1.0;
  cfl->GetValue("param-stddev", &param_stddev);
  cfl->GetValue("bias-stddev", &bias_stddev);

  if (cfl->HasUnusedValues()) {
    KALDI_ERR << "Could not process these elements in initializer: "
              << cfl->UnusedValues();
  }
  if (!ok) {
    KALDI_ERR << "Bad initializer " << cfl->WholeLine();
  }

  Init(input_x_dim, input_y_dim, input_z_dim,
       filt_x_dim, filt_y_dim, filt_z_dim,
       filt_x_stride, filt_y_stride, filt_z_stride,
       num_filters,
       pad_x_dim, pad_y_dim, pad_z_dim,
       upscale_x_dim, upscale_y_dim, upscale_z_dim,
       input_vectorization,
       param_stddev, bias_stddev);
}

std::vector<int32> CuDNN3DConvolutionComponent::GetOutputDims() const {
  cudnnTensorDescriptor_t in_desc;
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&in_desc));
  int32 input_dims[kConvolutionDimension_ + 2] = {1, num_filters_,
                                                  input_z_dim_,
                                                  input_y_dim_,
                                                  input_x_dim_};
  int32 input_strides[kConvolutionDimension_ + 2] =
    {num_filters_*input_z_dim_*input_y_dim_*input_x_dim_,
     input_z_dim_*input_y_dim_*input_x_dim_,
     input_y_dim_*input_x_dim_,
     input_x_dim_,
     1};
  CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(in_desc,
                                             cudnn::GetDataType(),
                                             kConvolutionDimension_ + 2,
                                             input_dims,
                                             input_strides
                                             ));
  int32 output_dims[kConvolutionDimension_ +2];
  CUDNN_SAFE_CALL(
    cudnnGetConvolutionNdForwardOutputDim(conv_desc_,
                                          in_desc,
                                          filter_desc_,
                                          kConvolutionDimension_ + 2,
                                          output_dims
                                          )
                  );
  KALDI_ASSERT(output_dims[0] == 1); // Sanity check: Only one element in fake batch.
  KALDI_ASSERT(output_dims[0] == num_filters_);
  std::vector<int32> output_dims_vec(kConvolutionDimension_);
  for(int i = 0; i < output_dims_vec.size(); i++) {
    output_dims_vec[i] = output_dims[i + 2];
  }

  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(in_desc));
  return output_dims_vec;
}

int32 CuDNN3DConvolutionComponent::OutputDim() const {
  std::vector<int32> output_dims = GetOutputDims();
  int32 output_dim = std::accumulate(output_dims.begin(), output_dims.end(), 1,
                                     std::multiplies<int32>());
  return output_dim;
}

void CuDNN3DConvolutionComponent::Propagate(const ComponentPrecomputedIndexes *indexes,
                                            const CuMatrixBase<BaseFloat> &in,
                                            CuMatrixBase<BaseFloat> *out) const {

  KALDI_ASSERT(input_vectorization_ == kZyx && "Only zyx vectorization supported right now.");

  cudnnTensorDescriptor_t in_desc;
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&in_desc));
  int input_dims[kConvolutionDimension_ + 2] = {in.NumRows(),
                                                num_filters_,
                                                input_z_dim_,
                                                input_y_dim_,
                                                input_x_dim_};
  KALDI_ASSERT(num_filters_ * input_z_dim_ * input_y_dim_ * input_x_dim_ ==
               in.Stride());
  int input_strides[kConvolutionDimension_ + 2] =
    {num_filters_ * input_z_dim_ * input_y_dim_ * input_x_dim_, // == in.Stride()
     input_z_dim_ * input_y_dim_ * input_x_dim_,
     input_y_dim_ * input_x_dim_,
     input_x_dim_,
     1};
  CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(in_desc,
                                             cudnn::GetDataType(),
                                             // 3D convolution means:
                                             // batch dimension, channel, depth, height, and width.
                                             // thus the input tensor is 5 dimensional.
                                             kConvolutionDimension_ + 2,
                                             input_dims,
                                             input_strides
                                             )
                  );
  cudnnTensorDescriptor_t out_desc;
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_SAFE_CALL(
    cudnnConvolutionForward(CuDevice::Instantiate().GetCudnnHandle(),
                            &cudnn::one,
                            in_desc,
                            in.Data(),
                            filter_desc_,
                            filter_params_.Data(),
                            conv_desc_,
                            forward_algo_,
                            work_space_,
                            work_space_size_,
                            &cudnn::zero,
                            out_desc,
                            out->Data()
                            )
                  );
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(out_desc));
}

void CuDNN3DConvolutionComponent::Backprop(const std::string &debug_info,
                                         const ComponentPrecomputedIndexes *indexes,
                                         const CuMatrixBase<BaseFloat> &in_value,
                                         const CuMatrixBase<BaseFloat> &, //out_value,
                                         const CuMatrixBase<BaseFloat> &out_deriv,
                                         Component *to_update_in,
                                         CuMatrixBase<BaseFloat> *in_deriv) const {

  CuDNN3DConvolutionComponent *to_update =
    dynamic_cast<CuDNN3DConvolutionComponent*>(to_update_in);

  cudnnTensorDescriptor_t out_deriv_desc;
  CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&out_deriv_desc));
  int32 out_deriv_dims[kConvolutionDimension_ + 2] =
      {in_deriv->NumRows(),
       num_filters_,
       input_z_dim_,
       input_y_dim_,
       input_x_dim_};
  std::vector<int32> output_dims = GetOutputDims();
  int32 out_deriv_strides[kConvolutionDimension_ + 2] =
      {out_deriv.NumRows(),
       num_filters_,
       output_dims[2],
       output_dims[1],
       output_dims[0]};
  CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(out_deriv_desc,
                                             cudnn::GetDataType(),
                                             kConvolutionDimension_ + 2,
                                             out_deriv_dims,
                                             out_deriv_strides
                                             )
                  );

    KALDI_ASSERT(in_deriv->NumCols() == in_deriv->Stride());
    KALDI_ASSERT(in_deriv->NumCols() == num_filters_ * input_z_dim_ * input_y_dim_ * input_x_dim_);
    int32 in_dims[kConvolutionDimension_ + 2] =
      {in_deriv->NumRows(),
       num_filters_,
       input_z_dim_,
       input_y_dim_,
       input_x_dim_};
    int32 in_strides[kConvolutionDimension_ + 2] = {
      in_deriv->NumCols(),
      input_z_dim_ * input_y_dim_ * input_x_dim_,
      input_y_dim_ * input_x_dim_,
      input_x_dim_,
      1};
  if (in_deriv) {
    cudnnTensorDescriptor_t in_deriv_desc;
    CUDNN_SAFE_CALL(cudnnCreateTensorDescriptor(&in_deriv_desc));
    CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(in_deriv_desc,
                                               cudnn::GetDataType(),
                                               kConvolutionDimension_ + 2,
                                               in_dims,
                                               in_strides
                                               )
                    );


    CUDNN_SAFE_CALL(
      cudnnConvolutionBackwardData(CuDevice::Instantiate().GetCudnnHandle(),
                                   &cudnn::one,
                                   filter_desc_,
                                   filter_params_.Data(),
                                   out_deriv_desc,
                                   out_deriv.Data(),
                                   conv_desc_,
                                   backward_data_algo_,
                                   work_space_,
                                   work_space_size_,
                                   // Even if in_deriv contains NaN values,
                                   // they will not propagate as a special
                                   // case when beta points to 0.
                                   // Section 2.6 in v4.0 manual.
                                   &cudnn::zero,
                                   in_deriv_desc,
                                   in_deriv->Data())
                    );
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(in_deriv_desc));
  }

  if (to_update) {
    cudnnTensorDescriptor_t in_value_desc;
    CUDNN_SAFE_CALL(
      cudnnSetTensorNdDescriptor(in_value_desc,
                                 cudnn::GetDataType(),
                                 kConvolutionDimension_ + 2,
                                 in_dims,
                                 in_strides
                                 )
                    );
    CUDNN_SAFE_CALL(
      cudnnConvolutionBackwardFilter(CuDevice::Instantiate().GetCudnnHandle(),
                                     &learning_rate_, // alpha
                                     in_value_desc,
                                     in_value.Data(),
                                     out_deriv_desc,
                                     out_deriv.Data(),
                                     conv_desc_,
                                     backward_filter_algo_,
                                     work_space_,
                                     work_space_size_,
                                     &cudnn::one, // beta
                                     filter_desc_,
                                     to_update->filter_params_.Data()
                                     )
                    );

    CUDNN_SAFE_CALL(
      cudnnConvolutionBackwardBias(CuDevice::Instantiate().GetCudnnHandle(),
                                   &learning_rate_,
                                   out_deriv_desc,
                                   out_deriv.Data(),
                                   &cudnn::one,
                                   bias_desc_,
                                   to_update->bias_params_.Data()
                                   )
                    );
    CUDNN_SAFE_CALL(cudnnDestroyTensorDescriptor(in_value_desc));
  }
}

} // namespace nnet3
} // namespace kaldi
