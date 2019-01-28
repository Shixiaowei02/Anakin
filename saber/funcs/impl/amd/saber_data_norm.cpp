
/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */

#include "include/saber_data_norm.h"
#include "saber/funcs/saber_util.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberDataNorm<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    DataNormParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberDataNorm<AMD, OpDtype>::CreateKernelList(int device_id, KernelInfo& kernelInfo) {
    AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);
    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }
    _kernels.push_back(kptr);
}

template <DataType OpDtype>
SaberStatus SaberDataNorm<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    DataNormParam<AMD>& param,
    Context<AMD>& ctx) {

    _kernels.clear();

    const int count     = outputs[0]->valid_size();
    cl_context context  = 0;
    cl_device_id device = 0;

    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()];
    device          = dev.get_device();
    context         = dev.get_context();

    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "DataNorm.cl";
    kernelInfo.wk_dim      = 1;
    kernelInfo.kernel_type = SABER;
    kernelInfo.l_wk        = {256};
    kernelInfo.g_wk = {(count + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0]* kernelInfo.l_wk[0]};

    kernelInfo.kernel_name = "ker_data_norm_fwd";
    CreateKernelList(inputs[0]->device_id(), kernelInfo);
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberDataNorm<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    DataNormParam<AMD>& param) {

    int num = inputs[0]->num();
    int channel = inputs[0]->channel();

    Shape channel_shape({1, channel, 1, 1});
    utils::try_expand_tensor(_means, channel_shape);
    utils::try_expand_tensor(_scales, channel_shape);

    AMD_API::stream_t cm    = this->_ctx->get_compute_stream();
    bool err                = false;
    const int count         = outputs[0]->valid_size();
    OpDataType* top_data    = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* bottom_data = (OpDataType*)inputs[0]->data();

    amd_kernel_list::iterator it = _kernels.begin();

    float *means = (float*)_means.mutable_data();
    float *scales = (float*)_scales.mutable_data();
    const float *b_size = (float *)param.batch_size->data();
    const float *b_sum = (float *)param.batch_sum->data();
    const float *b_square_sum = (float *)param.batch_square_sum->data();

    err = it->get()->SetKernelArgs(
              (PtrDtype)top_data,
              (PtrDtype)bottom_data,
              (int)channel,
              (PtrDtype)b_size,
              (PtrDtype)b_sum,
              (PtrDtype)b_square_sum,
              (PtrDtype)means,
              (PtrDtype)scales);

    if (!err) {
        LOG(ERROR) << "Fail to set args";
        return SaberInvalidValue;
    }

    err = LaunchKernel(cm, _kernels);
    if (!err) {
        LOG(ERROR) << "Fail to set execution";
        return SaberInvalidValue;
    }

    return SaberSuccess;
}

}
}
