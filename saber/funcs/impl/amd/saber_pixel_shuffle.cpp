
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

#include "include/saber_pixel_shuffle.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberPixelShuffle<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PixelShuffleParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    _old_steps.re_alloc(Shape({1, 1, 1, 6}), AK_INT32);
    _new_steps.re_alloc(Shape({1, 1, 1, 6}), AK_INT32);
    _permute_order.re_alloc(Shape({1, 1, 1, 6}), AK_INT32);

    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberPixelShuffle<AMD, OpDtype>::CreateKernelList(int device_id, KernelInfo& kernelInfo) {
    AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);
    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }
    _kernels.push_back(kptr);
}

template <DataType OpDtype>
SaberStatus SaberPixelShuffle<AMD, OpDtype>::load_clkernel(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PixelShuffleParam<AMD>& param,
    Context<AMD>& ctx) {

    _kernels.clear();

    const int count     = outputs[0]->valid_size();
    cl_context context  = 0;
    cl_device_id device = 0;

    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()];
    device          = dev.get_device();
    context         = dev.get_context();

    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "PixelShuffle.cl";
    kernelInfo.wk_dim      = 1;
    kernelInfo.kernel_type = SABER;
    kernelInfo.l_wk        = {256};
    kernelInfo.g_wk = {(count + kernelInfo.l_wk[0] - 1) / kernelInfo.l_wk[0]* kernelInfo.l_wk[0]};

    kernelInfo.kernel_name = "ker_permute_fwd";
    CreateKernelList(inputs[0]->device_id(), kernelInfo);
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberPixelShuffle<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PixelShuffleParam<AMD>& param,
    Context<AMD>& ctx) {

    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    const int num     = inputs[0]->num();
    const int channel = inputs[0]->channel();
    const int height  = inputs[0]->height();
    const int width   = inputs[0]->width();

    const int factor      = param.upscale_factor;
    const int out_channel = channel / (factor * factor);
    const int count       = inputs[0]->valid_shape().count();

    const Shape old_shape({num, out_channel, factor, factor, height, width});
    const Shape new_shape({num, out_channel, height, factor, width, factor});

    std::vector<int> old_steps;
    std::vector<int> new_steps;
    std::vector<int> permute_order({0, 1, 4, 2, 5, 3});

    for (int i = 0; i < 6; i++) {
        old_steps.push_back(old_shape.count(i + 1));
        new_steps.push_back(new_shape.count(i + 1));
    }

    cl_mem old_steps_p     = (cl_mem)_old_steps.mutable_data();
    cl_mem new_steps_p     = (cl_mem)_new_steps.mutable_data();
    cl_mem permute_order_p = (cl_mem)_permute_order.mutable_data();

    if (inputs[0]->is_continue_mem() && outputs[0]->is_continue_mem()) {
        AMD_API::async_memcpy(old_steps_p, 0, (int)inputs[0]->device_id(),
            (void*) & old_steps[0], 0, (int)inputs[0]->device_id(),
            sizeof(int) * 6, cm, __HtoD());
        AMD_API::async_memcpy(new_steps_p, 0, (int)inputs[0]->device_id(),
            (void*) & new_steps[0], 0, (int)inputs[0]->device_id(),
            sizeof(int) * 6, cm, __HtoD());
        AMD_API::async_memcpy(permute_order_p, 0, (int)inputs[0]->device_id(),
            (void*) & permute_order[0], 0, (int)inputs[0]->device_id(),
            sizeof(int) * 6, cm, __HtoD());
    } else {
        return SaberInvalidValue;
    }

    return load_clkernel(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberStatus SaberPixelShuffle<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    PixelShuffleParam<AMD>& param) {

    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    bool err                     = false;
    const int count              = outputs[0]->valid_size();
    const int* old_steps_p       = (int*)_old_steps.data();
    const int* new_steps_p       = (int*)_new_steps.data();
    const int* permute_order_p   = (int*)_permute_order.data();

    OpDataType* top_data         = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* bottom_data      = (OpDataType*)inputs[0]->data();

    int img_size                 = outputs[0]->height() * outputs[0]->width();
    amd_kernel_list::iterator it = _kernels.begin();

    err = it->get()->SetKernelArgs(
              (PtrDtype)top_data,
              6,
              (int)count,
              (PtrDtype)permute_order_p,
              (PtrDtype)new_steps_p,
              (PtrDtype)old_steps_p,
              (PtrDtype)bottom_data);
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
