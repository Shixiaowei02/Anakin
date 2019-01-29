
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

#include "include/saber_sequence_pool.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberSequencePool<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    SequencePoolParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberSequencePool<AMD, OpDtype>::CreateKernelList(int device_id, KernelInfo& kernelInfo) {
    AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);
    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }
    _kernels.push_back(kptr);
}

template <DataType OpDtype>
SaberStatus SaberSequencePool<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    SequencePoolParam<AMD>& param,
    Context<AMD>& ctx) {

    _kernels.clear();

    const int count     = outputs[0]->valid_size();
    cl_context context  = 0;
    cl_device_id device = 0;

    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()];
    device          = dev.get_device();
    context         = dev.get_context();

    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "SequencePool.cl";
    kernelInfo.wk_dim      = 1;
    kernelInfo.kernel_type = SABER;
    std::vector<size_t> l_wk({256});
    std::vector<size_t> g_wk({256});

    kernel_direct_map = {
        {Sequence_pool_unknow,  "seq_pool_unknow"},
        {Sequence_pool_average, "seq_pool_average_kernel"},
        {Sequence_pool_sum,     "seq_pool_sum_kernel"},
        {Sequence_pool_sqrt,    "seq_pool_sqrt_kernel"},
        {Sequence_pool_max,     "seq_pool_max_kernel"},
        {Sequence_pool_last,    "seq_pool_last_kernel"},
        {Sequence_pool_first,   "seq_pool_first_kernel"},
    };
    kernelInfo.kernel_name = kernel_direct_map[param.sequence_pool_type];
    CreateKernelList(inputs[0]->device_id(), kernelInfo);
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberSequencePool<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    SequencePoolParam<AMD>& param) {

    AMD_API::stream_t cm    = this->_ctx->get_compute_stream();
    bool err                = false;
    const int count         = outputs[0]->valid_size();
    OpDataType* top_data    = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* bottom_data = (OpDataType*)inputs[0]->data();

    int num     = inputs[0]->num();
    int channel = inputs[0]->channel();
    int height  = inputs[0]->height();
    int width   = inputs[0]->width();

    CHECK_EQ(channel, outputs[0]->channel());
    CHECK_EQ(height, outputs[0]->height());
    CHECK_EQ(width, outputs[0]->width());

    std::vector<int> seq_offset = inputs[0]->get_seq_offset()[0];
    int slice_size = channel * height * width;
    int batch_size = seq_offset.size() - 1;
    _seq_offset.re_alloc(Shape({1, 1, 1, (int)seq_offset.size()}), AK_INT32);

    AMD_API::sync_memcpy(_seq_offset.mutable_data(), 0, (int)inputs[0]->device_id(),
        (void*)seq_offset.data(), 0, (int)inputs[0]->device_id(),
         sizeof(int) * seq_offset.size(), __HtoD());

    std::vector<size_t> l_wk({256});
    std::vector<size_t> g_wk({256 * batch_size * slice_size});

    amd_kernel_list::iterator it = _kernels.begin();

    err = it->get()->SetKernelArgs(
              (PtrDtype)top_data,
              (PtrDtype)bottom_data,
              (int)batch_size,
              (PtrDtype)_seq_offset.data(),
              (int)slice_size);

    err = err && (it->get()->SetLocalWorkSize(l_wk));
    err = err && (it->get()->SetGlobalWorkSize(g_wk));
    if (!err) {
        LOG(ERROR) << "Fail to set args";
        return SaberInvalidValue;
    }

    err = LaunchKernel(cm, _kernels);
    if (!err) {
        LOG(ERROR) << "Fail to set execution";
        return SaberInvalidValue;
    }

    std::vector<int> offset_new(batch_size + 1);
    for (int i = 0; i <= batch_size; ++i) {
        offset_new[i] = i;
    }
    std::vector<std::vector<int> > voffset_new;
    voffset_new.push_back(offset_new);
    outputs[0]->set_seq_offset(voffset_new);

    return SaberSuccess;
}

}
}
