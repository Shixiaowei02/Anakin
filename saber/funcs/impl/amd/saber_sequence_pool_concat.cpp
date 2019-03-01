
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

#include "include/saber_sequence_pool_concat.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberSequencePoolConcat<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    SequencePoolConcatParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberSequencePoolConcat<AMD, OpDtype>::CreateKernelList(int device_id, KernelInfo& kernelInfo) {
    AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);
    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }
    _kernels.push_back(kptr);
}

template <DataType OpDtype>
SaberStatus SaberSequencePoolConcat<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    SequencePoolConcatParam<AMD>& param,
    Context<AMD>& ctx) {

    _kernels.clear();

    const int count     = outputs[0]->valid_size();
    cl_context context  = 0;
    cl_device_id device = 0;

    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()];
    device          = dev.get_device();
    context         = dev.get_context();

    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "SequencePoolConcat.cl";
    kernelInfo.wk_dim      = 1;
    kernelInfo.kernel_type = SABER;
    kernelInfo.kernel_name = "sequence_pool_sum_concat";
    CreateKernelList(inputs[0]->device_id(), kernelInfo);
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberSequencePoolConcat<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    SequencePoolConcatParam<AMD>& param) {

    AMD_API::stream_t cm    = this->_ctx->get_compute_stream();
    bool err                = false;
    OpDataType* top_data    = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* bottom_data = (OpDataType*)inputs[0]->data();

    CHECK_GE(inputs[0]->get_seq_offset().size(), 1);
    auto offset = inputs[0]->get_seq_offset()[0];
    CHECK_GE(offset.size(), 1);
//    if (_offset_buffer.get_count() == 0) {
        _offset_buffer.from_vector(offset);
//    }

    const int* offset_data  = (const int*)_offset_buffer.get_data();

    int slot_num = param.slot_num;
    int batch = (offset.size() - 1) / slot_num;
    int xdim = outputs[0]->valid_size();
    CHECK_EQ((xdim % slot_num), 0) << "some data is wrong!!!" << xdim << " " << slot_num;
    CHECK_GE(batch, 1);
    xdim /= slot_num;
    xdim /= batch;

    int count = slot_num * batch * xdim;

    std::vector<size_t> l_wk({256});
    std::vector<size_t> g_wk({(count + l_wk[0] - 1) / l_wk[0] * l_wk[0]});

    amd_kernel_list::iterator it = _kernels.begin();

    err = it->get()->SetKernelArgs(
              (PtrDtype)bottom_data,
              (PtrDtype)top_data,
              (PtrDtype)offset_data,
              (int)(slot_num * batch),
              (int)xdim);
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
    return SaberSuccess;
}

}
}
