
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

#include "include/saber_cos_sim.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
SaberStatus SaberCosSim<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    CosSimParam<AMD>& param,
    Context<AMD>& ctx) {
    this->_ctx = &ctx;
    return create(inputs, outputs, param, ctx);
}

template <DataType OpDtype>
SaberCosSim<AMD, OpDtype>::CreateKernelList(int device_id, KernelInfo& kernelInfo) {
    AMDKernelPtr kptr = CreateKernel(device_id, &kernelInfo);
    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }
    _kernels.push_back(kptr);
}

template <DataType OpDtype>
SaberStatus SaberCosSim<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    CosSimParam<AMD>& param,
    Context<AMD>& ctx) {

    _kernels.clear();

    int localSize  = 256;
    int globalSize = localSize * inputs[0]->num();

    const int count     = outputs[0]->valid_size();
    cl_context context  = 0;
    cl_device_id device = 0;

    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()];
    device          = dev.get_device();
    context         = dev.get_context();

    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "CosSim.cl";
    kernelInfo.wk_dim      = 1;
    kernelInfo.kernel_type = SABER;
    kernelInfo.l_wk        = {localSize};
    kernelInfo.g_wk        = {globalSize};

    kernelInfo.kernel_name = "ker_cos_sim_fwd";
    CreateKernelList(inputs[0]->device_id(), kernelInfo);
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberCosSim<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    CosSimParam<AMD>& param) {

    CHECK_EQ(inputs.size(), 2) << "CosSim input num should be 2, instead of " << inputs.size();
    CHECK_EQ(outputs.size(), 1) << "CosSim output num need be 1, instead of " << outputs.size();
    CHECK_EQ(inputs[0]->valid_size(), inputs[1]->valid_size()) \
    << "size of the two inputs are not equal.";

    AMD_API::stream_t cm    = this->_ctx->get_compute_stream();
    bool err                = false;
    const int count         = outputs[0]->valid_size();
    OpDataType* top_data    = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* bottom_data = (OpDataType*)inputs[0]->data();

    int num        = inputs[0]->num();
    int inner_size = inputs[0]->valid_size() / inputs[0]->num();
    float epsilon  = param.epsilon;

    OpDataType* out_data    = (OpDataType*)outputs[0]->mutable_data();
    OpDataType* in_0_data = (OpDataType*)inputs[0]->data();
    OpDataType* in_1_data = (OpDataType*)inputs[1]->data();

    amd_kernel_list::iterator it = _kernels.begin();

    err = it->get()->SetKernelArgs(
              (PtrDtype)out_data,
              (PtrDtype)in_0_data,
              (PtrDtype)in_1_data,
              (int)num,
              (int)inner_size,
              (float)epsilon);

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
