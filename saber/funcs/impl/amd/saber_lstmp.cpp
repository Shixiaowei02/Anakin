
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

#include "include/saber_lstmp.h"

namespace anakin {

namespace saber {

static inline int round_up(int k, int c) {
    return ((k + c - 1) / c) * c;
}

template <DataType OpDtype>
SaberStatus SaberLstmp<AMD, OpDtype>::init(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    LstmParam<AMD>& param,
    Context<AMD>& ctx) {

    this->_ctx = &ctx;
    _inner_hidden_dim = param.cell_dim;
    _output_hidden_dim = param.project_dim;

    CHECK_GT(param.cell_dim, 0);
    CHECK_GT(param.project_dim, 0);
    CHECK_EQ(inputs.size(), 1) << "only support input size = 1";
    CHECK_EQ(outputs.size(), 1) << "only support outputs size = 1";
    CHECK_EQ(param.init_hidden() == nullptr, true) << "only support param.init_hidden() == nullptr";
    CHECK_EQ(param.num_layers, 1) << "only support param.num_layers == 1";

    return create(inputs, outputs, param, ctx);
}
/*
template <DataType OpDtype>
static SaberStatus create_lstm_kernel(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    LstmParam<AMD>& param,
    Context<AMD>& ctx,
    std::unordered_map<int, AMDKernelPtr>& lstm_kernels) {

    lstm_kernels.clear();

    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "Lstmp.cl";
    kernelInfo.wk_dim      = 1;
    kernelInfo.kernel_type = SABER;
    kernelInfo.kernel_name = "kernel_lstm_with_peephole";
    std::vector<bool> is_first_iter_vec{true, false};
    for (auto is_first_iter : is_first_iter_vec) {
        kernelInfo.comp_options = std::string(" -DFIRST_ITER=") + std::to_string(int(is_first_iter_vec));
        AMDKernelPtr kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);
        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to load program";
            return;
        }
        lstm_kernels[is_first_iter] = kptr;
    }
    return SaberSuccess;
}

template <DataType OpDtype>
static SaberStatus create_act_kernel(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    LstmParam<AMD>& param,
    Context<AMD>& ctx,
    std::unordered_map<int, AMDKernelPtr>& act_kernels) {

    act_kernels.clear();

    KernelInfo kernelInfo;
    kernelInfo.kernel_file = "Lstmp.cl";
    kernelInfo.wk_dim      = 1;
    kernelInfo.kernel_type = SABER;
    kernelInfo.kernel_name = "kernel_vTanh";
    AMDKernelPtr kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo);
    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return;
    }
    act_kernels[Active_tanh] = kptr;
    return SaberSuccess;
}

template <DataType OpDtype>
static SaberStatus create_gemm_kernel(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    LstmParam<AMD>& param,
    Context<AMD>& ctx,
    std::vector<AMDKernelPtr>& gemm_kernels) {

    gemm_kernels.clear();

    bool err = false;
    bool Ta = false, Tb = false, Tc = false;
    int M = inputs[0]->num();
    int N = 4 * _inner_hidden_dim;
    int K = word_dim;
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
    err = findGemmKernel(M, N, K, Ta, Tb, Tc, cm, gemm_kernels, inputs[0]->device_id());
    if (!err) {
        return SaberInvalidValue;
    }
    return SaberSuccess;
}
*/
template <DataType OpDtype>
SaberStatus SaberLstmp<AMD, OpDtype>::create(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    LstmParam<AMD>& param,
    Context<AMD>& ctx) {

    const int count     = outputs[0]->valid_size();
    cl_context context  = 0;
    cl_device_id device = 0;

    Device<AMD> dev = Env<AMD>::cur_env()[inputs[0]->device_id()];
    device          = dev.get_device();
    context         = dev.get_context();

    _act_kernels.clear();
    _lstm_kernels.clear();
    _gemm_kernels.clear();

    KernelInfo kernelInfo_1;
    kernelInfo_1.kernel_file = "Lstmp.cl";
    kernelInfo_1.wk_dim      = 1;
    kernelInfo_1.kernel_type = SABER;
    kernelInfo_1.kernel_name = "kernel_vTanh";
    AMDKernelPtr kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo_1);
    if (!kptr.get()->isInit()) {
        LOG(ERROR) << "Failed to load program";
        return SaberInvalidValue;
    }
    _act_kernels[Active_tanh] = kptr;

    KernelInfo kernelInfo_2;
    kernelInfo_2.kernel_file = "Lstmp.cl";
    kernelInfo_2.wk_dim      = 1;
    kernelInfo_2.kernel_type = SABER;
    kernelInfo_2.kernel_name = "kernel_lstm_with_peephole";
    std::vector<bool> is_first_iter_vec{true, false};
    for (auto is_first_iter : is_first_iter_vec) {
        kernelInfo_2.comp_options = std::string(" -DFIRST_ITER=") + std::to_string(int(is_first_iter));
        AMDKernelPtr kptr = CreateKernel(inputs[0]->device_id(), &kernelInfo_2);
        if (!kptr.get()->isInit()) {
            LOG(ERROR) << "Failed to load program";
            return SaberInvalidValue;
        }
        _lstm_kernels[is_first_iter] = kptr;
    }

    bool err = false;
    bool Ta = false, Tb = false, Tc = false;
    int M = inputs[0]->num();
    int N = 4 * _inner_hidden_dim;
    int K = inputs[0]->channel();
    AMD_API::stream_t cm = this->_ctx->get_compute_stream();
    err = findGemmKernel(M, N, K, Ta, Tb, Tc, cm, _gemm_kernels, inputs[0]->device_id());
    if (!err) {
        return SaberInvalidValue;
    }
    return SaberSuccess;
}

template <typename dtype, bool first_iter>
static SaberStatus cal_lstm_batch(
    int emit_word_id_size, dtype* temp_wx, const dtype* weight_peephole,
    dtype* hout, dtype* inner_cell, const dtype* b_i_in, const dtype* b_f_in,
    const dtype* b_c_in, const dtype* b_o_in, int hidden_size,
    AMD_API::stream_t cm, std::unordered_map<int, AMDKernelPtr>& lstm_kernels) {

    typedef AMD_API::TPtr PtrDtype;

    bool err = false;
    const int block_dim = 256;
    const int grid_dim = round_up(emit_word_id_size * hidden_size, block_dim);
    const dtype* wc_i = weight_peephole;
    const dtype* wc_f = weight_peephole + hidden_size;
    const dtype* wc_o = weight_peephole + 2 * hidden_size;

    std::vector<size_t> l_wk({block_dim});
    std::vector<size_t> g_wk({grid_dim});

    AMDKernelPtr it = lstm_kernels[first_iter];

    err = it->SetKernelArgs(
              (PtrDtype)temp_wx,
              (PtrDtype)b_i_in,
              (PtrDtype)b_f_in,
              (PtrDtype)b_c_in,
              (PtrDtype)b_o_in,
              (PtrDtype)wc_i,
              (PtrDtype)wc_f,
              (PtrDtype)wc_o,
              (PtrDtype)inner_cell,
              (int)hidden_size,
              (int)emit_word_id_size,
              (PtrDtype)hout);
    err = err && (it->SetLocalWorkSize(l_wk));
    err = err && (it->SetGlobalWorkSize(g_wk));

    if (!err) {
        LOG(ERROR) << "Fail to set args";
        return SaberInvalidValue;
    }
    err = LaunchKernel(cm, it);
    if (!err) {
        LOG(ERROR) << "Fail to set execution";
        return SaberInvalidValue;
    }
    return SaberSuccess;
}

template <typename Dtype>
static SaberStatus vTanh(
    Dtype* data, int count, AMD_API::stream_t cm,
    std::unordered_map<int, AMDKernelPtr>& act_kernels) {

    typedef AMD_API::TPtr PtrDtype;

    int globalSize = count;
    int localSize  = 256;
    std::vector<size_t> l_wk({localSize});
    std::vector<size_t> g_wk({(globalSize + localSize - 1) / localSize * localSize});
    AMDKernelPtr it = act_kernels[Active_tanh];
    int err = it->SetKernelArgs(
              (PtrDtype)data,
              (int)count);
    err = err && (it->SetLocalWorkSize(l_wk));
    err = err && (it->SetGlobalWorkSize(g_wk));
    if (!err) {
        LOG(ERROR) << "Fail to set args";
        return SaberInvalidValue;
    }
    err = LaunchKernel(cm, act_kernels[Active_tanh]);
    if (!err) {
        LOG(ERROR) << "Fail to set execution";
        return SaberInvalidValue;
    }
    return SaberSuccess;
}

template <DataType OpDtype>
SaberStatus SaberLstmp<AMD, OpDtype>::dispatch(
    const std::vector<Tensor<AMD>*>& inputs,
    std::vector<Tensor<AMD>*>& outputs,
    LstmParam<AMD>& param) {

    AMD_API::stream_t cm = this->_ctx->get_compute_stream();

    auto offset_vec = inputs[0]->get_seq_offset();
    CHECK_EQ(offset_vec.size(), 1);
    auto offset = offset_vec[0];
    CHECK_EQ(offset.size(), 2);
    const int skip_num = param.skip_num;
    CHECK_GT(skip_num, 1);
    int word_num = inputs[0]->num();
    int word_dim = inputs[0]->channel();
    int iter_num = utils::round_up(word_num, skip_num) / skip_num;

    utils::try_expand_tensor(_wx_tensor, word_num * 4 * _inner_hidden_dim);
    utils::try_expand_tensor(_temp_hidden_tensor, skip_num * _inner_hidden_dim);
    utils::try_expand_tensor(_temp_cell_tensor, skip_num * _inner_hidden_dim);

    float* wx_ptr                     = (float*)(_wx_tensor.mutable_data());
    const float* x_ptr                = (const float*)(inputs[0]->data());
    const float* weights_x_ptr        = (const float*)(param.weight()->data());
    const float* weights_h_ptr        = weights_x_ptr + word_dim * _inner_hidden_dim * 4;
    const float* weights_project_ptr  = weights_h_ptr + _output_hidden_dim * _inner_hidden_dim * 4;
    const float* weights_bias_ptr     = (const float*)(param.bias()->data());
    const float* weights_bias_i_ptr   = weights_bias_ptr;
    const float* weights_bias_f_ptr   = weights_bias_i_ptr + _inner_hidden_dim;
    const float* weights_bias_c_ptr   = weights_bias_f_ptr + _inner_hidden_dim;
    const float* weights_bias_o_ptr   = weights_bias_c_ptr + _inner_hidden_dim;
    const float* weights_peephole_ptr = weights_bias_ptr + _inner_hidden_dim * 4;
    float* output_ptr                 = (float*)(outputs[0]->mutable_data());
    float* temp_hidden_out            = (float*)(_temp_hidden_tensor.mutable_data());
    float* temp_cell_out              = (float*)(_temp_cell_tensor.mutable_data());

    int err = launchGemmKernel(1.f, x_ptr, 0, 0.f, weights_x_ptr, 0, wx_ptr, 0, cm, _gemm_kernels);
    if (!err) {
        LOG(ERROR) << "Fail to set args";
        return SaberInvalidValue;
    }

    for (int i = 0; i < iter_num; i++) {
        const int run_batch_dim = (i == (iter_num - 1)) ? (word_num - skip_num * i) : skip_num;
        float* wx_iter = wx_ptr + i * skip_num * 4 * _inner_hidden_dim;
        if (i >= 1) {
            float* hidden_in = output_ptr + (i - 1) * skip_num * _output_hidden_dim;
            err = findGemmKernel(run_batch_dim, 4 * _inner_hidden_dim, _output_hidden_dim, false,
                false, false, cm, _gemm_kernels, inputs[0]->device_id());
            if (!err) {
                LOG(ERROR) << "Fail to set args";
                return SaberInvalidValue;
            }
            err = launchGemmKernel(1.f, hidden_in, 0, 1.f, weights_h_ptr, 0, wx_iter, 0,
                cm, _gemm_kernels);
            if (!err) {
                LOG(ERROR) << "Fail to set args";
                return SaberInvalidValue;
            }
            cal_lstm_batch<float, false>(run_batch_dim, wx_iter, weights_peephole_ptr, temp_hidden_out,
                temp_cell_out, weights_bias_i_ptr, weights_bias_f_ptr, weights_bias_c_ptr,
                weights_bias_o_ptr, _inner_hidden_dim, cm, _lstm_kernels);
        } else {
            cal_lstm_batch<float, true>(run_batch_dim, wx_iter, weights_peephole_ptr, temp_hidden_out,
                temp_cell_out, weights_bias_i_ptr, weights_bias_f_ptr, weights_bias_c_ptr,
                weights_bias_o_ptr, _inner_hidden_dim, cm, _lstm_kernels);
        }
        float* hidden_out = output_ptr + i * skip_num * _output_hidden_dim;

        err = findGemmKernel(run_batch_dim, _output_hidden_dim, _inner_hidden_dim, false, false, false,
            cm, _gemm_kernels, inputs[0]->device_id());
        if (!err) {
            LOG(ERROR) << "Fail to set args";
            return SaberInvalidValue;
        }
        err = launchGemmKernel(1.f, temp_hidden_out, 0, 0.f, weights_project_ptr, 0,
            hidden_out, 0, cm, _gemm_kernels);
        if (!err) {
            LOG(ERROR) << "Fail to set args";
            return SaberInvalidValue;
        }
        vTanh(hidden_out, run_batch_dim * _output_hidden_dim, cm, _act_kernels);
    }
    return SaberSuccess;
}

}
}
