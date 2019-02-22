/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_LSTMP_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_LSTMP_H

#include "saber/funcs/base.h"
#include "saber/core/impl/amd/utils/amd_base.h"
#include "saber/funcs/impl/impl_lstmp.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/impl/amd/utils/amd_kernel.h"
#include "saber/funcs/impl/amd/include/amd_utils.h"
#include "saber/funcs/saber_util.h"
#include "saber/funcs/impl/amd/include/amd_gemm.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberLstmp<AMD, OpDtype>:
      public ImplBase<AMD, OpDtype, LstmParam<AMD> > {

public:
    typedef typename DataTrait<AMD, OpDtype>::Dtype OpDataType;
    typedef AMD_API::TPtr PtrDtype;
    typedef Tensor<AMD> OpTensor;

    SaberLstmp() {}

    ~SaberLstmp() {}

    virtual SaberStatus
    init(const std::vector<Tensor<AMD>*>& inputs,
         std::vector<Tensor<AMD>*>& outputs,
         LstmParam<AMD>& param,
         Context<AMD>& ctx) override;

    virtual SaberStatus
    create(const std::vector<Tensor<AMD>*>& inputs,
           std::vector<Tensor<AMD>*>& outputs,
           LstmParam<AMD>& param,
           Context<AMD>& ctx) override;

    virtual SaberStatus
    dispatch(const std::vector<Tensor<AMD>*>& inputs,
            std::vector<Tensor<AMD>*>& outputs,
            LstmParam<AMD>& param) override;

private:
    std::unordered_map<int, AMDKernelPtr> _lstm_kernels;
    std::unordered_map<int, AMDKernelPtr> _act_kernels;
    std::vector<AMDKernelPtr> _gemm_kernels;
    CreateKernelList(int device_id, KernelInfo& kernelInfo);
    Tensor<AMD> _wx_tensor;
    Tensor<AMD> _temp_hidden_tensor;
    Tensor<AMD> _temp_cell_tensor;
    int _output_hidden_dim;
    int _inner_hidden_dim;
};

template class SaberLstmp<AMD, AK_FLOAT>;
template class SaberLstmp<AMD, AK_INT8>;
template class SaberLstmp<AMD, AK_HALF>;

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_LSTMP_H

