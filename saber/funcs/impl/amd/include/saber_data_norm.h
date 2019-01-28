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

#ifndef ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_DATA_NORM_H
#define ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_DATA_NORM_H

#include "saber/funcs/base.h"
#include "saber/core/impl/amd/utils/amd_base.h"
#include "saber/funcs/impl/impl_data_norm.h"
#include "saber/saber_types.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/saber_funcs_param.h"
#include "saber/core/impl/amd/utils/amd_kernel.h"
#include "saber/funcs/impl/amd/include/amd_utils.h"

namespace anakin {

namespace saber {

template <DataType OpDtype>
class SaberDataNorm<AMD, OpDtype>:
      public ImplBase<AMD, OpDtype, DataNormParam<AMD> > {

public:
    typedef typename DataTrait<AMD, OpDtype>::Dtype OpDataType;
    typedef AMD_API::TPtr PtrDtype;
    typedef Tensor<AMD> OpTensor;

    SaberDataNorm() {}

    ~SaberDataNorm() {}

    virtual SaberStatus
    init(const std::vector<Tensor<AMD>*>& inputs,
         std::vector<Tensor<AMD>*>& outputs,
         DataNormParam<AMD>& param,
         Context<AMD>& ctx) override;

    virtual SaberStatus
    create(const std::vector<Tensor<AMD>*>& inputs,
           std::vector<Tensor<AMD>*>& outputs,
           DataNormParam<AMD>& param,
           Context<AMD>& ctx) override;

    virtual SaberStatus
    dispatch(const std::vector<Tensor<AMD>*>& inputs,
            std::vector<Tensor<AMD>*>& outputs,
            DataNormParam<AMD>& param) override;

private:
    amd_kernel_list _kernels;
    CreateKernelList(int device_id, KernelInfo& kernelInfo);
    OpTensor _means;
    OpTensor _scales;
};

template class SaberDataNorm<AMD, AK_FLOAT>;

} // namespace saber
} // namespace anakin

#endif // ANAKIN_SABER_FUNCS_IMPL_AMD_SABER_DATA_NORM_H