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
#include "framework/operators/data_norm.h"

namespace anakin {

namespace ops {

#define INSTANCE_DATA_NORM(Ttype, Ptype) \
template<> \
void DataNorm<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<DataNormHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = impl->_param; \
    impl->_funcs(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
DataNormHelper<Ttype, Ptype>::~DataNormHelper() {
}

template<typename Ttype, Precision Ptype>
Status DataNormHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing DataNorm op parameter.";

    using pblock_type = PBlock<Ttype>;
    auto batch_sum = GET_PARAMETER(pblock_type, weight_1);
    auto batch_size = GET_PARAMETER(pblock_type, weight_2);
    auto batch_square_sum = GET_PARAMETER(pblock_type, weight_3);
    auto epsilon = GET_PARAMETER(float, epsilon);

    saber::DataNormParam<Ttype> axpy_param(epsilon, &(batch_sum.d_tensor()), &(batch_size.d_tensor()),
        &(batch_square_sum.d_tensor()));
    _param = axpy_param;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status DataNormHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs.init(ins, outs, _param, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status DataNormHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >&
        ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs.compute_output_shape(ins, outs, _param));
    return Status::OK();
}

#ifdef USE_CUDA
INSTANCE_DATA_NORM(NV, Precision::FP32);
template class DataNormHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(DataNorm, DataNormHelper, NV, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_DATA_NORM(AMD, Precision::FP32);
template class DataNormHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(DataNorm, DataNormHelper, AMD, Precision::FP32);
#endif

#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_DATA_NORM(X86, Precision::FP32);
template class DataNormHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(DataNorm, DataNormHelper, X86, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(DataNorm)
.Doc("DataNorm operator")
.num_in(3)
.num_out(1);

} /* namespace ops */

} /* namespace anakin */


