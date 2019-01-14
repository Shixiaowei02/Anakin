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
#include "framework/operators/reshape.h"

namespace anakin {

namespace ops {

#define INSTANCE_RESHAPE(Ttype, Ptype) \
template<> \
void Reshape<Ttype, Ptype>::operator()(OpContext<Ttype>& ctx, \
    const std::vector<Tensor4dPtr<Ttype> >& ins, \
    std::vector<Tensor4dPtr<Ttype> >& outs) { \
    auto* impl = \
        static_cast<ReshapeHelper<Ttype, Ptype>*>(this->_helper); \
    auto& param = \
        static_cast<ReshapeHelper<Ttype, Ptype>*>(this->_helper)->_param_reshape; \
    impl->_funcs_reshape(ins, outs, param, ctx); \
}

template<typename Ttype, Precision Ptype>
Status ReshapeHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing Reshape op parameter.";
    LayoutType layout_type = Layout_invalid;
    auto dims = GET_PARAMETER(PTuple<int>, dims);
    auto dims_v = dims.vector();

    if (FIND_PARAMETER(layout)) {
        std::string layout = GET_PARAMETER(std::string, layout);
        layout_type = layout_from_string(layout);
        CHECK_EQ(dims_v.size(), dims_from_layout(layout_type)) << "Layout in params does not match dims";
    } else if (dims_v.size() == 4) {
        layout_type = Layout_NCHW;
    } else {
        LOG(FATAL) << "The layout is required when the shape is not equal to 4";
    }

    CHECK_NE(layout_type, Layout_invalid) << "Unknown layout parameter";
    ReshapeParam<Ttype> param_reshape(dims_v, layout_type);
    _param_reshape = param_reshape;
    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status ReshapeHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype>>& ins,
        std::vector<Tensor4dPtr<Ttype>>& outs) {
    SABER_CHECK(_funcs_reshape.init(ins, outs, _param_reshape, SPECIFY, SABER_IMPL, ctx));
    return Status::OK();
}


template<typename Ttype, Precision Ptype>
Status ReshapeHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype>>& ins,
        std::vector<Tensor4dPtr<Ttype>>& outs) {
    SABER_CHECK(_funcs_reshape.compute_output_shape(ins, outs, _param_reshape));
    outs[0]->set_seq_offset(ins[0]->get_seq_offset());
    return Status::OK();
}


#ifdef USE_CUDA
INSTANCE_RESHAPE(NV, Precision::FP32);
template class ReshapeHelper<NV, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Reshape, ReshapeHelper, NV, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_RESHAPE(AMD, Precision::FP32);
template class ReshapeHelper<AMD, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Reshape, ReshapeHelper, AMD, Precision::FP32);
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
INSTANCE_RESHAPE(X86, Precision::FP32);
template class ReshapeHelper<X86, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Reshape, ReshapeHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_RESHAPE(ARM, Precision::FP32);
template class ReshapeHelper<ARM, Precision::FP32>;
ANAKIN_REGISTER_OP_HELPER(Reshape, ReshapeHelper, ARM, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(Reshape)
.Doc("Reshape operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("reshape")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("reshape")
#endif
#if defined USE_X86_PLACE || defined BUILD_LITE
.__alias__<X86, Precision::FP32>("reshape")
#endif
.num_in(1)
.num_out(1)
.Args<PTuple<int>>("dims", " dims of redhape target");

} /* namespace ops */

} /* namespace anakin */
