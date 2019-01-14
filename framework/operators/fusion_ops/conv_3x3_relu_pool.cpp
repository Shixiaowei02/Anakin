#include "framework/operators/fusion_ops/conv_3x3_relu_pool.h"

namespace anakin {

namespace ops {

#define INSTANCE_SASSCONVRELUPOOL(Ttype, Ptype) \
template<> \
void SassConvReluPool<Ttype, Ptype>::operator()(\
    OpContext<Ttype>& ctx,\
    const std::vector<Tensor4dPtr<Ttype> >& ins,\
    std::vector<Tensor4dPtr<Ttype> >& outs) {\
    auto* impl =\
        static_cast<SassConvReluPoolHelper<Ttype, Ptype>*>(this->_helper);\
    auto& param = static_cast<SassConvReluPoolHelper<Ttype, Ptype>*>\
            (this->_helper)->_param_conv_relu_pooling;\
    impl->_funcs_conv_relu_pooling(ins, outs, param, ctx);\
}

/// set helper
template<typename Ttype, Precision Ptype>
SassConvReluPoolHelper<Ttype, Ptype>::~SassConvReluPoolHelper() {}

template<typename Ttype, Precision Ptype>
Status SassConvReluPoolHelper<Ttype, Ptype>::InitParam() {
    DLOG(WARNING) << "Parsing SassConvReluPool op parameter.";

    saber::ConvParam<Ttype> conv_param_temp;
    PoolingParam<Ttype> pooling_param_temp;
    // get conv param
    auto group = GET_PARAMETER(int, group);
    auto bias_term = GET_PARAMETER(bool, bias_term);
    auto padding = GET_PARAMETER(PTuple<int>, padding);
    auto strides = GET_PARAMETER(PTuple<int>, strides);
    auto dilation_rate = GET_PARAMETER(PTuple<int>, dilation_rate);
    auto filter_num = GET_PARAMETER(int, filter_num);
    auto kernel_size = GET_PARAMETER(PTuple<int>, kernel_size);
    auto axis = GET_PARAMETER(int, axis);

    using pblock_type = PBlock<Ttype>;
    auto weights = GET_PARAMETER(pblock_type, weight_1);
    auto weight_vec = weights.vector();

    // get relu param
    auto alpha = GET_PARAMETER(float, relu_0_alpha);
    ActivationParam<Ttype> active_param(Active_relu, alpha); // Temp

    // get pooling param
    auto global_pooling = GET_PARAMETER(bool, pooling_0_global_pooling);
    auto pool_padding = GET_PARAMETER(PTuple<int>, pooling_0_padding);
    auto pool_strides = GET_PARAMETER(PTuple<int>, pooling_0_strides);
    auto pool_size = GET_PARAMETER(PTuple<int>, pooling_0_pool_size);
    auto pool_method = GET_PARAMETER(std::string, pooling_0_method);
    auto cmp_out_shape_floor_as_conv = GET_PARAMETER(bool, pooling_0_cmp_out_shape_floor_as_conv);

    if (pool_method == "MAX") {
        PoolingParam<Ttype> pooling_param(pool_size[0], pool_size[1],
                                          pool_padding[0], pool_padding[1], pool_strides[0], pool_strides[1],
                                          Pooling_max, global_pooling, cmp_out_shape_floor_as_conv);

        pooling_param_temp = pooling_param;
    } else if (pool_method == "AVG") {
        PoolingParam<Ttype> pooling_param(pool_size[0], pool_size[1],
                                          pool_padding[0], pool_padding[1], pool_strides[0], pool_strides[1],
                                          Pooling_average_include_padding, global_pooling,
                                          cmp_out_shape_floor_as_conv);

        pooling_param_temp = pooling_param;
    } else {
        LOG(FATAL) << " SassConvReluPool fusion op doesn't support : "
                   << pool_method << " pooling.";
    }

    if (bias_term) {
        auto bias = GET_PARAMETER(pblock_type, weight_2);
        saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                                           strides[0], strides[1], dilation_rate[0], dilation_rate[1],
                                           &(weights.d_tensor()), &(bias.d_tensor()),
                                           active_param);
        conv_param_temp = conv_param;
    } else {
        Tensor4d<Ttype>* bias = new Tensor4d<Ttype>();
        saber::ConvParam<Ttype> conv_param(group, padding[0], padding[1],
                                           strides[0], strides[1], dilation_rate[0], dilation_rate[1],
                                           &(weights.d_tensor()), bias, active_param);
        conv_param_temp = conv_param;
    }

    ConvPoolingParam<Ttype> conv_act_pooling_param(conv_param_temp, pooling_param_temp);
    _param_conv_relu_pooling = conv_act_pooling_param;

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SassConvReluPoolHelper<Ttype, Ptype>::Init(OpContext<Ttype>& ctx,
        const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {

    auto group = GET_PARAMETER(int, group);
    auto strides = GET_PARAMETER(PTuple<int>, strides);
    auto weights = GET_PARAMETER(PBlock<Ttype>, weight_1);
    auto bias_term = GET_PARAMETER(bool, bias_term);

    saber::ImplEnum impl_e = SABER_IMPL;
    _funcs_conv_relu_pooling.init(ins, outs, _param_conv_relu_pooling, SPECIFY,
                                  impl_e, ctx);

    // check if weights have been transposed
    auto is_weights_transed = CHECK_PARAMETER(is_weights_transed);

    if (!is_weights_transed) {
        SET_PARAMETER(is_weights_transed, true, bool);

        if (bias_term) {
            auto bias = GET_PARAMETER(PBlock<Ttype>, weight_2);
            graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                std::bind(&ConvPooling < Ttype,
                          PrecisionWrapper<Ptype>::saber_type >::trans_weights,
                          &_funcs_conv_relu_pooling, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10),
                weights.d_tensor(), bias.d_tensor(), _param_conv_relu_pooling.conv_param.pad_h,
                _param_conv_relu_pooling.conv_param.pad_w, _param_conv_relu_pooling.conv_param.dilation_h,
                _param_conv_relu_pooling.conv_param.dilation_w,
                strides[0], strides[1], group, impl_e);
            bias.map_to_host();
        } else {
            PBlock<Ttype> bias_empty;
            graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
                std::bind(&ConvPooling < Ttype,
                          PrecisionWrapper<Ptype>::saber_type >::trans_weights,
                          &_funcs_conv_relu_pooling, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10),
                weights.d_tensor(), bias_empty.d_tensor(), _param_conv_relu_pooling.conv_param.pad_h,
                _param_conv_relu_pooling.conv_param.pad_w, _param_conv_relu_pooling.conv_param.dilation_h,
                _param_conv_relu_pooling.conv_param.dilation_w,
                strides[0], strides[1], group, impl_e);
        }

        weights.map_to_host();

    } else {
        PBlock<Ttype> weight_empty;
        PBlock<Ttype> bias_empty;
        graph::GraphGlobalMem<Ttype>::Global().template apply<Level_0>(
            std::bind(&ConvPooling<Ttype, PrecisionWrapper<Ptype>::saber_type>::trans_weights,
                      &_funcs_conv_relu_pooling, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10),
            weight_empty.d_tensor(), bias_empty.d_tensor(), _param_conv_relu_pooling.conv_param.pad_h,
            _param_conv_relu_pooling.conv_param.pad_w, _param_conv_relu_pooling.conv_param.dilation_h,
            _param_conv_relu_pooling.conv_param.dilation_w,
            strides[0], strides[1], group, impl_e);
    }

    return Status::OK();
}

template<typename Ttype, Precision Ptype>
Status SassConvReluPoolHelper<Ttype, Ptype>::InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins,
        std::vector<Tensor4dPtr<Ttype> >& outs) {
    SABER_CHECK(_funcs_conv_relu_pooling.compute_output_shape(ins, outs, _param_conv_relu_pooling));
    return Status::OK();
}

#ifdef USE_CUDA
template<>
Status SassConvReluPoolHelper<NV, Precision::INT8>::Init(OpContext<NV>& ctx,
        const std::vector<Tensor4dPtr<NV> >& ins,
        std::vector<Tensor4dPtr<NV> >& outs) {

    auto group = GET_PARAMETER(int, group);
    auto strides = GET_PARAMETER(PTuple<int>, strides);
    auto weights = GET_PARAMETER(PBlock<NV>, weight_1);
    auto bias_term = GET_PARAMETER(bool, bias_term);

    saber::ImplEnum impl_e = VENDER_IMPL;
    _funcs_conv_relu_pooling.init(ins, outs, _param_conv_relu_pooling, SPECIFY,
                                  impl_e, ctx);

    // check if weights have been transposed
    auto is_weights_transed = CHECK_PARAMETER(is_weights_transed);

    if (!is_weights_transed) {
        SET_PARAMETER(is_weights_transed, true, bool);

        if (bias_term) {
            auto bias = GET_PARAMETER(PBlock<NV>, weight_2);
            graph::GraphGlobalMem<NV>::Global().template apply<Level_0>(
                std::bind(&ConvPooling < NV,
                          PrecisionWrapper<Precision::INT8>::saber_type >::trans_weights,
                          &_funcs_conv_relu_pooling, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10),
                weights.d_tensor(), bias.d_tensor(), _param_conv_relu_pooling.conv_param.pad_h,
                _param_conv_relu_pooling.conv_param.pad_w, _param_conv_relu_pooling.conv_param.dilation_h,
                _param_conv_relu_pooling.conv_param.dilation_w,
                strides[0], strides[1], group, impl_e);
            bias.map_to_host();
        } else {
            PBlock<NV> bias_empty;
            graph::GraphGlobalMem<NV>::Global().template apply<Level_0>(
                std::bind(&ConvPooling < NV,
                          PrecisionWrapper<Precision::INT8>::saber_type >::trans_weights,
                          &_funcs_conv_relu_pooling, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10),
                weights.d_tensor(), bias_empty.d_tensor(), _param_conv_relu_pooling.conv_param.pad_h,
                _param_conv_relu_pooling.conv_param.pad_w, _param_conv_relu_pooling.conv_param.dilation_h,
                _param_conv_relu_pooling.conv_param.dilation_w,
                strides[0], strides[1], group, impl_e);
        }

        weights.map_to_host();

    } else {
        PBlock<NV> weight_empty;
        PBlock<NV> bias_empty;
        graph::GraphGlobalMem<NV>::Global().template apply<Level_0>(
            std::bind(&ConvPooling<NV, PrecisionWrapper<Precision::INT8>::saber_type>::trans_weights,
                      &_funcs_conv_relu_pooling, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10),
            weight_empty.d_tensor(), bias_empty.d_tensor(), _param_conv_relu_pooling.conv_param.pad_h,
            _param_conv_relu_pooling.conv_param.pad_w, _param_conv_relu_pooling.conv_param.dilation_h,
            _param_conv_relu_pooling.conv_param.dilation_w,
            strides[0], strides[1], group, impl_e);
    }

    return Status::OK();
}
#endif

#ifdef USE_CUDA
template class SassConvReluPoolHelper<NV, Precision::FP32>;
template class SassConvReluPoolHelper<NV, Precision::FP16>;
template class SassConvReluPoolHelper<NV, Precision::INT8>;
#endif

#ifdef USE_ARM_PLACE
template class SassConvReluPoolHelper<ARM, Precision::FP32>;
template class SassConvReluPoolHelper<ARM, Precision::FP16>;
template class SassConvReluPoolHelper<ARM, Precision::INT8>;
#endif

#ifdef AMD_GPU
template class SassConvReluPoolHelper<AMD, Precision::FP32>;
template class SassConvReluPoolHelper<AMD, Precision::FP16>;
template class SassConvReluPoolHelper<AMD, Precision::INT8>;
#endif

// register helper
#ifdef USE_CUDA
INSTANCE_SASSCONVRELUPOOL(NV, Precision::FP32);
INSTANCE_SASSCONVRELUPOOL(NV, Precision::INT8);
ANAKIN_REGISTER_OP_HELPER(SassConvReluPool, SassConvReluPoolHelper, NV, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(SassConvReluPool, SassConvReluPoolHelper, NV, Precision::INT8);
#endif

#ifdef USE_X86_PLACE
INSTANCE_SASSCONVRELUPOOL(X86, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(SassConvReluPool, SassConvReluPoolHelper, X86, Precision::FP32);
#endif

#ifdef USE_ARM_PLACE
INSTANCE_SASSCONVRELUPOOL(ARM, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(SassConvReluPool, SassConvReluPoolHelper, ARM, Precision::FP32);
#endif

#ifdef AMD_GPU
INSTANCE_SASSCONVRELUPOOL(AMD, Precision::FP32);
ANAKIN_REGISTER_OP_HELPER(SassConvReluPool, SassConvReluPoolHelper, AMD, Precision::FP32);
#endif

//! register op
ANAKIN_REGISTER_OP(SassConvReluPool)
.Doc("SassConvReluPool fusion operator")
#ifdef USE_CUDA
.__alias__<NV, Precision::FP32>("convolution_relu_pooling")
.__alias__<NV, Precision::INT8>("convolution_relu_pooling")
#endif
#ifdef USE_ARM_PLACE
.__alias__<ARM, Precision::FP32>("convolution_relu_pooling")
#endif
#ifdef AMD_GPU
.__alias__<AMD, Precision::FP32>("convolution_relu_pooling")
#endif
.num_in(1)
.num_out(1)
.Args<int>("group", " group of conv ")
.Args<bool>("bias_term", " whether conv weights have bias")
.Args<PTuple<int>>("padding", "padding of conv (x, y)")
                .Args<PTuple<int>>("strides", "strides of conv (x)")
                .Args<PTuple<int>>("dilation_rate", "dilation rate of conv (x)")
                .Args<int>("filter_num", "filter(kernel) number of weights")
                .Args<PTuple<int>>("kernel_size", "kernel size of kernel (x, y)")
                .Args<int>("axis", "axis of conv")
                .Args<bool>("pooling_0_global_pooling", " whether use pooling for all input area.")
                .Args<PTuple<int>>("pooling_0_padding", " paddding of pooling ")
                .Args<PTuple<int>>("pooling_0_strides", " strides of pooling ")
                .Args<PTuple<int>>("pooling_0_pool_size", "pooling size of pooling")
                .Args<std::string>("pooling_0_method", " pooling methods")
                .Args<bool>("pooling_0_cmp_out_shape_floor_as_conv", "cmp_out_shape_floor_as_conv")
                .Args<float>("relu_0_alpha", " alpha for relu");

} /* namespace ops */

} /* namespace anakin */


