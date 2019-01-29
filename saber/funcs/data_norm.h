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

#ifndef ANAKIN_SABER_FUNCS_DATA_NORM_H
#define ANAKIN_SABER_FUNCS_DATA_NORM_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_data_norm.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_data_norm.h"
#endif

#ifdef USE_X86_PLACE
//#include "saber/funcs/impl/x86/saber_data_norm.h"
#endif

#ifdef AMD_GPU 
#include "saber/funcs/impl/amd/include/saber_data_norm.h"
#endif

#ifdef USE_ARM_PLACE
//#include "saber/funcs/impl/arm/saber_data_norm.h"
#endif

#ifdef USE_BM_PLACE 
//#include "saber/funcs/impl/bm/vender_data_norm.h"
#endif


namespace anakin {
namespace saber {

template<typename TargetType,
        DataType OpDtype>
class DataNorm : public BaseFunc<
        TargetType,
        OpDtype,
        ImplBase,
        DataNormParam> {
public:
    using BaseFunc<
            TargetType,
            OpDtype,
            ImplBase,
            DataNormParam>::BaseFunc;

    DataNorm() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef DataNormParam<TargetType> Param_t;
    typedef std::vector<InDataTensor *> Input_v;
    typedef std::vector<OutDataTensor *> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v &input,
                                             Output_v &output, Param_t &param) override {

        CHECK_EQ(input.size(), 1) << "Input of DataNorm should be 1.";
        if (input[0]->get_layout() == Layout_NCHW) {
            CHECK_LE(input[0]->height(), 1) << "Shape of DataNorm Input should be (N, C, 1, 1).";
            CHECK_LE(input[0]->width(), 1) << "Shape of DataNorm Input should be (N, C, 1, 1).";
        } else if (input[0]->get_layout() != Layout_NC) {
            LOG(FATAL) << "The layout of input is incorrect.";
        };

        Shape output_shape({input[0]->num(), input[0]->channel()}, Layout_NC);

        output[0]->set_seq_offset(input[0]->get_seq_offset());
        return output[0]->set_shape(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case SABER_IMPL:
                this->_impl.push_back(new SaberDataNorm <TargetType, OpDtype>);
                return SaberSuccess;

            default:
                return SaberUnImplError;
        }
    }

private:

    virtual void pick_best_static() override {
        this->_best_impl = this->_impl[0];
    }

    virtual void pick_best_specify(ImplEnum implenum) override {
        this->_best_impl = this->_impl[0];
    }

};



} // namespace saber
} // namespace anakin

#endif
