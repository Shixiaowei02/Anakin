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

#ifndef ANAKIN_SABER_FUNCS_LSTMP_H
#define ANAKIN_SABER_FUNCS_LSTMP_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_lstmp.h"

#ifdef NVIDIA_GPU
#include "saber/funcs/impl/cuda/saber_lstmp.h"
#endif

#ifdef USE_X86_PLACE
#include "saber/funcs/impl/x86/saber_lstmp.h"
#endif

#ifdef AMD_GPU
#include "saber/funcs/impl/amd/include/saber_lstmp.h"
#endif

namespace anakin {
namespace saber {
template<typename TargetType,
         DataType OpDtype>
class Lstmp : public BaseFunc <
    TargetType,
    OpDtype,
    ImplBase,
    LstmParam > {
public:
    using BaseFunc <
    TargetType,
    OpDtype,
    ImplBase,
    LstmParam >::BaseFunc;

    Lstmp() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef LstmParam<TargetType> Param_t;
    typedef std::vector<InDataTensor*> Input_v;
    typedef std::vector<OutDataTensor*> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& input,
            Output_v& output, Param_t& param) override {

        int seqLength = input[0]->num();

        Shape output_shape = Shape({seqLength, param.project_dim, param.num_direction, 1}, input[0]->get_layout());
        output[0]->set_seq_offset(input[0]->get_seq_offset());

        if (output.size() >= 2) {
            output[1]->set_seq_offset(input[0]->get_seq_offset());
        }

        return output[0]->set_shape_without_layout(output_shape);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
        case VENDER_IMPL:
            //this->_impl.push_back(new VenderLstmp <TargetType,
            this->_impl.push_back(new VenderLstmp <TargetType,
                                  OpDtype>);
            return SaberSuccess;

        case SABER_IMPL:
            this->_impl.push_back(new SaberLstmp <TargetType,
                                  OpDtype>);
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
} // namepace anakin


#endif // ANAKIN_SABER_FUNCS_LSTM_H

