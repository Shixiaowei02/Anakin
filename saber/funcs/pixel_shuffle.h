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

#ifndef ANAKIN_SABER_PIXEL_SHUFFLE_H
#define ANAKIN_SABER_PIXEL_SHUFFLE_H

#include "saber/funcs/base.h"
#include "saber/funcs/impl/impl_base.h"
#include "saber/funcs/impl/impl_pixel_shuffle.h"

#ifdef AMD_GPU
#include "saber/funcs/impl/amd/include/saber_pixel_shuffle.h"
#endif

namespace anakin {

namespace saber {

template<typename TargetType, DataType OpDtype>
class PixelShuffle: public BaseFunc <
    TargetType,
    OpDtype,
    ImplBase,
    PixelShuffleParam > {

public:
    using BaseFunc <
    TargetType,
    OpDtype,
    ImplBase,
    PixelShuffleParam >::BaseFunc;

    PixelShuffle() = default;

    typedef Tensor<TargetType> InDataTensor;
    typedef Tensor<TargetType> OutDataTensor;
    typedef Tensor<TargetType> OpTensor;
    typedef PixelShuffleParam<TargetType> Param_t;
    typedef std::vector<InDataTensor*> Input_v;
    typedef std::vector<OutDataTensor*> Output_v;
    typedef std::vector<Shape> Shape_v;

    virtual SaberStatus compute_output_shape(const Input_v& inputs, Output_v& outputs, \
            Param_t& param) override {

        CHECK_EQ(inputs.size(), 1) << "The size of input vector is incorrect.";

        Shape shape_out = inputs[0]->valid_shape();
        CHECK_EQ(shape_out.dims(), 4) << "Only support 4-D (NCHW) layout.";
        const int factor = param.upscale_factor;
        shape_out.set_channel(shape_out.channel() / (factor * factor));
        shape_out.set_height(shape_out.height() * factor);
        shape_out.set_width(shape_out.width() * factor);
        return outputs[0]->set_shape(shape_out);
    }

    virtual SaberStatus init_impl(ImplEnum implenum) override {
        switch (implenum) {
            case SABER_IMPL:
                this->_impl.push_back(new SaberPixelShuffle <TargetType, OpDtype>);
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

}
}

#endif // ANAKIN_SABER_PIXEL_SHUFFLE_H
