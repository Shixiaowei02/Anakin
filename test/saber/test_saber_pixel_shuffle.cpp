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

#include <vector>
#include <limits>

#include "saber/core/context.h"
#include "test/saber/test_saber_base.h"
#include "test/saber/test_saber_func.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include "saber/funcs/pixel_shuffle.h"

using namespace anakin::saber;

template<typename TargetType_D, typename TargetType_H>
SaberStatus update_step(
    const std::vector<Tensor<TargetType_H>*>& inputs,
    PixelShuffleParam<TargetType_D>& param,
    std::vector<int>& old_steps,
    std::vector<int>& new_steps) {

    const int num     = inputs[0]->num();
    const int channel = inputs[0]->channel();
    const int height  = inputs[0]->height();
    const int width   = inputs[0]->width();
    const int factor      = param.upscale_factor;
    const int out_channel = channel / (factor * factor);
    const std::vector<int> old_shape({num, out_channel, factor, factor, height, width}); 
    const std::vector<int> new_shape({num, out_channel, height, factor, width, factor});
    for (int i = 0, old_step = 1, new_step = 1; i < 6; i++) {
        LOG(INFO) << "old_shape " << "i: " << old_shape[i];
        LOG(INFO) << "new_shape " << "i: " << new_shape[i];
        old_step *= old_shape[i];
        new_step *= new_shape[i];
        old_steps.push_back(old_step);
        new_steps.push_back(new_step);
    }
    return SaberSuccess;

}


template<typename dtype,typename TargetType_D, typename TargetType_H>
void pixel_shuffle_cpu_func (
    const std::vector<Tensor<TargetType_H>*>& input, \
    std::vector<Tensor<TargetType_H>*>& output, \
    PixelShuffleParam<TargetType_D>& param) {

    std::vector<int> new_steps;
    std::vector<int> old_steps;
    update_step(input, param, old_steps, new_steps);

    for (int step: old_steps) {
        LOG(INFO) << "cpu old step: " << step;
    }
    for (int step: new_steps) {
        LOG(INFO) << "cpu new step: " << step;
    }

    const dtype* src_ptr = static_cast<const dtype*>(input[0]->data());
    dtype* dst_ptr = static_cast<dtype*>(output[0]->mutable_data());
    std::vector<int> permute_order({0, 1, 2, 3, 4, 5});
    int out_size = output[0]->valid_size();
    int num_axes = 6;
    std::vector<int> new_valid_shape = output[0]->valid_shape();
    for (int j = 0; j < out_size; ++j) {
        int temp_idx = j;
        int old_idx = 0;
        for (int i = 0; i < num_axes; ++i) {
            int order = permute_order[i];
            old_idx += (temp_idx / new_steps[i]) * old_steps[order];
            temp_idx %= new_steps[i];
        }
        dst_ptr[j] = src_ptr[old_idx];
    }

}

template <typename TargetType_D, typename TargetType_H, DataType OpDtype>
void test_pixel_shuffle(){
    typedef typename DataTrait<TargetType_H, OpDtype>::Dtype dtype;
    TestSaberBase<TargetType_D, TargetType_H, OpDtype, PixelShuffle, PixelShuffleParam> testbase;
    for (int s0 : {1}){
        PixelShuffleParam<TargetType_D> param({s0});
        for (int n : {1}){
            for (int c : {9}){
                for (int h : {2}){
                    for (int w: {2}){
                        testbase.set_param(param);
                        testbase.set_input_shape(Shape({n, c, h, w}));
                        testbase.run_test(pixel_shuffle_cpu_func<dtype, TargetType_D, TargetType_H>);
                    }
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_pixel_shuffle)
{
#ifdef USE_CUDA
    test_pixel_shuffle<NV, NVHX86, AK_FLOAT>();
#endif
#ifdef USE_X86_PLACE
    test_pixel_shuffle<X86, X86, AK_FLOAT>();
#endif
#ifdef AMD_GPU
    Env<AMD>::env_init();
    test_pixel_shuffle<AMD, AMDHX86, AK_FLOAT>();
    test_pixel_shuffle<AMD, AMDHX86, AK_FLOAT>();
#endif
}

int main(int argc, const char** argv) {
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
