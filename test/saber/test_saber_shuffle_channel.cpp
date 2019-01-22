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
#include "saber/funcs/shuffle_channel.h"

using namespace anakin::saber;

template<typename dtype,typename TargetType_D, typename TargetType_H>
void shuffle_channel_cpu_func(
    const std::vector<Tensor<TargetType_H>*>& input, \
    std::vector<Tensor<TargetType_H>*>& output, \
    ShuffleChannelParam<TargetType_D>& param) {

    int num = input[0]->num();
    int channel = input[0]->channel();
    int height = input[0]->height();
    int width = input[0]->width();
    int feature_map_size = channel * height * width;
    int len = height * width;
    int group_row = param.group;
    int group_column = channel / group_row;
    int out_size = output[0]->valid_size();

    const dtype* src_ptr = static_cast<const dtype*>(input[0]->data());
    dtype* dst_ptr = static_cast<dtype*>(output[0]->mutable_data());

    for (int index = 0; index < out_size; ++index) {
        const int n = index / group_row / group_column / len;
        const int i = (index / group_column / len) % group_row;
        const int j = index / len % group_column;
        const int k = index - (n * feature_map_size + (i * group_column + j) * len);
        dtype* p_o = dst_ptr + n * feature_map_size + (j * group_row + i) * len;
        p_o[k] = src_ptr[index];
    }
}

template <typename TargetType_D, typename TargetType_H, DataType OpDtype>
void test_shuffle_channel(){
    typedef typename DataTrait<TargetType_H, OpDtype>::Dtype dtype;
    TestSaberBase<TargetType_D, TargetType_H, OpDtype, ShuffleChannel, ShuffleChannelParam> testbase;
    for (int s0 : {1, 3}){
        ShuffleChannelParam<TargetType_D> param({s0});
        for (int n : {1, 3}){
            for (int c : {9, 27}){
                for (int h : {2, 8}){
                    for (int w: {8, 2}){
                        testbase.set_param(param);
                        testbase.set_input_shape(Shape({n, c, h, w}));
                        testbase.run_test(shuffle_channel_cpu_func<dtype, TargetType_D, TargetType_H>);
                    }
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_pixel_shuffle)
{
#ifdef USE_CUDA
    test_shuffle_channel<NV, NVHX86, AK_FLOAT>();
#endif
#ifdef USE_X86_PLACE
    //test_shuffle_channel<X86, X86, AK_FLOAT>();
#endif
#ifdef AMD_GPU
    Env<AMD>::env_init();
    test_shuffle_channel<AMD, AMDHX86, AK_FLOAT>();
#endif
}

int main(int argc, const char** argv) {
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}
