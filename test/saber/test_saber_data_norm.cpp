#include <stdlib.h>
#include "saber/core/context.h"
#include "saber/funcs/data_norm.h"
#include "test_saber_func.h"
#include "test_saber_base.h"
#include "saber/core/tensor_op.h"
#include "saber/saber_types.h"
#include <vector>
#include <cmath>

namespace anakin {

namespace saber {

template <typename dtype, typename TargetType_D, typename TargetType_H>
void data_norm_cpu(const std::vector<Tensor<TargetType_H>*>& input,
                 std::vector<Tensor<TargetType_H>*>& output, \
                 DataNormParam<TargetType_D>& param) {

    const dtype* in_data = (const dtype*)input[0]->data();
    dtype* out_data = (dtype*)output[0]->mutable_data();
    Shape sh_in = input[0]->valid_shape();
    Shape sh_out = output[0]->valid_shape();

    int num = input[0]->num();
    int channel = input[0]->channel();
    int count = num * channel;

    dtype* means = (dtype*)malloc(channel * sizeof(dtype));
    dtype* scales = (dtype*)malloc(channel * sizeof(dtype));

    Tensor<TargetType_H> b_size_host(Shape({1, channel, 1, 1}));
    Tensor<TargetType_H> b_sum_host(Shape({1, channel, 1, 1}));
    Tensor<TargetType_H> b_square_sum_host(Shape({1, channel, 1, 1}));

    b_size_host.copy_from(*param.batch_size);
    b_sum_host.copy_from(*param.batch_sum);
    b_square_sum_host.copy_from(*param.batch_square_sum);

    const float *b_size = (float *)b_size_host.data();
    const float *b_sum = (float *)b_sum_host.data();
    const float *b_square_sum = (float *)b_square_sum_host.data();

    for (int i = 0; i < channel; i++) {
        means[i]  = b_sum[i] / b_size[i];
        scales[i] = sqrt(b_size[i] / b_square_sum[i]);
    }
    for (int i = 0; i < count; i++) {
        const int c = i % channel;
        out_data[i] = (in_data[i] - means[c]) * scales[c];
    }
    free(means);
    free(scales);
}

template <DataType Dtype,typename TargetType_D, typename TargetType_H>
void test_model() {
    TestSaberBase<TargetType_D, TargetType_H, AK_FLOAT, DataNorm, DataNormParam> testbase;
    for (auto num : {
                1, 3, 4, 11
            }) {
        for (auto c : {
                    1, 3, 11, 4
                }) {
            for (auto h : {
                        1
                    }) {
                for (auto w : {
                            1
                        }) {
                    float eps = 1e-4;
                    Tensor<TargetType_D> batch_sum(Shape({1, c, 1, 1}));
                    Tensor<TargetType_D> batch_size(Shape({1, c, 1, 1}));
                    Tensor<TargetType_D> batch_square_sum(Shape({1, c, 1, 1}));
                    fill_tensor_rand(batch_sum, 0, 1);
                    fill_tensor_rand(batch_size, 0 ,1);
                    fill_tensor_rand(batch_square_sum, 0, 1);
                    DataNormParam<TargetType_D> param(eps, &batch_sum, &batch_size, &batch_square_sum);
                    testbase.set_param(param);
                    testbase.set_input_shape(Shape({num, c, h, w}));
                    testbase.run_test(data_norm_cpu<float, TargetType_D, TargetType_H>);
                }
            }
        }
    }
}

TEST(TestSaberFunc, test_func_data_norm) {
#ifdef USE_CUDA
    Env<NV>::env_init();
    test_model<AK_FLOAT, NV, NVHX86>();
#endif
#ifdef AMD_GPU
    Env<AMD>::env_init();
    test_model<AK_FLOAT, AMD, AMDHX86>();
#endif
}

} // namespace saber
} // namespace anakin

int main(int argc, const char** argv) {
    // initial logger
    //logger::init(argv[0]);
    InitTest();
    RUN_ALL_TESTS(argv[0]);
    return 0;
}



