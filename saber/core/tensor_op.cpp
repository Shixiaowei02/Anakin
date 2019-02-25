#include "tensor_op.h"
#include <random>
#include <math.h>
#include <stdio.h>

namespace anakin {

namespace saber {

template <typename HostType>
static void reorder_nhwc_nchw(const Tensor<HostType>& input,
                              Tensor<HostType>& output) {
    CHECK_EQ(input.get_dtype(),AK_FLOAT)<<"only support AK_FLOAT";
    CHECK_EQ(output.get_dtype(),AK_FLOAT)<<"only support AK_FLOAT";
    const float* input_ptr= static_cast<const float *>(input.data());
    float* output_ptr= static_cast<float *>(output.mutable_data());
    int n_value=input.num();
    int c_value=input.channel();
    int h_value=input.height();
    int w_value=input.width();
    if (input.get_layout()==Layout_NHWC&&output.get_layout()==Layout_NCHW){
#pragma omp parallel for collapse(4) schedule(static)
        for (int n = 0; n < n_value; ++n) {
            for (int c = 0; c < c_value; ++c) {
                for (int h = 0; h < h_value; ++h) {
                    for (int w = 0; w < w_value; ++w) {
                        int in_index=n*h_value*w_value*c_value+h*w_value*c_value+w*c_value+c;
                        int out_index=n*c_value*h_value*w_value+c*h_value*w_value+h*w_value+w;
                        output_ptr[out_index]=input_ptr[in_index];
                    }
                }
            }
        }
    }else if (input.get_layout()==Layout_NCHW&&output.get_layout()==Layout_NHWC){
#pragma omp parallel for collapse(4) schedule(static)
        for (int n = 0; n < n_value; ++n) {
            for (int c = 0; c < c_value; ++c) {
                for (int h = 0; h < h_value; ++h) {
                    for (int w = 0; w < w_value; ++w) {
                        int in_index=n*c_value*h_value*w_value+c*h_value*w_value+h*w_value+w;
                        int out_index=n*h_value*w_value*c_value+h*w_value*c_value+w*c_value+c;
                        output_ptr[out_index]=input_ptr[in_index];
                    }
                }
            }
        }
    }else{
                LOG(FATAL)<<"not support layout "<<input.get_layout()<<","<<output.get_layout();
    }

}

#if 0
template <typename HostType>
static void reorder_nchwc_nchw(Tensor<HostType>& input,
                               Tensor<HostType>& output) {

    CHECK_EQ(input.get_dtype(), AK_FLOAT) << "only support float type";
    CHECK((input.get_layout()==Layout_NCHW_C16R||input.get_layout()==Layout_NCHW_C8R)&&output.get_layout()==Layout_NCHW)<<"not support "<<input.get_layout();

    Shape shape = output.valid_shape();
    int n_value = shape[0];
    int c_value = shape[1];
    int h_value = shape[2];
    int w_value = shape[3];
    Shape shape_input = input.valid_shape();
    int aligned_length=shape_input.get_layout_aligned_length();
            CHECK_GT(aligned_length, 0) << "input aligned should > 0";
    int c_round_divk = shape_input[1];

    c_round_divk = (shape_input.channel() + aligned_length-1) / aligned_length;

    float* output_ptr = static_cast<float*>(output.mutable_data());
    const float* input_ptr = static_cast<const float*>(input.data());
#pragma omp parallel for collapse(4) schedule(static)
    for (int n = 0; n < n_value; ++n) {
        for (int c = 0; c < c_value; ++c) {
            for (int h = 0; h < h_value; ++h) {
                //#pragma ivdep
                for (int w = 0; w < w_value; ++w) {
                    int round_c = c / aligned_length;
                    int remainder_c = c % aligned_length;
                    int input_idx = n * c_round_divk * h_value * w_value * aligned_length + round_c * h_value * w_value * aligned_length +
                                    h * w_value * aligned_length + w * aligned_length + remainder_c;
                    int output_idx = n * c_value * h_value * w_value + c * h_value * w_value  +
                                     h * w_value  + w ;

                    *(output_ptr + output_idx) = input_ptr[input_idx];
                }
            }
        }
    }

}

template <typename TargetType>
void tensor_reorder(Tensor<TargetType>& input, Tensor<TargetType>& output){
    if (input.valid_shape()==output.valid_shape()){
        output.copy_from(input);
        return;
    }
    LayoutType in_layout= input.get_layout();
    LayoutType out_layout= output.get_layout();
    bool nhwc_flag=(in_layout==Layout_NHWC&&in_layout==Layout_NCHW)||(out_layout==Layout_NCHW&&out_layout==Layout_NHWC);
    if ((in_layout==Layout_NCHW_C16R||in_layout==Layout_NCHW_C8R)&&out_layout==Layout_NCHW){
        reorder_nchwc_nchw(input,output);
    }else if (nhwc_flag){
        reorder_nhwc_nchw(input,output);
    }else{
        LOG(FATAL)<<"not support this "<<in_layout<<","<<out_layout;
    }
}
#endif

template <typename Dtype>
void fill_tensor_host_const_impl(Dtype* dio, Dtype value, long long size) {
    for (long long i = 0; i < size; ++i) {
        dio[i] = value;
    }
}

template <typename TargetType>
void fill_tensor_const(Tensor<TargetType>& tensor, float value, typename Tensor<TargetType>::API::stream_t stream) {

    long long size = tensor.size();
    void* dio = tensor.mutable_data();
    DataType type = tensor.get_dtype();
    switch (type){
        case AK_UINT8: fill_tensor_host_const_impl((unsigned char*)dio, static_cast<unsigned char>(value), size); break;
        case AK_INT8: fill_tensor_host_const_impl((char*)dio, static_cast<char>(value), size); break;
        case AK_INT16: fill_tensor_host_const_impl((short*)dio, static_cast<short>(value), size); break;
        case AK_UINT16: fill_tensor_host_const_impl((unsigned short*)dio, static_cast<unsigned short>(value), size); break;
        case AK_HALF: fill_tensor_host_const_impl((short*)dio, static_cast<short>(value), size); break;
        case AK_UINT32: fill_tensor_host_const_impl((unsigned int*)dio, static_cast<unsigned int>(value), size); break;
        case AK_INT32: fill_tensor_host_const_impl((int*)dio, static_cast<int>(value), size); break;
        case AK_FLOAT: fill_tensor_host_const_impl((float*)dio, static_cast<float>(value), size); break;
        case AK_DOUBLE: fill_tensor_host_const_impl((double*)dio, static_cast<double>(value), size); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}

template <typename Dtype>
void fill_tensor_host_rand_impl(Dtype* dio, long long size) {
    for (long long i = 0; i < size; ++i) {
        Dtype rand_x=static_cast<Dtype>(rand()%256);
        dio[i] = (rand_x-128)/128;
    }
}
template <>
void fill_tensor_host_rand_impl<char>(char* dio, long long size) {
    for (long long i = 0; i < size; ++i) {
        dio[i] = rand()%256-128;
    }
}
template <>
void fill_tensor_host_rand_impl<unsigned char>(unsigned char* dio, long long size) {
    for (long long i = 0; i < size; ++i) {
        dio[i] = rand()%256;
    }
}
template <typename Dtype>
void fill_tensor_host_seq_impl(Dtype* dio, long long size) {
    for (long long i = 0; i < size; ++i) {
        dio[i] = static_cast<Dtype>(i);
    }
}

template <typename TargetType>
void fill_tensor_rand(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream) {
    long long size = tensor.size();
    void* dio = tensor.mutable_data();
    DataType type = tensor.get_dtype();
    switch (type){
        case AK_UINT8: fill_tensor_host_rand_impl((unsigned char*)dio, size); break;
        case AK_INT8: fill_tensor_host_rand_impl((char*)dio, size); break;
        case AK_INT16: fill_tensor_host_rand_impl((short*)dio, size); break;
        case AK_UINT16: fill_tensor_host_rand_impl((unsigned short*)dio, size); break;
        case AK_UINT32: fill_tensor_host_rand_impl((unsigned int*)dio, size); break;
        case AK_INT32: fill_tensor_host_rand_impl((int*)dio, size); break;
        case AK_HALF: fill_tensor_host_rand_impl((short*)dio, size); break;
        case AK_FLOAT: fill_tensor_host_rand_impl((float*)dio, size); break;
        case AK_DOUBLE: fill_tensor_host_rand_impl((double*)dio, size); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}

template <typename TargetType>
void fill_tensor_seq(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream) {
    long long size = tensor.size();
    void* dio = tensor.mutable_data();
    DataType type = tensor.get_dtype();
    switch (type){
        case AK_UINT8: fill_tensor_host_seq_impl((unsigned char*)dio, size); break;
        case AK_INT8: fill_tensor_host_seq_impl((char*)dio, size); break;
        case AK_INT16: fill_tensor_host_seq_impl((short*)dio, size); break;
        case AK_UINT16: fill_tensor_host_seq_impl((unsigned short*)dio, size); break;
        case AK_UINT32: fill_tensor_host_seq_impl((unsigned int*)dio, size); break;
        case AK_INT32: fill_tensor_host_seq_impl((int*)dio, size); break;
        case AK_HALF: fill_tensor_host_seq_impl((short*)dio, size); break;
        case AK_FLOAT: fill_tensor_host_seq_impl((float*)dio, size); break;
        case AK_DOUBLE: fill_tensor_host_seq_impl((double*)dio, size); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}

template <typename Dtype>
void fill_tensor_host_rand_impl2(Dtype* dio, Dtype vstart, Dtype vend, long long size) {
    std::random_device rd;
    std::mt19937 gen(rd());
//    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dis(0, 1.f);
    for (long long i = 0; i < size; ++i) {
        Dtype random_num = static_cast<Dtype>(vstart + (vend - vstart) * dis(gen));
        dio[i] = random_num;
    }
}

template <typename TargetType>
void fill_tensor_rand(Tensor<TargetType>& tensor, float vstart, float vend, \
    typename Tensor<TargetType>::API::stream_t stream) {

    long long size = tensor.size();
    void* dio = tensor.mutable_data();
    DataType type = tensor.get_dtype();
    switch (type){
        case AK_UINT8: fill_tensor_host_rand_impl2((unsigned char*)dio, static_cast<unsigned char>(vstart),
                                                   static_cast<unsigned char>(vend), size); break;
        case AK_INT8: fill_tensor_host_rand_impl2((char*)dio, static_cast<char>(vstart), static_cast<char>(vend), size); break;
        case AK_INT16: fill_tensor_host_rand_impl2((short*)dio, static_cast<short>(vstart), static_cast<short>(vend), size); break;
        case AK_UINT16: fill_tensor_host_rand_impl2((unsigned short*)dio, static_cast<unsigned short>(vstart),
                                                    static_cast<unsigned short>(vend), size); break;
        case AK_UINT32: fill_tensor_host_rand_impl2((unsigned int*)dio, static_cast<unsigned int>(vstart),
                                                    static_cast<unsigned int>(vend), size); break;
        case AK_INT32: fill_tensor_host_rand_impl2((int*)dio, static_cast<int>(vstart), static_cast<int>(vend), size); break;
        case AK_HALF: fill_tensor_host_rand_impl2((short*)dio, static_cast<short>(vstart), static_cast<short>(vend), size); break;
        case AK_FLOAT: fill_tensor_host_rand_impl2((float*)dio, static_cast<float>(vstart), static_cast<float>(vend), size); break;
        case AK_DOUBLE: fill_tensor_host_rand_impl2((double*)dio, static_cast<double>(vstart), static_cast<double>(vend), size); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}

template <typename Dtype>
void print_tensor_host_impl(const Dtype* din, long long size, int width) {
    for (int i = 0; i < size; ++i) {
        printf("%.6f ", static_cast<float>(din[i]));
        if ((i + 1) % width == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

template <typename TargetType>
void print_tensor(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream) {

    LOG(INFO) << "host tensor data:" << tensor.size();
    const void* data_ptr = tensor.data();
    long long size = tensor.size();
    int width = tensor.width();
    if (tensor.get_layout()==Layout_NCHW_C8){
        width*=8;
    }
    DataType type = tensor.get_dtype();
    switch(type) {
        case AK_UINT8: print_tensor_host_impl((const unsigned char*)data_ptr, size, width); break;
        case AK_INT8: print_tensor_host_impl((const char*)data_ptr, size, width); break;
        case AK_UINT16: print_tensor_host_impl((const unsigned short*)data_ptr, size, width); break;
        case AK_INT16: print_tensor_host_impl((const short*)data_ptr, size, width); break;
        case AK_UINT32: print_tensor_host_impl((const unsigned int*)data_ptr, size, width); break;
        case AK_INT32: print_tensor_host_impl((const int*)data_ptr, size, width); break;
        case AK_FLOAT: print_tensor_host_impl((const float*)data_ptr, size, width); break;
        case AK_DOUBLE: print_tensor_host_impl((const double*)data_ptr, size, width); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
    printf("\n");
}

template <typename TargetType>
void print_tensor_device(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream){
    CHECK(false)<<"not imply print_tensor_device";
}



template <typename TargetType>
void print_tensor_valid(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream) {

    LOG(INFO) << "host tensor data:" << tensor.valid_size();
    const void* data_ptr = (const void*)((const char*)tensor.data() + tensor.data_offset() * type_length(tensor.get_dtype()));
    long long size = tensor.valid_size();
    int width = tensor.width();
    DataType type = tensor.get_dtype();
    if (tensor.is_continue_mem()) {
        switch(type) {
            case AK_UINT8: print_tensor_host_impl((const unsigned char*)data_ptr, size, width); break;
            case AK_INT8: print_tensor_host_impl((const char*)data_ptr, size, width); break;
            case AK_UINT16: print_tensor_host_impl((const unsigned short*)data_ptr, size, width); break;
            case AK_INT16: print_tensor_host_impl((const short*)data_ptr, size, width); break;
            case AK_UINT32: print_tensor_host_impl((const unsigned int*)data_ptr, size, width); break;
            case AK_INT32: print_tensor_host_impl((const int*)data_ptr, size, width); break;
            case AK_FLOAT: print_tensor_host_impl((const float*)data_ptr, size, width); break;
            case AK_DOUBLE: print_tensor_host_impl((const double*)data_ptr, size, width); break;
            default: LOG(FATAL) << "data type: " << type << " is unsupported now";
        }
        printf("\n");
    } else {
        Tensor<TargetType> tvalid(tensor.valid_shape());
        tvalid.copy_from(tensor);
        print_tensor<TargetType>(tvalid, stream);
    }

}

template <typename Dtype>
void tensor_cmp_host(const Dtype* src1, const Dtype* src2, \
                     int size, double& max_ratio, double& max_diff) {

    const double eps = 1e-6f;
    max_diff = fabs(src1[0] - src2[0]);
    max_ratio = fabs(2.0 * max_diff / (src1[0] + src2[0] + eps));

    for (int i = 1; i < size; ++i) {
        double diff = fabs(src1[i] - src2[i]);

        if (max_diff < diff) {
            max_diff = diff;
            max_ratio = fabs(2.0 * max_diff / (src1[i] + src2[i] + eps));
            //LOG(INFO) << "compare two src1: "<< src1[i] << " src2: "<< src2[i] << "i = "<< i << " max_ratio: " << max_ratio ;
        }
    }
}

template <typename Dtype>
void tensor_cmp_host_mlu(const Dtype* correct, const Dtype* sample, \
         int size, double& diff) {

    double sum_diff = 0.0;
    double sum_abs=0.0;

    for (int i = 1; i < size; ++i) {
        double diff = fabs(correct[i] - sample[i]);
        sum_diff += diff*diff;
        sum_abs += fabsf(correct[i])*fabsf(correct[i]);
    }
    diff=sqrt(sum_diff/sum_abs);

}

template <typename Dtype>
double tensor_mean_value_host_impl(const Dtype* din, long long size) {
    double sum = 0.0;
    for (long long i = 0; i < size; ++i) {
        sum += din[i];
    }
    return sum / size;
}

template <typename TargetType>
double tensor_mean_value(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream) {

    const void* data_ptr = tensor.data();
    long long size = tensor.size();
    DataType type = tensor.get_dtype();
    switch (type) {
        case AK_UINT8: return tensor_mean_value_host_impl((const unsigned char*)data_ptr, size);
        case AK_INT8: return tensor_mean_value_host_impl((const char*)data_ptr, size);
        case AK_UINT16: return tensor_mean_value_host_impl((const unsigned short*)data_ptr, size);
        case AK_INT16: return tensor_mean_value_host_impl((const short*)data_ptr, size);
        case AK_UINT32: return tensor_mean_value_host_impl((const unsigned int*)data_ptr, size);
        case AK_INT32: return tensor_mean_value_host_impl((const int*)data_ptr, size);
        case AK_FLOAT: return tensor_mean_value_host_impl((const float*)data_ptr, size);
        case AK_DOUBLE: return tensor_mean_value_host_impl((const double*)data_ptr, size);
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
    return 0.0;
}

template <typename TargetType>
double tensor_mean_value_valid(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream) {

    const void* data_ptr = (const void*)((const char*)tensor.data() + tensor.data_offset() * type_length(tensor.get_dtype()));
    long long size = tensor.valid_size();
    DataType type = tensor.get_dtype();

    if (tensor.is_continue_mem()) {
        switch (type) {
            case AK_UINT8: return tensor_mean_value_host_impl((const unsigned char*)data_ptr, size);
            case AK_INT8: return tensor_mean_value_host_impl((const char*)data_ptr, size);
            case AK_UINT16: return tensor_mean_value_host_impl((const unsigned short*)data_ptr, size);
            case AK_INT16: return tensor_mean_value_host_impl((const short*)data_ptr, size);
            case AK_UINT32: return tensor_mean_value_host_impl((const unsigned int*)data_ptr, size);
            case AK_INT32: return tensor_mean_value_host_impl((const int*)data_ptr, size);
            case AK_FLOAT: return tensor_mean_value_host_impl((const float*)data_ptr, size);
            case AK_DOUBLE: return tensor_mean_value_host_impl((const double*)data_ptr, size);
            default: LOG(FATAL) << "data type: " << type << " is unsupported now";
        }
    } else {
        Tensor<TargetType> tvalid(tensor.valid_shape());
        tvalid.copy_from(tensor);
        return tensor_mean_value<TargetType>(tvalid, stream);
    }

    return 0.0;
}


#define FILL_TENSOR_HOST(target) \
    template void fill_tensor_const<target>(Tensor<target>& tensor, float value, typename Tensor<target>::API::stream_t stream); \
    template void fill_tensor_seq<target>(Tensor<target>& tensor, typename Tensor<target>::API::stream_t stream); \
    template void fill_tensor_rand<target>(Tensor<target>& tensor, typename Tensor<target>::API::stream_t stream); \
    template void fill_tensor_rand<target>(Tensor<target>& tensor, float vstart, float vend, typename Tensor<target>::API::stream_t stream); \
    template void print_tensor<target>(Tensor<target>& tensor, typename Tensor<target>::API::stream_t stream); \
    template void print_tensor_valid<target>(Tensor<target>& tensor, typename Tensor<target>::API::stream_t stream); \
    template double tensor_mean_value<target>(Tensor<target>& tensor, typename Tensor<target>::API::stream_t stream); \
    template double tensor_mean_value_valid<target>(Tensor<target>& tensor, typename Tensor<target>::API::stream_t stream);

#if defined(BUILD_LITE) || defined(USE_X86_PLACE) || defined(USE_CUDA) ||defined(USE_BM_PLACE)
FILL_TENSOR_HOST(X86)
template<>
void tensor_reorder<X86>(Tensor<X86>& input, Tensor<X86>& output);
#endif

#ifdef USE_CUDA
FILL_TENSOR_HOST(NVHX86)
template<>
void tensor_reorder<NVHX86>(Tensor<NVHX86>& input, Tensor<NVHX86>& output);
#endif

#ifdef AMD_GPU
FILL_TENSOR_HOST(AMDHX86)
#endif

#ifdef USE_ARM_PLACE
FILL_TENSOR_HOST(ARM)
#endif

#ifdef USE_BM_PLACE 

#endif

template void tensor_cmp_host_mlu<float>(const float* correct, const float* sample, \
     int size, double& diff);
template void tensor_cmp_host_mlu<int>(const int* correct, const int* sample, \
     int size, double& diff);

template void tensor_cmp_host<float>(const float* src1, const float* src2, \
                                     int size, double& max_ratio, double& max_diff);
template void tensor_cmp_host<int>(const int* src1, const int* src2, \
                                     int size, double& max_ratio, double& max_diff);
template void tensor_cmp_host<char>(const char* src1, const char* src2, int size, \
                                    double& max_ratio, double& max_diff);

} //namespace saber

} //namespace anakin
