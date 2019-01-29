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

__kernel void seq_pool_average_kernel(
        global float* dst,
        global const float* src_in,
        const int batch_size,
        global const int* seq_offset,
        const int slice_size) {

    const int total = slice_size * batch_size;
    const int tid = get_global_id(0);

    if (tid < total) {
        int out_batch_id = tid / slice_size;
        int out_id = tid % slice_size;
        int in_slice_num = seq_offset[out_batch_id + 1] - seq_offset[out_batch_id];
        int in_offset = seq_offset[out_batch_id] * slice_size;
        src_in += in_offset + out_id;
        float sum = (float)0;
        for(int i = 0; i < in_slice_num; ++i){
            sum += src_in[i * slice_size];
        }
        dst[out_batch_id * slice_size + out_id] = sum / in_slice_num; 
    }
}

__kernel void seq_pool_sum_kernel(
        global float* dst,
        global const float* src_in,
        const int batch_size,
        global const int* seq_offset,
        const int slice_size) {

    const int total = slice_size * batch_size;
    const int tid = get_global_id(0);

    if (tid < total) {
        int out_batch_id = tid / slice_size;
        int out_id = tid % slice_size;
        int in_slice_num = seq_offset[out_batch_id + 1] - seq_offset[out_batch_id];
        int in_offset = seq_offset[out_batch_id] * slice_size;
        src_in += in_offset + out_id;
        float sum = (float)0;
        for(int i = 0; i < in_slice_num; ++i){
            sum += src_in[i * slice_size];
        }
        dst[out_batch_id * slice_size + out_id] = sum; 
    }
}

__kernel void seq_pool_sqrt_kernel(
        global float* dst,
        global const float* src_in,
        const int batch_size,
        global const int* seq_offset,
        const int slice_size) {

    const int total = slice_size * batch_size;
    const int tid = get_global_id(0);

    if (tid < total) {
        int out_batch_id = tid / slice_size;
        int out_id = tid % slice_size;
        int in_slice_num = seq_offset[out_batch_id + 1] - seq_offset[out_batch_id];
        int in_offset = seq_offset[out_batch_id] * slice_size;
        src_in += in_offset + out_id;
        float sum = (float)0;
        for(int i = 0; i < in_slice_num; ++i) {
            sum += src_in[i * slice_size];
        }
        dst[out_batch_id * slice_size + out_id] = sum * rsqrt((float)in_slice_num);
    }
}

__kernel void seq_pool_max_kernel(
        global float* dst,
        global const float* src_in,
        const int batch_size,
        global const int* seq_offset,
        const int slice_size) {

    const int total = slice_size * batch_size;
    const int tid = get_global_id(0);

    if (tid < total) {
        int out_batch_id = tid / slice_size;
        int out_id = tid % slice_size;
        int in_slice_num = seq_offset[out_batch_id + 1] - seq_offset[out_batch_id];
        int in_offset = seq_offset[out_batch_id] * slice_size;
        src_in += in_offset + out_id;
        float max = src_in[0];
        for (int i = 1; i < in_slice_num; ++i){
            float val = src_in[i * slice_size];
            if (val > max){
                max = val;
            }
        }
        dst[out_batch_id * slice_size + out_id] = max; 
    }
}

__kernel void seq_pool_last_kernel(
        global float* dst,
        global const float* src_in,
        const int batch_size,
        global const int* seq_offset,
        const int slice_size) {

    const int total = slice_size * batch_size;
    const int tid = get_global_id(0);

    if (tid < total) {
        int out_batch_id = tid / slice_size;
        int out_id = tid % slice_size;
        int in_offset = (seq_offset[out_batch_id + 1]  - 1) * slice_size;
        dst[tid] = src_in[in_offset + out_id];
    }
}

__kernel void seq_pool_first_kernel(
        global float* dst,
        global const float* src_in,
        const int batch_size,
        global const int* seq_offset,
        const int slice_size) {

    const int total = slice_size * batch_size;
    const int tid = get_global_id(0);

    if (tid < total) {
        int out_batch_id = tid / slice_size;
        int out_id = tid % slice_size;
        int in_offset = seq_offset[out_batch_id] * slice_size;
        dst[tid] = src_in[in_offset + out_id];
    }
}

