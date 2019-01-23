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

#define FLOAT_BYTES 4
#define MAX_LOCAL_SIZE 256
#define SHARE_MEM_SIZE (3 * MAX_LOCAL_SIZE * FLOAT_BYTES)

__kernel void
ker_cos_sim_fwd(__global float* __restrict out_data,
                __global const float* in_0,
                __global const float* in_1,
                const int num,
                const int len,
                const float epsilon) {

    int thread_idx = get_local_id(0);
    int block_idx = get_group_id(0);
    int block_size = get_local_size(0);

    __local float share_mem[SHARE_MEM_SIZE];

    __local float* aa_sum = share_mem;
    __local float* bb_sum = share_mem + block_size;
    __local float* ab_sum = bb_sum + block_size;
    aa_sum[thread_idx] = 0;
    bb_sum[thread_idx] = 0;
    ab_sum[thread_idx] = 0;

    __global const float* in_0_tmp = in_0 + block_idx * len;
    __global const float* in_1_tmp = in_1 + block_idx * len;

    for (int i = thread_idx; i < len; i += block_size) {
        aa_sum[thread_idx] += in_0_tmp[i] * in_0_tmp[i];
        bb_sum[thread_idx] += in_1_tmp[i] * in_1_tmp[i];
        ab_sum[thread_idx] += in_0_tmp[i] * in_1_tmp[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (block_size >= 256) {
        if (thread_idx < 128) {
            int index = thread_idx + 128;
            aa_sum[thread_idx] += aa_sum[index];
            bb_sum[thread_idx] += bb_sum[index];
            ab_sum[thread_idx] += ab_sum[index];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (block_size >= 128) {
        if (thread_idx < 64) {
            int index = thread_idx + 64;
            aa_sum[thread_idx] += aa_sum[index];
            bb_sum[thread_idx] += bb_sum[index];
            ab_sum[thread_idx] += ab_sum[index];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (block_size >= 64) {
        if (thread_idx < 32) {
            int index = thread_idx + 32;
            aa_sum[thread_idx] += aa_sum[index];
            bb_sum[thread_idx] += bb_sum[index];
            ab_sum[thread_idx] += ab_sum[index];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (block_size >= 32) {
        __local volatile float* vaa_sum = aa_sum;
        __local volatile float* vbb_sum= bb_sum;
        __local volatile float* vab_sum= ab_sum;
        if (thread_idx < 16) {
            int index = thread_idx + 16;
            vaa_sum[thread_idx] += vaa_sum[index];
            vbb_sum[thread_idx] += vbb_sum[index];
            vab_sum[thread_idx] += vab_sum[index];
        }
        if (thread_idx < 8) {
            int index = thread_idx + 8;
            vaa_sum[thread_idx] += vaa_sum[index];
            vbb_sum[thread_idx] += vbb_sum[index];
            vab_sum[thread_idx] += vab_sum[index];
        }
        if (thread_idx < 4) {
            int index = thread_idx + 4;
            vaa_sum[thread_idx] += vaa_sum[index];
            vbb_sum[thread_idx] += vbb_sum[index];
            vab_sum[thread_idx] += vab_sum[index];
        }
        if (thread_idx < 4) {
            int index = thread_idx + 2;
            vaa_sum[thread_idx] += vaa_sum[index];
            vbb_sum[thread_idx] += vbb_sum[index];
            vab_sum[thread_idx] += vab_sum[index];
        }
        if (thread_idx < 2) {
            int index = thread_idx + 1;
            vaa_sum[thread_idx] += vaa_sum[index];
            vbb_sum[thread_idx] += vbb_sum[index];
            vab_sum[thread_idx] += vab_sum[index];
        }
    }
    if (thread_idx == 0) {
        float c = aa_sum[0] * bb_sum[0];
        if (c < epsilon) {
            out_data[block_idx] = 0;
        } else {
            out_data[block_idx] = ab_sum[0] / sqrt(c);
        }
    }
}
