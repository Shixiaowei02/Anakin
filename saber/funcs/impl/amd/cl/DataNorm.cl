
/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *
 */

__kernel void ker_data_norm_fwd(
    global float* out_data,
    global const float* in_data,
    const int channel,
    global const float* b_size,
    global const float* b_sum,
    global const float* b_square_sum,
    global float* means,
    global float* scales) {

    int tid = get_global_id(0);

    if (tid < channel) {
        means[tid]  = b_sum[tid] / b_size[tid];
        scales[tid] = sqrt(b_size[tid] / b_square_sum[tid]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int c = tid % channel;
    out_data[tid] = (in_data[tid] - means[c]) * scales[c];


}
