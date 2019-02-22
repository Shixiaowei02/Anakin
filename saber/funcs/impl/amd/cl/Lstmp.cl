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

#ifndef FIRST_ITER
#define FIRST_ITER 1
#endif

#include "saber/funcs/impl/amd/include/saber_normal_activation.h"

__kernel void kernel_lstm_with_peephole(
    global const float* w_x,
    global const float* b_i,
    global const float* b_f,
    global const float* b_c,
    global const float* b_o,
    global const float* w_ci,
    global const float* w_cf,
    global const float* w_co,
    global float* cell,
    const int hidden_size,
    const int batch_size,
    global float* output) {

    const int thread_idx = get_local_id(0);
    const int batch_id = thread_id / hidden_size;
    const int tid = thread_id % hidden_size;

    if (tid < hidden_size && batch_id < batch_size) {
        const int emit_wx_offset = batch_id * hidden_size * 4;
        __global const float* w_x_i = w_x + emit_wx_offset;
        __global const float* w_x_f = w_x_i + hidden_size;
        __global const float* w_x_c = w_x_f + hidden_size;
        __global const float* w_x_o = w_x_c + hidden_size;
        __global float* gate_h_p = output + batch_id * hidden_size;
        __global float* gate_c_p = cell + batch_id * hidden_size;
#if FIRST_ITER
        const float gate_i = Sigmoid(w_x_i[tid] + b_i[tid]);
        const float gate_f = Sigmoid(w_x_f[tid] + b_f[tid]);

        const float gate_c_s = Tanh(w_x_c[tid]  + b_c[tid]);
        const float gate_c = gate_i * gate_c_s;
        const float gate_o = Sigmoid(w_x_o[tid] + b_o[tid] + gate_c * w_co[tid]);
        gate_c_p[tid] = gate_c;
        gate_h_p[tid] = gate_o * Tanh(gate_c);
#else
        const float c_1 = gate_c_p[tid];
        const float gate_i = Sigmoid(w_x_i[tid] + b_i[tid] + w_ci[tid] * c_1);
        const float gate_f = Sigmoid(w_x_f[tid] + b_f[tid] + w_cf[tid] * c_1);

        const float gate_c_s = Tanh(w_x_c[tid]  + b_c[tid]);
        const float gate_c = gate_f * c_1 + gate_i * gate_c_s;
        const float gate_o = Sigmoid(w_x_o[tid] + b_o[tid] + gate_c * w_co[tid]);
        gate_c_p[tid] = gate_c;
        gate_h_p[tid] = gate_o * Tanh(gate_c);
#endif
    }
}

__kernel void kernel_vTanh(
    global float* data,
    const int count) {
    if (thread_id<count) {
        data[thread_id] = Tanh(data[thread_id]);
    }
}
