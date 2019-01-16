
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

__kernel void ker_permute_fwd(
    global float* out_data,
    const int num_axes,
    const int count,
    global const int* permute_order,
    global const int* new_steps,
    global const int* old_steps,
    global const float* in_data) {
    int global_idx = get_global_id(0);
    int org_idx    = global_idx;
    int in_idx     = 0;

    if (global_idx < count) {
        for (int i = 0; i < num_axes; i++) {
            int order    = permute_order[i];
            int new_step = new_steps[i];
            int old_step = old_steps[order];
            in_idx += (org_idx / new_step) * old_step;
            org_idx %= new_step;
        }
        out_data[global_idx] = in_data[in_idx];
    }
}

