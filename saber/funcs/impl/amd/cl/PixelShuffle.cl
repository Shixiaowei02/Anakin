
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
    global float* top_data,
    const int num_axes,
    const int count,
    global const int* permute_order,
    global const int* new_steps,
    global const int* old_steps,
    global const float* bottom_data) {

    int index = get_global_id(0);

    if (index < count) {
        int temp_idx = index;
        int old_idx = 0;

        for (int i = 0; i < num_axes; ++i) {
            int order = permute_order[i];
            old_idx += (temp_idx / new_steps[i]) * old_steps[order];
            temp_idx %= new_steps[i];
        }

        top_data[index] = bottom_data[old_idx];
    }
}



