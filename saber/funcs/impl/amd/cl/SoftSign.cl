
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

__kernel void ker_soft_sign_fwd(
    global float* out_data,
    global const float* in_data,
    const int count) {

    int tid = get_global_id(0);

    float in_var = in_data[tid];
    float in_abs = in_var > 0 ? in_var : -in_var;
    out_data[tid] = in_var / (in_abs + (float)1.f);

}
