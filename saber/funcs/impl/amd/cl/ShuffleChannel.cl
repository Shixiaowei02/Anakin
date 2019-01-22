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

__kernel void ker_shuffle_channel_fwd(
        const int count,
        global float* output,
        const int feature_map_size,
        const int group_row,
        const int group_column,
        const int len,
        global const float* input) {

    int index = get_global_id(0);

    if (index < count) {
        const int n = index / group_row / group_column / len;
        const int i = (index / group_column / len) % group_row;
        const int j = index / len % group_column;
        const int k = index - (n * feature_map_size + (i * group_column + j) * len);
        global float* p_o = output + n * feature_map_size + (j * group_row + i) * len;
        p_o[k] = input[index];
    }
}

