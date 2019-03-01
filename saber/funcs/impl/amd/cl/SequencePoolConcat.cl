
__kernel void sequence_pool_sum_concat(
        global const float* input_data,
        global float* output_data,
        global const int* offset,
        int n_total,
        int xdim) {

    const int bid = get_group_id(0);
    const int tid = get_local_id(0);
    const int gid = get_global_id(0);
    const int n_idx = gid / xdim;
    int feature_num;
    int x_idx = gid % xdim;
    if (n_idx < n_total) {
        feature_num = offset[n_idx + 1] - offset[n_idx];
        global float* out_data = output_data + n_idx * xdim;
        global const float* in_data = input_data + offset[n_idx] * xdim;
        float res = 0.f;
        for (int i = 0; i < feature_num; ++i) {
            res += in_data[x_idx];
            in_data += xdim;
        }
        out_data[x_idx] = res;
    }
}
