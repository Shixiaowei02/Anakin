#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include "debug.h"
#include <fstream>
#if defined(USE_CUDA)
using Target = NV;
using Target_H = X86;
#elif defined(USE_X86_PLACE)
using Target = X86;
using Target_H = X86;
#elif defined(USE_ARM_PLACE)
using Target = ARM;
using Target_H = ARM;
#elif defined(AMD_GPU)
using Target = AMD;
using Target_H = X86;
#endif

//#define USE_DIEPSE

std::string g_model_path = "/path/to/your/anakin_model";

std::string model_saved_path = g_model_path + ".saved";
int g_batch_size = 1;
int g_warm_up = 10;
int g_epoch = 1000;
int g_device_id = 0;


TEST(NetTest, net_execute_base_test) {
    Graph<Target, Precision::FP32>* graph = new Graph<Target, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << g_model_path << " ...";
    // load anakin model files.
    auto status = graph->load(g_model_path);
    if(!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }
    int n = 412;
    std::vector<int> seq_offset{0};
    for (int i = 1; i < 413; ++i) {
        seq_offset.push_back(seq_offset[i-1] + 20);
    }
    // reshape the input_0 's shape for graph model
    Shape new_shape({n, 1, 20, 11}, Layout_NCHW);
    graph->Reshape("input_0", new_shape);
    graph->Optimize();
    Net<Target, Precision::FP32> net_executer(true);
    net_executer.init(*graph);
    // get in
    auto ins = graph->get_ins();
    auto d_tensor_in_p = net_executer.get_in(ins[0]);
    d_tensor_in_p->reshape(new_shape);
    Tensor4d<Target_H> h_tensor_in;

    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i=0; i<valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input_0 dims[" << i << "]" << valid_shape_in[i];
    }

    h_tensor_in.re_alloc(valid_shape_in);
    float* h_data = (float*)(h_tensor_in.mutable_data());

    for (int i=0; i<h_tensor_in.size(); i++) {
        h_data[i] = 1.0f;
    }

    d_tensor_in_p->copy_from(h_tensor_in);
    d_tensor_in_p->set_seq_offset({seq_offset});

    Context<Target> ctx(g_device_id, 0, 0);
    saber::SaberTimer<Target> my_time;
    LOG(WARNING) << "EXECUTER !!!!!!!! ";

	for(int i = 0; i < g_warm_up; i++) {
		net_executer.prediction();
	}
    for(auto x:net_executer.get_in_list()){
        fill_tensor_const(*x, 1);
    }
#ifdef ENABLE_OP_TIMER
    net_executer.reset_op_time();
#endif

    my_time.start(ctx);

    for(int i = 0; i < g_epoch; i++) {
        net_executer.prediction();
    }

    clFlush(ctx.get_compute_stream());
    clFinish(ctx.get_compute_stream());
    my_time.end(ctx);
#ifdef ENABLE_OP_TIMER
    net_executer.print_and_reset_optime_summary(g_epoch);
#endif

    LOG(INFO)<<"aveage time "<<my_time.get_average_ms()/g_epoch << " ms";
    write_tensorfile(*net_executer.get_out_list()[0],"output.txt");
	//} // inner scope over

	LOG(ERROR) << "inner net exe over !";
}


int main(int argc, const char** argv){
    if (argc < 2){
        LOG(ERROR) << "no input!!!, usage: ./" << argv[0] << " model_path [batch size] [warm_up_iter] [test_iter] [device_id]";
        return -1;
    }
    if (argc > 1) {
        g_model_path = std::string(argv[1]);
    }
    if (argc > 2) {
        g_batch_size = atoi(argv[2]);
    }
    if (argc > 3) {
        g_warm_up = atoi(argv[3]);
    }
    if (argc > 4) {
        g_epoch = atoi(argv[4]);
    }
    if (argc > 5) {
        g_device_id = atoi(argv[5]);
    }
    TargetWrapper<Target>::set_device(g_device_id);
    Env<Target>::env_init();
    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
