#include "framework/core/net/calibrator_factory.h"

namespace anakin {

OperatorBase* create_op_with_pt(std::string op_name, std::string precision, std::string target) {
    LOG(INFO) << "creating op:" << op_name << "( precision:" << precision << ",target:" << target <<
              ")";
#ifdef USE_X86_PLACE

    if (target == "X86") {
        if (precision == "fp32") {
            return OpFactory<X86, Precision::FP32>::Global()[op_name];
        }

        if (precision == "int8") {
            return OpFactory<X86, Precision::INT8>::Global()[op_name];
        }

        //
        /*for more precision to add*/
        //
    }

#endif
#ifdef USE_CUDA

    if (target == "NV") {
        if (precision == "fp32") {
            return OpFactory<NV, Precision::FP32>::Global()[op_name];
        }

        if (precision == "int8") {
            return OpFactory<NV, Precision::INT8>::Global()[op_name];
        }

        //
        /*for more precision to add*/
        //
    }

#endif
#ifdef USE_ARM_PLACE

    if (target == "ARM") {
        if (precision == "fp32") {
            return OpFactory<ARM, Precision::FP32>::Global()[op_name];
        }

        if (precision == "int8") {
            return OpFactory<ARM, Precision::INT8>::Global()[op_name];
        }

        //
        /*for more precision to add*/
        //
    }

#endif
#ifdef USE_AMD_PLACE

    if (target == "AMD") {
        if (precision == "fp32") {
            return OpFactory<AMD, Precision::FP32>::Global()[op_name];
        }

        if (precision == "int8") {
            return OpFactory<AMD, Precision::INT8>::Global()[op_name];
        }

        //
        /*for more precision to add*/
        //
    }

#endif

    LOG(FATAL) << "unsupport target or precision! (opname: " << op_name << ",target:" << target <<
               ", precision:" << precision << ")";
    return nullptr;
}

}
