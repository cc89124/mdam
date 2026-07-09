#include "clifft/util/cost_meter.h"

namespace clifft {

CostMeter& cost_meter() {
    static CostMeter g;
    return g;
}

const char* cost_kernel_name(int i) {
    switch (static_cast<CostKernel>(i)) {
        case CostKernel::ARRAY_CNOT: return "array_cnot";
        case CostKernel::ARRAY_CZ: return "array_cz";
        case CostKernel::ARRAY_SWAP: return "array_swap";
        case CostKernel::ARRAY_MULTI_CNOT: return "array_multi_cnot";
        case CostKernel::ARRAY_MULTI_CZ: return "array_multi_cz";
        case CostKernel::ARRAY_H: return "array_h";
        case CostKernel::ARRAY_S: return "array_s";
        case CostKernel::ARRAY_S_DAG: return "array_s_dag";
        case CostKernel::ARRAY_T: return "array_t";
        case CostKernel::ARRAY_T_DAG: return "array_t_dag";
        case CostKernel::ARRAY_ROT: return "array_rot";
        case CostKernel::ARRAY_U2: return "array_u2";
        case CostKernel::ARRAY_U4: return "array_u4";
        case CostKernel::EXPAND: return "expand";
        case CostKernel::EXPAND_T: return "expand_t";
        case CostKernel::EXPAND_T_DAG: return "expand_t_dag";
        case CostKernel::EXPAND_ROT: return "expand_rot";
        case CostKernel::MEAS_DIAGONAL: return "meas_diagonal";
        case CostKernel::MEAS_INTERFERE: return "meas_interfere";
        case CostKernel::SWAP_MEAS_INTERFERE: return "swap_meas_interfere";
        case CostKernel::EXP_VAL: return "exp_val";
        default: return "unknown";
    }
}

}  // namespace clifft
