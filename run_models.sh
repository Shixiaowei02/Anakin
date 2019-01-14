#!/bin/bash

MODEL_ROOT=/home/public/anakin-models
ANAKIN_ROOT=/home/shixiaowei02/workspace/Anakin

model_list=`find ${MODEL_ROOT} -name "*.bin"`

mkdir -p ${ANAKIN_ROOT}/amd_log
mkdir -p ${ANAKIN_ROOT}/amd_log/failed

for model_path in ${model_list}
    do
    #echo "[RUN] ${model_path}"
    model=${model_path##*/}
    #echo ${ANAKIN_ROOT}/amd_log/${model}.txt
    ${ANAKIN_ROOT}/output/unit_test/net_exec_test ${model_path} 1 1 1 >& ${ANAKIN_ROOT}/amd_log/${model}.txt
    if [ $? -eq 0 ]; then
        echo "[SUCCEED] ${model_path}"
    else
        echo "[FAILED] ${model_path}"
        mv ${ANAKIN_ROOT}/amd_log/${model}.txt ${ANAKIN_ROOT}/amd_log/failed
    fi
    done
