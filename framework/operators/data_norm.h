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

#ifndef ANAKIN_OPERATOR_DATA_NORM_H
#define ANAKIN_OPERATOR_DATA_NORM_H

#include "framework/core/base.h"
#include "framework/core/data_types.h"
#include "framework/core/operator/operator.h"
#include "utils/logger/logger.h"
#include "saber/funcs/data_norm.h"

namespace anakin {

namespace ops {

template<typename Ttype, Precision Ptype>
class DataNormHelper;

/// data_norm op
/**
 * \brief operation of DataNorm class
 * public inheritance Operator
 */
template<typename Ttype, Precision Ptype>
class DataNorm : public Operator<Ttype, Ptype> {
public:
    DataNorm() {}

    /// forward impl
    virtual void operator() (OpContext<Ttype> &ctx, 
                             const std::vector<Tensor4dPtr<Ttype> >& ins, 
                             std::vector<Tensor4dPtr<Ttype> >& outs) {
    LOG(ERROR) << "Not Impl Yet Operator DataNorm< Ttype("
           << target_name<Ttype>::value << "), Precision("<< Ptype <<") >"; 
    }

    friend class DataNormHelper<Ttype, Ptype>;
};

/**
 * \breif provide defined help for some operation
 *  public inheritance OperatorHelper
 *  including init operation context and the size of shape
 */
template<typename Ttype, Precision Ptype>
class DataNormHelper : public OperatorHelper<Ttype, Ptype> {
public:
    DataNormHelper()=default;

    ~DataNormHelper();

    Status InitParam() override;

    /**
    * \brief initial all the resource needed by DataNorm
    * \param ctx stand for operation context
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status Init(OpContext<Ttype> &ctx,
                const std::vector<Tensor4dPtr<Ttype> >& ins, 
                std::vector<Tensor4dPtr<Ttype> >& outs) override;

    /**
    * \brief infer the shape of output and input.
    * \param ins stand for input tensor vector
    * \param outs stand for output tensor vector
    * \return status
    */
    Status InferShape(const std::vector<Tensor4dPtr<Ttype> >& ins,
                      std::vector<Tensor4dPtr<Ttype> >& outs) override;

public:
    ///< _param_data_norm stand for data_norm parameter
    saber::DataNormParam<Ttype> _param;
    ///< _funcs_data_norm stand for data_norm function
    saber::DataNorm<Ttype, PrecisionWrapper<Ptype>::saber_type> _funcs;
};



} /* namespace ops */

} /* namespace anakin */

#endif // ANAKIN_OPERATOR_DATA_NORM_H
