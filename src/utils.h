#pragma once
#include "CLTensor.h"

namespace ptdlprim {
    /// 
    inline void sync_if_needed(c10::Device const &d)
    {
        CLContextManager::sync_if_needed(d.index());
    }

    dlprim::DataType todp(c10::ScalarType tp);
    
    inline dlprim::DataType todp(caffe2::TypeMeta meta)
    {
        return todp(meta.toScalarType());

    }

    cl::Buffer buffer_from_tensor(torch::Tensor const &tt);
    dlprim::Tensor todp(torch::Tensor const &tt);
    torch::Tensor new_ocl_tensor(torch::IntArrayRef size,c10::Device dev,c10::ScalarType type=c10::kFloat);

    inline dlprim::ExecutionContext getExecutionContext(c10::Device dev)
    {
        return CLContextManager::getCommandQueue(dev.index());
    }
    inline dlprim::ExecutionContext getExecutionContext(torch::Tensor const &t)
    {
        return getExecutionContext(t.device());
    }

    torch::Tensor new_tensor_as(dlprim::Shape const &s,torch::Tensor const &as);
    dlprim::Tensor make_workspace(at::DataPtr &ws_ptr,size_t ws_size,c10::Device const &dev);

    class WSGuard {
    public:
        WSGuard(size_t size,c10::Device const &dev)
        {
            ws = make_workspace(ws_ptr_,size,dev);
        }
        dlprim::Tensor ws;
    private:
        at::DataPtr ws_ptr_;
    };



} // namespace
