#include "utils.h"

namespace ptdlprim {

    dlprim::DataType todp(c10::ScalarType tp)
    {
        switch(tp) {
        case c10::kFloat:
            return dlprim::float_data;
        case c10::kDouble:
            return dlprim::double_data;
        case c10::kHalf:
            return dlprim::half_data;
        case c10::kBFloat16:
            return dlprim::bfloat16_data;
        case c10::kLong:
            return dlprim::int64_data;
        case c10::kInt:
            return dlprim::int32_data;
        case c10::kShort:
            return dlprim::int16_data;
        case c10::kChar:
            return dlprim::int8_data;
        case c10::kByte:
            return dlprim::uint8_data;
        case c10::kBool:
            TORCH_CHECK(sizeof(bool)==1,"Need to make sure tensors have same size");
            return dlprim::uint8_data;
        default:
            throw std::runtime_error(std::string("Unsupported data type:") + c10::toString(tp));
        }
    }

    cl::Buffer buffer_from_tensor(torch::Tensor const &tt)
    {
        TORCH_CHECK(tt.device().type() == OpenCLDeviceType,"OpenCL device is required for tensor");
        TORCH_CHECK(tt.numel() > 0,"Buffer is not valid for unallocated defvice");
        cl_mem p=static_cast<cl_mem>(const_cast<void*>(tt.getIntrusivePtr()->storage().data()));
        cl::Buffer buf(p,true);
        return buf;
    }
    
    dlprim::Tensor todp(torch::Tensor const &tt)
    {
        TORCH_CHECK(tt.device().type() == OpenCLDeviceType,"OpenCL device is required for tensor");
        TORCH_CHECK(tt.is_contiguous(),"dlprim::Tensor must be contiguous");
        auto sizes = tt.sizes();
        auto offset = tt.storage_offset();
        auto dtype = tt.dtype();
        cl::Buffer buf = buffer_from_tensor(tt);
        dlprim::Shape sp;
        if(sizes.empty())
            sp = dlprim::Shape(1); // scalar
        else
            sp = dlprim::Shape::from_range(sizes.begin(),sizes.end());
        dlprim::Tensor res(buf,offset,sp,todp(dtype));
        return res;
    }

    torch::Tensor new_ocl_tensor(torch::IntArrayRef size,c10::Device dev,c10::ScalarType type)
    {
        size_t n = 1;
        for(auto const &v:size)
            n*=v;
        dlprim::DataType dt = todp(type);
        size_t mem = std::max(size_t(1),n)*dlprim::size_of_data_type(dt);
        c10::Storage storage(c10::Storage::use_byte_size_t(),mem,CLContextManager::allocate(dev,mem));

        c10::DispatchKeySet ks = c10::DispatchKeySet{c10::DispatchKey::OpenCL, c10::DispatchKey::AutogradOpenCL};
        
        c10::intrusive_ptr<c10::TensorImpl> impl=c10::make_intrusive<c10::TensorImpl>(
            std::move(storage),
            ks,
            caffe2::TypeMeta::fromScalarType(type));

        impl->set_sizes_contiguous(size);


        return torch::Tensor::wrap_tensor_impl(impl);

    }

    torch::Tensor new_tensor_as(dlprim::Shape const &s,torch::Tensor const &as)
    {
        int64_t shape[dlprim::max_tensor_dim];
        for(int i=0;i<s.size();i++)
            shape[i]=s[i];
        torch::Tensor result = new_ocl_tensor(c10::IntArrayRef(shape,s.size()),
                                              as.device(),
                                              as.dtype().toScalarType());
        return result;
    }

    dlprim::Tensor make_workspace(at::DataPtr &ws_ptr,size_t ws_size,c10::Device const &dev)
    {
        dlprim::Tensor ws;
        if(ws_size) {
            ws_ptr = std::move(CLContextManager::allocate(dev,ws_size));
            ws=dlprim::Tensor(cl::Buffer((cl_mem)ws_ptr.get(),true),0,dlprim::Shape(ws_size),dlprim::uint8_data);
        }
        return ws;
    }


} // namespace
