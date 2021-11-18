#include "CLTensor.h"
#include "utils.h"

#include <dlprim/core/util.hpp>
#include <dlprim/core/pointwise.hpp>

#include <iostream>
namespace ptdlprim {

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;


    using torch::Tensor;
    
    torch::Tensor allocate_empty(torch::IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> /*layout*/, c10::optional<Device> device, c10::optional<bool> /*pin_memory*/, c10::optional<MemoryFormat> /*memory_format*/)
    {
        GUARD;
        c10::ScalarType st = dtype ? *dtype : c10::kFloat; 
        c10::Device dev = device ? *device : Device(c10::DeviceType::OPENCL,0);
        return ptdlprim::new_ocl_tensor(size,dev,st);
    }

    /// "aten::empty_strided"
    Tensor empty_strided(torch::IntArrayRef size, torch::IntArrayRef /*stride*/, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) 
    {
        GUARD;
        return allocate_empty(size,dtype,layout,device,pin_memory,c10::nullopt);
    }

    torch::Tensor _reshape_alias(const Tensor & self, c10::IntArrayRef size, c10::IntArrayRef stride)
    {
        GUARD;
        torch::Tensor data = at::alias(self);
        data.getIntrusivePtr()->set_sizes_and_strides(size,stride);
        return data;
    }

    Tensor view(const Tensor & self, IntArrayRef size)
    {
        GUARD;
        torch::Tensor data=at::alias(self);
        TORCH_CHECK(data.is_contiguous(),"View imlemented on contiguous array");
        std::vector<int64_t> v(size.begin(),size.end());
        int64_t total=1,index=-1;
        for(unsigned i=0;i<v.size();i++) {
            if(v[i] == -1) {
                TORCH_CHECK(index==-1,"Must be unique -1");
                index=i;
            }
            else {
                total *= v[i];
            }
        }
        if(index != -1) {
            TORCH_CHECK(self.numel() % total == 0);
            v[index] = self.numel() / total;
        }
        else {
            TORCH_CHECK(total == self.numel());
        }
        c10::IntArrayRef new_size(v.data(),v.size());
        data.getIntrusivePtr()->set_sizes_contiguous(new_size);
        return data;
    }
    Tensor _copy_from(const Tensor & self, const Tensor & dst, bool /*non_blocking*/)
    {
        GUARD;
        if(dst.device().type() == c10::DeviceType::CPU && self.device().type() == c10::DeviceType::OPENCL) {
            Tensor self_c = self.contiguous();
            dlprim::Tensor t(todp(self_c));
            TORCH_CHECK(dst.is_contiguous(),"cpu/gpu need to be contiguous");
            auto ec = getExecutionContext(self);
            void *ptr = dst.data_ptr();
            t.to_host(ec,ptr);
        }
        else if(self.device().type() == c10::DeviceType::CPU && dst.device().type() == c10::DeviceType::OPENCL) {
            dlprim::Tensor t(todp(dst));
            auto ec = getExecutionContext(dst);
            Tensor cself = self.contiguous();
            void *ptr = cself.data_ptr();
            t.to_device(ec,ptr);
        }
        else if(self.device().type() == c10::DeviceType::OPENCL && dst.device().type() == c10::DeviceType::OPENCL) {
            if(self.is_contiguous() && dst.is_contiguous()) {
                dlprim::core::pointwise_operation({todp(self)},{todp(dst)},{},"y0=x0;",getExecutionContext(self.device()));
            }
            else {
                auto src_sizes  = self.sizes();
                auto src_stride = self.strides();
                auto src_offset = self.storage_offset();
                auto tgt_sizes  = dst.sizes();
                auto tgt_stride = dst.strides();
                auto tgt_offset = dst.storage_offset();
                TORCH_CHECK(src_sizes == tgt_sizes);
                dlprim::Shape shape=dlprim::Shape::from_range(src_sizes.begin(),src_sizes.end());
                dlprim::Shape src_std=dlprim::Shape::from_range(src_stride.begin(),src_stride.end());
                dlprim::Shape tgt_std=dlprim::Shape::from_range(tgt_stride.begin(),tgt_stride.end());
                dlprim::core::copy_strided(shape,buffer_from_tensor(self),src_offset,src_std,
                                                 buffer_from_tensor(dst), tgt_offset,tgt_std,
                                                 todp(self.dtype()),getExecutionContext(self.device()));
            }
            sync_if_needed(self.device());
        }
        else {
            throw std::runtime_error("OpenCL supports copy to CPU backend only");
        }
        return self;
    }

    Tensor &fill_(Tensor &self, const c10::Scalar &value)
    {
        GUARD;
        dlprim::Tensor t(todp(self));
        auto q = getExecutionContext(self);
        dlprim::core::fill_tensor(t,value.to<double>(),q);
        sync_if_needed(self.device());
        return self;
    }
    
    Tensor &zero_(Tensor &self)
    {
        GUARD;
        dlprim::Tensor t(todp(self));
        dlprim::core::fill_tensor(t,0.0,getExecutionContext(self));
        return self;
    }

    Tensor as_strided(const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset)
    {
        GUARD;
        Tensor result = at::alias(self);
        result.getIntrusivePtr()->set_sizes_and_strides(size,stride);
        if(storage_offset)
            result.getIntrusivePtr()->set_storage_offset(*storage_offset);
        return result;

    }

    // {"schema": "aten::_local_scalar_dense(Tensor self) -> Scalar", "dispatch": "True", "default": "False"}
    Scalar _local_scalar_dense(const Tensor & self)
    {
        GUARD;
        TORCH_CHECK(self.numel()==1);
        dlprim::Tensor x=todp(self);
        x.to_host(getExecutionContext(self));
        switch(x.dtype()) {
        case dlprim::float_data:
            return *x.data<float>();
        case dlprim::double_data:
            return *x.data<double>();
        case dlprim::int8_data:
            return *x.data<int8_t>();
        case dlprim::uint8_data:
            return *x.data<uint8_t>();
        case dlprim::int16_data:
            return *x.data<int16_t>();
        case dlprim::uint16_data:
            return *x.data<uint16_t>();
        case dlprim::int32_data:
            return (int64_t)*x.data<int32_t>();
        case dlprim::uint32_data:
            return (int64_t)*x.data<uint32_t>();
        case dlprim::int64_data:
            return (int64_t)*x.data<int64_t>();
        case dlprim::uint64_data:
            return (int64_t)*x.data<uint64_t>();
        default:
            TORCH_CHECK(!"Not implemented dtype","Not implemented");
        }
    }


} // namespace dtype

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
      m.impl("aten::empty.memory_format", &ptdlprim::allocate_empty);
      m.impl("aten::empty_strided",&ptdlprim::empty_strided);
      m.impl("aten::_reshape_alias",&ptdlprim::_reshape_alias);
      m.impl("aten::view",&ptdlprim::view);
      m.impl("aten::_copy_from",&ptdlprim::_copy_from);
      m.impl("aten::fill_.Scalar",&ptdlprim::fill_);
      m.impl("aten::zero_",&ptdlprim::zero_);
      m.impl("aten::as_strided",&ptdlprim::as_strided);
      m.impl("aten::_local_scalar_dense",&ptdlprim::_local_scalar_dense);
}
