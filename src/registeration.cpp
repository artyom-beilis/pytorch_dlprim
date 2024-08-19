#include "CLTensor.h"
#include "utils.h"
#include <torch/version.h>

#if TORCH_VERSION_MAJOR <= 1
    #define DLPRIM_NO_HOOKS_INTERFACE
    #if TORCH_VERSION_MINOR < 13
    #error "Supported pytorch versions are ==1.13 or >=2.4"
    #endif
#else
    #if TORCH_VERSION_MINOR < 4
    #error "Supported pytorch 2.x should be  >=2.4"
    #endif
#endif


#ifndef DLPRIM_NO_HOOKS_INTERFACE
#include <ATen/detail/PrivateUse1HooksInterface.h>
#endif

namespace ptdlprim {

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;

class OCLDevImpl : public c10::impl::DeviceGuardImplInterface {
public:
    OCLDevImpl() 
    {
    } 
    virtual DeviceType type() const { return OpenCLDeviceType; }
    virtual Device exchangeDevice(Device d) const {
        Device prev = dt_;
        dt_ = d;
        return prev;
    }
    virtual Device getDevice() const { 
        return dt_; 
    }
    virtual void setDevice(Device d) const { 
        dt_ = d; 
    }
    virtual void uncheckedSetDevice(Device d) const noexcept  { 
        dt_ = d; 
    }
    virtual c10::Stream getStream(Device d) const noexcept { 
        return c10::Stream(c10::Stream::UNSAFE,d,0);
    }
    virtual c10::Stream getDefaultStream(Device d) const { 
        return getStream(d); 
    }
    virtual c10::Stream exchangeStream(Stream) const noexcept { 
        return getStream(dt_); 
    }
    virtual DeviceIndex deviceCount() const noexcept { 
        try {
            return CLContextManager::count();
        }
        catch(...) {
            return 0;
        }
    }
    virtual bool queryStream(const Stream& /*stream*/) const {
        return false;
    }
    virtual void synchronizeStream(const Stream& stream) const {
        auto device = stream.device();
        CLContextManager::getCommandQueue(device.index()).finish();
    }
private:

    static thread_local Device dt_; 
    static thread_local Stream s_; 
} ocl_impl_instance;

thread_local Device OCLDevImpl::dt_ = Device(OpenCLDeviceType,0);
thread_local Stream OCLDevImpl::s_  = Stream(c10::Stream::UNSAFE,Device(OpenCLDeviceType,0),0);


class OCLAllocator : public at::Allocator {
public:
    at::Device current_device() const
    {
        return ocl_impl_instance.getDevice();
    }

    at::DataPtr allocate(size_t nbytes) override
    {
        at::Device device = current_device();
        return CLContextManager::allocate(device,nbytes);
    }
    virtual void copy_data(void* dest, const void* src, std::size_t count) const override
    {
        GUARD;
        at::Device device = current_device();
        cl::Buffer buf_dst((cl_mem)dest,true);
        cl::Buffer buf_src((cl_mem)src, true);
        auto q = getExecutionContext(device);
        q.queue().enqueueCopyBuffer(buf_src,buf_dst,0,0,count,q.events(),q.event("copy_data"));
        sync_if_needed(device);
    }
} ocl_allocator_instance;

REGISTER_ALLOCATOR(OpenCLDeviceType, &ocl_allocator_instance);

#ifndef DLPRIM_NO_HOOKS_INTERFACE
struct HooksInterface : public at::PrivateUse1HooksInterface {
    bool hasPrimaryContext(c10::DeviceIndex device_index) const override
    {
        return CLContextManager::is_ready(device_index);
    }
    virtual ~HooksInterface() {};
};

struct _Reg {
    static HooksInterface interface;
    _Reg() {
        at::RegisterPrivateUse1HooksInterface(&interface);
    }
} reg_impl;

HooksInterface _Reg::interface;

#endif


// register backend
c10::impl::DeviceGuardImplRegistrar ocl_impl_reg(OpenCLDeviceType,&ocl_impl_instance);

} // namespace

