#include "CLTensor.h"

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
// register backend
c10::impl::DeviceGuardImplRegistrar ocl_impl_reg(OpenCLDeviceType,&ocl_impl_instance);
} // namespace


