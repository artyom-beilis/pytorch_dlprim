#include <torch/torch.h>
#include <ATen/ATen.h>
#include "CLTensor.h"

#include <dlprim/core/ip.hpp>
#include <dlprim/core/bn.hpp>
#include <dlprim/core/util.hpp>
#include <dlprim/core/conv.hpp>
#include <dlprim/core/bias.hpp>
#include <dlprim/core/pool.hpp>
#include <dlprim/core/loss.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/core/activation.hpp>
#include <dlprim/random.hpp>

#include <iostream>

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
    virtual DeviceType type() const { return c10::DeviceType::OPENCL; }
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


thread_local Device OCLDevImpl::dt_ = Device(DeviceType::OPENCL,0);
thread_local Stream OCLDevImpl::s_  = Stream(c10::Stream::UNSAFE,Device(DeviceType::OPENCL,0),0);

//#define LOG_CALLS 
#define LOG_EXCEPTIONS

#if defined(BM_CALLS)
#define GUARD dlprim::ExecGuard exec_guard(getExecutionContext(ocl_impl_instance.getDevice()),__func__);
#elif defined(LOG_EXCEPTIONS)
struct ExcGuard {
    ExcGuard(char const *name) : name_(name)
    {
    }
    ~ExcGuard()
    {
        if(std::uncaught_exception()) {
            std::cerr << "Exception from : " << name_ << std::endl;
        }
    }
    char const *name_;
};
#define GUARD ExcGuard debug_guard(__PRETTY_FUNCTION__);
#elif defined(LOG_CALLS)
std::atomic<int> indent;
struct LogGuard {
    LogGuard(char const *name) : name_(name)
    {
        int v=indent++;
        for(int i=0;i<v;i++)
            std::cout << "  ";
        std::cout << "in:  " << name_ << std::endl;
    }
    ~LogGuard()
    {
        indent--;
    }
    char const *name_;
};
#define GUARD LogGuard debug_guard(__PRETTY_FUNCTION__);
#else
#define GUARD
#endif



// register backend
c10::impl::DeviceGuardImplRegistrar ocl_impl_reg(c10::DeviceType::OPENCL,&ocl_impl_instance);

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

    bool isCPUScalar(Tensor const &other,double &value)
    {
        if(other.device() == Device(c10::kCPU) && other.numel()==1) {
            switch(other.dtype().toScalarType()){
            case c10::kFloat:
                value = *static_cast<float const *>(other.data_ptr());
                break;
            case c10::kDouble:
                value = *static_cast<double const *>(other.data_ptr());
                break;
            case c10::kLong:
                value = *static_cast<int64_t const *>(other.data_ptr());
                break;
            default:
                TORCH_CHECK(false,"Unsupported cpu data type");
            }
            return true;
        }
        return false;
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
        TORCH_CHECK(data.is_contiguous(),"View imlemented on contigonous array");
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


    Tensor _copy_from(const Tensor & self, const Tensor & dst, bool /*non_blocking*/)
    {
        GUARD;
        if(dst.device().type() == c10::DeviceType::CPU && self.device().type() == c10::DeviceType::OPENCL) {
            dlprim::Tensor t(todp(self));
            TORCH_CHECK(dst.is_contiguous(),"cpu/gpu need to be contiguous");
            auto ec = getExecutionContext(self);
            void *ptr = dst.data_ptr();
            t.to_host(ec,ptr);
        }
        else if(self.device().type() == c10::DeviceType::CPU && dst.device().type() == c10::DeviceType::OPENCL) {
            dlprim::Tensor t(todp(dst));
            TORCH_CHECK(self.is_contiguous(),"cpu/gpu need to be contiguous");
            auto ec = getExecutionContext(dst);
            void *ptr = self.data_ptr();
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
                auto tgt_stride = self.strides();
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

    dlprim::core::Conv2DSettings conv_config(dlprim::Tensor &X,dlprim::Tensor &W,IntArrayRef padding,IntArrayRef stride,IntArrayRef dilation,int groups)
    {
        TORCH_CHECK(stride.size()==2 && padding.size() == 2 && dilation.size() == 2,"Expecting size of parameters=2");
        dlprim::Convolution2DConfigBase cfg_base;
        cfg_base.channels_in = X.shape()[1];
        cfg_base.channels_out = W.shape()[0];
        for(int i=0;i<2;i++) {
            cfg_base.kernel[i] = W.shape()[i+2];
            cfg_base.pad[i] = padding[i];
            cfg_base.stride[i] = stride[i];
            cfg_base.dilate[i] = dilation[i];
            cfg_base.groups = groups;
        }
        dlprim::core::Conv2DSettings cfg(cfg_base,X.shape(),X.dtype()); 
        return cfg;
    }


    dlprim::Tensor make_workspace(at::DataPtr &ws_ptr,size_t ws_size,Device const &dev)
    {
        dlprim::Tensor ws;
        if(ws_size) {
            ws_ptr = std::move(CLContextManager::allocate(dev,ws_size));
            ws=dlprim::Tensor(cl::Buffer((cl_mem)ws_ptr.get(),true),0,dlprim::Shape(ws_size),dlprim::uint8_data);
        }
        return ws;
    }

    class WSGuard {
    public:
        WSGuard(size_t size,Device const &dev)
        {
            ws = make_workspace(ws_ptr_,size,dev);
        }
        dlprim::Tensor ws;
    private:
        at::DataPtr ws_ptr_;
    };

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

    Tensor convolution_overrideable(const Tensor & input,
                                    const Tensor & weight,
                                    const c10::optional<Tensor> & bias,
                                    IntArrayRef stride,
                                    IntArrayRef padding,
                                    IntArrayRef dilation,
                                    bool transposed,
                                    IntArrayRef /*output_padding*/,
                                    int64_t groups)
    {
        GUARD;
        TORCH_CHECK(!transposed,"Transposed not implemeted yet");
        dlprim::Tensor X = todp(input.contiguous());
        dlprim::Tensor W = todp(weight);
        dlprim::Tensor B;
        TORCH_CHECK(X.shape().size() == 4,"Invalid input shape");
        TORCH_CHECK(W.shape().size() == 4,"Invalid input shape");
        bool with_bias = bias && bias->numel() != 0;
        if(with_bias) {
            B=todp(*bias);
        }
        dlprim::core::Conv2DSettings cfg = conv_config(X,W,padding,stride,dilation,groups);
        dlprim::ExecutionContext q = getExecutionContext(input);
        dlprim::Context ctx(q);
        auto conv = dlprim::core::Conv2DForward::create(ctx,cfg,with_bias);
        size_t ws_size = conv->workspace();
        at::DataPtr ws_ptr;
        dlprim::Tensor ws = make_workspace(ws_ptr,ws_size,input.device());
        dlprim::Shape rs = dlprim::core::Conv2DForward::get_output_shape(cfg,X.shape());

        torch::Tensor result = new_tensor_as(rs,input);
        dlprim::Tensor Y = todp(result);
        conv->enqueue(X,W,(with_bias ? &B : nullptr),Y,ws,0,q);
        sync_if_needed(input.device());

        return result;
    }



    Tensor & relu_(Tensor & self)
    {
        GUARD;
        dlprim::Tensor X = todp(self);
        dlprim::ExecutionContext q = getExecutionContext(self);
        dlprim::core::activation_forward(X,X,dlprim::StandardActivations::relu,q);
        sync_if_needed(self.device());
        return self;
    }


    Tensor _adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) // {"schema": "aten::_adaptive_avg_pool2d
    {
        GUARD;
        dlprim::Tensor X = todp(self.contiguous());
        int h=X.shape()[2];
        int w=X.shape()[3];
        TORCH_CHECK((output_size[0]==1 && output_size[1]==1) || (output_size[0]==h && output_size[1]==w),"Only global pooling or no-pooling supported");
        Tensor result;
        if(output_size[0]==1 && output_size[1] == 1) {
            result = new_tensor_as(dlprim::Shape(X.shape()[0],X.shape()[1],1,1),self);
            dlprim::Tensor Y = todp(result);

            dlprim::ExecutionContext q = getExecutionContext(self);
            dlprim::Context ctx(q);
            auto pool = dlprim::core::Pooling2DForward::create_global_avg_pooling(ctx,X.shape(),todp(self.dtype()));
            pool->enqueue(X,Y,q);
        }
        else {
            result = new_tensor_as(X.shape(),self);
            dlprim::Tensor Y = todp(result);
            dlprim::core::pointwise_operation({X},{Y},{},"y0=x0;",getExecutionContext(self));
        }
        sync_if_needed(self.device());
        return result;
    }
    
    // {"schema": "aten::_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor _adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self)
    {
        GUARD;
        dlprim::Tensor X = todp(self.contiguous());
        dlprim::Tensor dy = todp(grad_output.contiguous());
        torch::Tensor result;
        TORCH_CHECK((dy.shape()[2]==1 && dy.shape()[3]==1) || (dy.shape() == X.shape()),"Only global pooling or no-pooling supported");
        if(dy.shape()[2]==1 && dy.shape()[3]==1) {
            result = new_tensor_as(X.shape(),self);
            dlprim::Tensor dx = todp(result);

            dlprim::ExecutionContext q = getExecutionContext(self);
            dlprim::Context ctx(q);
            auto pool = dlprim::core::AvgPooling2DBackward::create_global(ctx,X.shape(),todp(self.dtype()));
            pool->enqueue(dx,dy,0,q);
        }
        else {
            result = new_tensor_as(dy.shape(),self);
            dlprim::Tensor dx = todp(result);
            dlprim::core::pointwise_operation({dy},{dx},{},"y0=x0;",getExecutionContext(self));
        }
        sync_if_needed(self.device());
        return result;
    }

    class linear_cls : public torch::autograd::Function<linear_cls> {
    public:
        static Tensor forward(AutogradContext *ctx,const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias)
        {
            GUARD;
            at::AutoDispatchBelowADInplaceOrView g;

            Tensor cinput = input.contiguous();
            dlprim::Tensor X = todp(cinput);
            dlprim::Tensor W = todp(weight);
            dlprim::Shape os(X.shape()[0],W.shape()[0]);
            Tensor result = new_tensor_as(os,input);
            dlprim::Tensor Y = todp(result);
            dlprim::ExecutionContext q = getExecutionContext(input);
            dlprim::Context dlprim_ctx(q);
            dlprim::core::IPSettings cfg;
            cfg.inputs = X.shape().size_no_batch();
            cfg.outputs = W.shape()[0];
            cfg.optimal_batch_size = X.shape()[0];
            cfg.dtype = todp(input.dtype());
            bool has_bias = bias && bias->numel() > 0;
            auto ip = dlprim::core::IPForward::create(dlprim_ctx,cfg,has_bias);
            dlprim::Tensor B;
            if(has_bias)
                B=todp(*bias);
            ip->enqueue(X,W,(has_bias ? &B : nullptr),Y,q);
            ctx->save_for_backward({cinput,weight});
            ctx->saved_data["has_bias"]=has_bias;

            sync_if_needed(input.device());
            return result;
        }
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            GUARD;
            dlprim::Tensor X = todp(ctx->get_saved_variables()[0]);
            dlprim::Tensor W = todp(ctx->get_saved_variables()[1]);
            Tensor dy_tensor = grad_outputs[0].contiguous();
            dlprim::Tensor dY = todp(dy_tensor);
            auto grad_output = grad_outputs[0];

            torch::Tensor dx_tensor = new_tensor_as(X.shape(),dy_tensor);
            dlprim::Tensor dX = todp(dx_tensor);

            torch::Tensor dW_tensor = new_tensor_as(W.shape(),dy_tensor);
            dlprim::Tensor dW = todp(dW_tensor);

            dlprim::core::IPSettings cfg;
            cfg.inputs = X.shape().size_no_batch();
            cfg.outputs = W.shape()[0];
            cfg.optimal_batch_size = X.shape()[0];
            cfg.dtype = todp(dx_tensor.dtype());

            auto q = getExecutionContext(dy_tensor);
            dlprim::Context dlprim_ctx(q);

            auto bwd_data = dlprim::core::IPBackwardData::create(dlprim_ctx,cfg);
            bwd_data->enqueue(dX,W,dY,0,q);
            auto bwd_filter = dlprim::core::IPBackwardFilter::create(dlprim_ctx,cfg);
            bwd_filter->enqueue(X,dW,dY,0,q);
            bool has_bias = ctx->saved_data["has_bias"].toBool();
            torch::Tensor dB_tensor;
            if(has_bias) {
                dB_tensor = new_tensor_as(dlprim::Shape(W.shape()[0]),dy_tensor);
                dlprim::Tensor dB=todp(dB_tensor);
                auto bwd_bias = dlprim::core::BiasBackwardFilter::create(dlprim_ctx,dY.shape(),cfg.dtype);
                at::DataPtr ptr;
                dlprim::Tensor ws = make_workspace(ptr,bwd_bias->workspace(),dy_tensor.device());
                bwd_bias->enqueue(dY,dB,ws,0,q);
            }

            sync_if_needed(grad_output.device());
            return {dx_tensor,dW_tensor,dB_tensor};
        }
    };
    Tensor linear(const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias)
    {
        GUARD;
        return linear_cls::apply(input,weight,bias);
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

    // {"schema": "aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & _log_softmax_out(const Tensor & self, int64_t dim, bool /*half_to_float*/, Tensor & out)
    {
        GUARD;
        TORCH_CHECK(dim==1,"Only case dim=1 is supported currently");
        dlprim::Tensor x=todp(self.contiguous());
        dlprim::Tensor y=todp(out);
        dlprim::core::softmax_forward(x,y,true,getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))", "dispatch": "True", "default": "False"}
    ::std::tuple<Tensor &,Tensor &> nll_loss_forward_out(const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, Tensor & output, Tensor & total_weight)
    {
        GUARD;
        TORCH_CHECK(!weight || weight->numel()==0,"Weight NLLLoss isn't supported");
        TORCH_CHECK(ignore_index <0,"Ignore index isn't supported");
        dlprim::Tensor x=todp(self.contiguous());
        dlprim::Tensor lbl=todp(target.contiguous());
        dlprim::Tensor y=todp(output);
        bool reduce = false;
        float scale = 1;
        switch(reduction) {
        case 0: reduce=false; break; // None
        case 1: reduce=true; scale = 1.0f/x.shape()[0]; break; // Mean
        case 2: reduce=true; break; // sum
        }
        dlprim::core::nll_loss_forward(x,lbl,y,reduce,scale,getExecutionContext(self));
        sync_if_needed(self.device());
        return std::tuple<Tensor &,Tensor &>(output,total_weight);
    }

    // {"schema": "aten::nll_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & nll_loss_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, const Tensor & /*total_weight*/, Tensor & grad_input)
    {
        GUARD;
        TORCH_CHECK(!weight || weight->numel()==0,"Weight NLLLoss isn't supported");
        TORCH_CHECK(ignore_index <0,"Ignore index isn't supported");
        dlprim::Tensor dx=todp(grad_input);
        dlprim::Tensor lbl=todp(target.contiguous());
        dlprim::Tensor dy=todp(grad_output.contiguous());
        bool reduce = false;
        float scale = 1;
        switch(reduction) {
        case 0: reduce=false; break; // None
        case 1: reduce=true; scale = 1.0f/dy.shape()[0]; break; // Mean
        case 2: reduce=true; break; // sum
        }
        dlprim::core::nll_loss_backward(dx,lbl,dy,reduce,scale,0.0f,getExecutionContext(self));
        sync_if_needed(self.device());
        return grad_input;
    }

    // {"schema": "aten::_log_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & _log_softmax_backward_data_out(const Tensor & grad_output, const Tensor & output, int64_t dim, ScalarType /*input_dtype*/, Tensor & out)
    {
        GUARD;
        dlprim::Tensor dx = todp(out);
        dlprim::Tensor y = todp(output.contiguous());
        dlprim::Tensor dy = todp(grad_output.contiguous());
        TORCH_CHECK(dim==1,"Only dim=1 is supported");

        dlprim::core::softmax_backward(dx,y,dy,true,0.0f,getExecutionContext(grad_output));
        sync_if_needed(grad_output.device());
        return out;
    }
    
    // {"schema": "aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)", "dispatch": "True", "default": "True"}
    ::std::tuple<Tensor,Tensor,Tensor> convolution_backward_overrideable(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef /*output_padding*/, int64_t groups, ::std::array<bool,3> output_mask)
    {
        GUARD;
        TORCH_CHECK(!transposed,"Transposed conv not implemented yet");
        dlprim::Tensor dy = todp(grad_output.contiguous());
        dlprim::Tensor x  = todp(input.contiguous());
        dlprim::Tensor W  = todp(weight);
        dlprim::core::Conv2DSettings cfg = conv_config(x,W,padding,stride,dilation,groups);
        dlprim::ExecutionContext q = getExecutionContext(input);
        dlprim::Context ctx(q);

        size_t ws_size = 0;
        std::unique_ptr<dlprim::core::Conv2DBackwardData> bwd_data;
        std::unique_ptr<dlprim::core::Conv2DBackwardFilter> bwd_filter;
        std::unique_ptr<dlprim::core::BiasBackwardFilter> bwd_bias;

        torch::Tensor data_diff,filter_diff,bias_diff;

        if(output_mask[0]) {
            bwd_data = std::move(dlprim::core::Conv2DBackwardData::create(ctx,cfg)); 
            ws_size = std::max(ws_size,bwd_data->workspace());
        }
        if(output_mask[1]) {
            bwd_filter = std::move(dlprim::core::Conv2DBackwardFilter::create(ctx,cfg)); 
            ws_size = std::max(ws_size,bwd_filter->workspace());
        }
        if(output_mask[2]) {
            bwd_bias = std::move(dlprim::core::BiasBackwardFilter::create(ctx,dy.shape(),dy.dtype()));
            ws_size = std::max(ws_size,bwd_bias->workspace());
        }
        at::DataPtr ws_ptr;
        dlprim::Tensor ws = make_workspace(ws_ptr,ws_size,input.device());

        if(output_mask[0]) {
            data_diff = new_tensor_as(x.shape(),input);
            dlprim::Tensor dx = todp(data_diff);
            bwd_data->enqueue(dx,W,dy,ws,0,q);
        }

        if(output_mask[1]) {
            filter_diff = new_tensor_as(W.shape(),weight);
            dlprim::Tensor dW = todp(filter_diff);
            bwd_filter->enqueue(x,dW,dy,ws,0,q);
        }

        if(output_mask[2]) {
            bias_diff = new_tensor_as(dlprim::Shape(dy.shape()[1]),weight);
            dlprim::Tensor dB = todp(bias_diff);
            bwd_bias->enqueue(dy,dB,ws,0,q);
        }
        
        sync_if_needed(grad_output.device());

        return std::tuple<torch::Tensor,torch::Tensor,torch::Tensor>(data_diff,filter_diff,bias_diff);
    }

    template<dlprim::StandardActivations Act>
    class act_cls : public torch::autograd::Function<act_cls<Act> > {
    public:
        static torch::Tensor forward(AutogradContext *ctx, torch::Tensor x) 
        {
            GUARD;
            at::AutoDispatchBelowADInplaceOrView g;
            
            dlprim::Tensor X = todp(x.contiguous());
            torch::Tensor result = new_tensor_as(X.shape(),x);
            ctx->save_for_backward({result});
            dlprim::Tensor Y = todp(result);
            dlprim::ExecutionContext q = getExecutionContext(x);
            dlprim::core::activation_forward(X,Y,Act,q);
            sync_if_needed(x.device());
            return result;
        }
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            GUARD;
            auto grad_output = grad_outputs[0].contiguous();
            torch::Tensor result = ctx->get_saved_variables()[0];
            dlprim::Tensor dy=todp(grad_output);
            dlprim::Tensor y=todp(result);
            torch::Tensor grad_input = new_tensor_as(dy.shape(),grad_output);
            dlprim::Tensor dx = todp(grad_input);
            dlprim::core::activation_backward(dx,dy,y,Act,0.0,getExecutionContext(grad_output));
            sync_if_needed(grad_output.device());
            return {grad_input};
        }
    };

    template<dlprim::StandardActivations Act>
    torch::Tensor act_autograd(torch::Tensor const &x) {
        GUARD;
        return act_cls<Act>::apply(x);
    }

#if 0
    // Don't know how to fix it yet
    template<dlprim::StandardActivations Act>
    class act_inplace_cls : public torch::autograd::Function<act_inplace_cls<Act> > {
    public:
        static torch::Tensor &forward(AutogradContext *ctx, torch::Tensor &x) 
        {
            at::AutoDispatchBelowADInplaceOrView g;
            TORCH_CHECK(x.is_contiguous(),"OpenCL requireds contiguous output");
            dlprim::Tensor X = todp(x);
            ctx->save_for_backward({x});
            dlprim::ExecutionContext q = getExecutionContext(x);
            dlprim::core::activation_forward(X,X,Act,q);
            sync_if_needed(x.device());
            return x;
        }
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            TORCH_CHECK(grad_outputs[0].is_contiguous(),"OpenCL requireds contiguous output");
            auto &grad_output = grad_outputs[0];
            torch::Tensor result = ctx->get_saved_variables()[0];
            dlprim::Tensor dy=todp(grad_output);
            dlprim::Tensor y=todp(result);
            dlprim::core::activation_backward(dy,dy,y,Act,0.0,getExecutionContext(grad_output));
            sync_if_needed(grad_output.device());
            return {grad_output};
        }
    };

    template<dlprim::StandardActivations Act>
    torch::Tensor &act_inplace_autograd(torch::Tensor &x) {
        return act_inplace_cls<Act>::apply(x);
    }

#endif



    class max_pool2d_cls : public torch::autograd::Function<max_pool2d_cls> {
    public:
        static torch::Tensor forward(AutogradContext *ctx,torch::Tensor const &self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) 
        {
            GUARD;
            at::AutoDispatchBelowADInplaceOrView g;

            TORCH_CHECK(kernel_size.size()==2,"Invalid sizes");
            TORCH_CHECK(dilation[0]==1 && dilation[1]==1,"Dilation != 1 is not implemented yet");
            TORCH_CHECK(ceil_mode==false,"ceil mode=true not implemeted yet");
            int kernel[2]={int(kernel_size[0]),int(kernel_size[1])};
            int pad[2]={int(padding[0]),int(padding[1])};
            int strd[2]={1,1};
            if(stride.size()!=0) {
                TORCH_CHECK(stride.size()==2);
                strd[0] = stride[0];
                strd[1] = stride[1];
            }
            else {
                strd[0] = kernel[0];
                strd[1] = kernel[1];
            }

            torch::Tensor self_cont = self.contiguous();
            dlprim::Tensor X = todp(self_cont);
            dlprim::Shape x_shape = X.shape();
            dlprim::Shape y_shape = dlprim::Shape(
                    x_shape[0],
                    x_shape[1],
                    (x_shape[2]+ 2 * pad[0] - kernel[0]) / strd[0] + 1,
                    (x_shape[3]+ 2 * pad[1] - kernel[1]) / strd[1] + 1);

            torch::Tensor out = new_tensor_as(y_shape,self);

            dlprim::Tensor Y = todp(out);
            dlprim::ExecutionContext q = getExecutionContext(self);
            dlprim::Context dlprim_ctx(q);
            auto pool = dlprim::core::Pooling2DForward::create_max_pooling(dlprim_ctx,kernel,pad,strd,todp(self.dtype()));
            pool->enqueue(X,Y,q);
            sync_if_needed(self.device());
            
            ctx->save_for_backward({self_cont});
            ctx->saved_data["kernel_0"]=kernel[0];
            ctx->saved_data["kernel_1"]=kernel[1];
            ctx->saved_data["pad_0"]=pad[0];
            ctx->saved_data["pad_1"]=pad[1];
            ctx->saved_data["strd_0"]=strd[0];
            ctx->saved_data["strd_1"]=strd[1];

            return out;
        }
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            GUARD;
            torch::Tensor grad_output = grad_outputs[0];
            torch::Tensor input = ctx->get_saved_variables()[0];
            int kernel[2],pad[2],strd[2];
            kernel[0] = ctx->saved_data["kernel_0"].toInt();
            kernel[1] = ctx->saved_data["kernel_1"].toInt();

            pad[0] = ctx->saved_data["pad_0"].toInt();
            pad[1] = ctx->saved_data["pad_1"].toInt();

            strd[0] = ctx->saved_data["strd_0"].toInt();
            strd[1] = ctx->saved_data["strd_1"].toInt();

            dlprim::Tensor dy=todp(grad_output);
            dlprim::Tensor x=todp(input);
            torch::Tensor grad_input = new_tensor_as(x.shape(),grad_output);
            dlprim::Tensor dx = todp(grad_input);

            dlprim::ExecutionContext q = getExecutionContext(grad_output);
            dlprim::Context dlprim_ctx(q);

            auto pool=dlprim::core::MaxPooling2DBackward::create(dlprim_ctx,kernel,pad,strd,todp(input.dtype()));
            pool->enqueue(x,dx,dy,0,q);
            sync_if_needed(grad_output.device());
            return {grad_input,torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor(),torch::Tensor()};
        }
    };

    // {"schema": "aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor", "dispatch": "False", "default": "True"}
    torch::Tensor max_pool2d_autograd(torch::Tensor const &self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
        GUARD;
        return max_pool2d_cls::apply(self,kernel_size,stride,padding,dilation,ceil_mode);
    }

    Tensor & mul_scalar_(Tensor & self, const Scalar & other)
    {
        GUARD;
        dlprim::Tensor x0=todp(self);
        float scale = other.to<double>();
        dlprim::core::pointwise_operation({x0},{x0},{scale},
                                          "y0 = x0*w0;",
                                          getExecutionContext(self));
        sync_if_needed(self.device());
        return self;
    }
    
    // {"schema": "aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & add_out(const Tensor & self, const Tensor & other, const Scalar & alpha, Tensor & out)
    {
        GUARD;
        dlprim::Tensor x0=todp(self.contiguous());
        dlprim::Tensor y0=todp(out);
        double value=0;
        if(isCPUScalar(other,value)) {
            float w0 = alpha.toDouble() * value;
            dlprim::core::pointwise_operation({x0},{y0},{w0},
                                      "y0 = x0 + w0;",
                                      getExecutionContext(self));
        }
        else {
            dlprim::Tensor x1=todp(other.contiguous());
            float w0 = alpha.toDouble();
            dlprim::core::pointwise_operation_broadcast({x0,x1},{y0},{w0},
                                      "y0 = x0 + x1 * w0;",
                                      getExecutionContext(self));
        }
        
        sync_if_needed(self.device());
        return out;
    }
    
    // {"schema": "aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & addcmul_out(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value, Tensor & out)
    {
        GUARD;
        dlprim::Tensor x0=todp(self.contiguous());
        dlprim::Tensor x1=todp(tensor1.contiguous());
        dlprim::Tensor x2=todp(tensor2.contiguous());
        dlprim::Tensor y0=todp(out);
        float w0 = value.toDouble();
        dlprim::core::pointwise_operation_broadcast({x0,x1,x2},{y0},{w0},
                                      "y0 = x0 + w0 * x1 * x2;",
                                      getExecutionContext(self));
        
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & sqrt_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        dlprim::Tensor x0=todp(self.contiguous());
        dlprim::Tensor y0=todp(out);
        dlprim::core::pointwise_operation({x0},{y0},{},
                                      "y0 = sqrt(x0);",
                                      getExecutionContext(self));
        
        sync_if_needed(self.device());
        return out;

    }

    // {"schema": "aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & div_out(const Tensor & self, const Tensor & other, Tensor & out)
    {
        GUARD;
        dlprim::Tensor x0=todp(self.contiguous());
        dlprim::Tensor y0=todp(out);
        double value=0;
        if(isCPUScalar(other,value)) {
            dlprim::core::pointwise_operation({x0},{y0},{float(1.0f/value)},
                                        "y0 = x0*w0;",
                                        getExecutionContext(self));
        }
        else {
            dlprim::Tensor x1=todp(other.contiguous());
            dlprim::core::pointwise_operation_broadcast({x0,x1},{y0},{},
                                        "y0 = x0/x1;",
                                        getExecutionContext(self));
        }
        
        sync_if_needed(self.device());
        return out;
    }

   
    Tensor & mul_out(const Tensor & self, const Tensor & other, Tensor & out)
    {
        GUARD;
        double scale=0;
        dlprim::Tensor x0=todp(self.contiguous());
        dlprim::Tensor y0=todp(out);

        if(isCPUScalar(other,scale)) {
            dlprim::core::pointwise_operation({x0},{y0},{float(scale)},
                                          "y0 = x0*w0;",
                                          getExecutionContext(self));
        }
        else {
            dlprim::Tensor x1=todp(other.contiguous());
            dlprim::core::pointwise_operation_broadcast({x0,x1},{y0},{},
                                          "y0 = x0*x1;",
                                          getExecutionContext(self));
        }
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & addcdiv_out(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value, Tensor & out)
    {
        GUARD;
        dlprim::Tensor x0 = todp(self.contiguous());
        dlprim::Tensor x1 = todp(tensor1.contiguous());
        dlprim::Tensor x2 = todp(tensor2.contiguous());
        dlprim::Tensor y0 = todp(out);
        float w0 = value.toDouble();
        dlprim::core::pointwise_operation_broadcast({x0,x1,x2},{y0},{w0},
                                      "y0 = x0 + w0 * (x1/x2);",
                                      getExecutionContext(self));

        sync_if_needed(self.device());
        return out;
    }
    // {"schema": "aten::_local_scalar_dense(Tensor self) -> Scalar", "dispatch": "True", "default": "False"}
    Scalar _local_scalar_dense(const Tensor & self)
    {
        GUARD;
        TORCH_CHECK(self.numel()==1);
        dlprim::Tensor x=todp(self);
        float value=0;
        x.to_host(getExecutionContext(self),&value);
        return value;
    }

    // {"schema": "aten::threshold_backward.grad_input(Tensor grad_output, Tensor self, Scalar threshold, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & threshold_backward_out(const Tensor & grad_output, const Tensor & self, const Scalar & threshold, Tensor & grad_input)
    {
        GUARD;
        dlprim::Tensor dy=todp(grad_output.contiguous());
        dlprim::Tensor dx=todp(grad_input);
        dlprim::Tensor Y=todp(self.contiguous());
        float th = threshold.toDouble();
        dlprim::core::pointwise_operation({Y,dy},{dx},{th},"y0 = (x0 > w0) ? x1 : 0;",getExecutionContext(self));
        sync_if_needed(self.device());
        return grad_input;
    }
    // {"schema": "aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & bernoulli_(Tensor & self, double p, c10::optional<Generator> generator)
    {
        GUARD;
        static dlprim::RandomState state(time(0));
        dlprim::Tensor rnd=todp(self);
        size_t rounds = (rnd.shape().total_size() +  dlprim::philox::result_items - 1) / dlprim::philox::result_items;
        auto seed = state.seed();
        auto seq  = state.sequence_bump(rounds);

        dlprim::core::fill_random(rnd,seed,seq,dlprim::core::rnd_bernoulli,p,0,getExecutionContext(self));
        sync_if_needed(self.device());
        return self;
    }

    // {"schema": "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)", "dispatch": "True", "default": "False"}
    ::std::tuple<Tensor,Tensor,Tensor> native_batch_norm(const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double momentum, double eps)
    {
        GUARD;
        bool weight_present = weight && weight->numel()>0; 
        bool bias_present = bias && bias->numel()>0; 
        bool mean_present = running_mean && running_mean->numel() > 0;
        bool var_present = running_var && running_var->numel() > 0;
        TORCH_CHECK(weight_present == bias_present,"Can have affince or not affine but not partial")
        bool affine = weight_present && bias_present;
        TORCH_CHECK(mean_present && var_present,"Running sums are expected to be present")
        dlprim::ExecutionContext q=getExecutionContext(input);
        dlprim::Context ctx(q);

        dlprim::Tensor X = todp(input.contiguous());
        Tensor result = new_tensor_as(X.shape(),input);
        dlprim::Tensor Y = todp(result);
        dlprim::Tensor gamma,beta;
        if(affine) {
            gamma = todp(*weight);
            beta  = todp(*bias);
        }
        dlprim::Tensor mean = todp(*running_mean);
        dlprim::Tensor var  = todp(*running_var);
        Tensor calc_mean_pt,calc_var_pt;
        dlprim::Tensor calc_mean,calc_var;
        
        if(training) {
            calc_mean_pt = new_tensor_as(mean.shape(),*running_mean);
            calc_mean = todp(calc_mean_pt);
            calc_var_pt  = new_tensor_as(var.shape(),*running_var);
            calc_var = todp(calc_var_pt);
        }

        auto bn = dlprim::core::BatchNormFwdBwd::create(ctx,X.shape(),X.dtype());
        size_t ws_size = bn->workspace();
        
        DataPtr tmp;
        dlprim::Tensor ws = make_workspace(tmp,ws_size,input.device());

        dlprim::Tensor fwd_mean,fwd_var;

        if(training) {
            size_t M = X.shape().total_size() / X.shape()[1];
            bn->enqueue_calculate_batch_stats(X,calc_mean,calc_var,ws,q);
            bn->enqueue_update_running_stats(
                            momentum,(1.0f-momentum),
                            calc_mean,mean,
                            (momentum * M) / (M-1),(1.0f-momentum),
                            calc_var,var,
                            ws,q);
            fwd_mean = calc_mean;
            fwd_var  = calc_var;
        }
        else {
            fwd_mean = mean;
            fwd_var  = var;
        }
        if(affine) {
            bn->enqueue_forward_affine(X,Y,gamma,beta,fwd_mean,fwd_var,eps,ws,q);
        }
        else {
            bn->enqueue_forward_direct(X,Y,fwd_mean,fwd_var,eps,ws,q);
        }
        return std::tuple<Tensor,Tensor,Tensor>(result,calc_mean_pt,calc_var_pt);
    }

    // {"schema": "aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)", "dispatch": "True", "default": "False"} 
    ::std::tuple<Tensor,Tensor,Tensor> native_batch_norm_backward(const Tensor & grad_out,
                                                                  const Tensor & input,
                                                                  const c10::optional<Tensor> & weight,
                                                                  const c10::optional<Tensor> & running_mean,
                                                                  const c10::optional<Tensor> & running_var,
                                                                  const c10::optional<Tensor> & save_mean,
                                                                  const c10::optional<Tensor> & save_var,
                                                                  bool train,
                                                                  double eps,
                                                                  ::std::array<bool,3> output_mask)
    {
        GUARD;
        bool weight_present = weight && weight->numel()>0; 
        bool affine = weight_present;
        dlprim::ExecutionContext q=getExecutionContext(input);
        dlprim::Context ctx(q);
        dlprim::Tensor dY = todp(grad_out);
        dlprim::Tensor X = todp(input);
        dlprim::Tensor W;
        if(weight_present)
            W = todp(*weight);
        Tensor x_diff,gamma_diff,beta_diff;

        bool bwd_data=output_mask[0];
        bool bwd_gamma=output_mask[1] && affine;
        bool bwd_beta=output_mask[2] && affine;
        dlprim::Tensor dX,dG,dB;
        if(bwd_data) {
            x_diff = new_tensor_as(X.shape(),input);
            dX = todp(x_diff);
        }
        if(bwd_gamma)  {
            gamma_diff = new_tensor_as(dlprim::Shape(X.shape()[1]),input);
            dG = todp(gamma_diff);
        }
        if(bwd_beta) {
            beta_diff = new_tensor_as(dlprim::Shape(X.shape()[1]),input);
            dB = todp(beta_diff);
        }

        auto bn = dlprim::core::BatchNormFwdBwd::create(ctx,X.shape(),X.dtype());
        size_t ws_size = bn->workspace();
        
        DataPtr tmp;
        dlprim::Tensor ws = make_workspace(tmp,ws_size,input.device());

        dlprim::Tensor mean = train ? todp(*save_mean) : todp(*running_mean);
        dlprim::Tensor var  = train ? todp(*save_var)  : todp(*running_var);

        if(affine) {
            bn->enqueue_backward_affine(
                    train,
                    X,dY,
                    mean,var,
                    W, 
                    (bwd_data  ? &dX : nullptr),
                    0.0,
                    (bwd_gamma ? &dG : nullptr),
                    0.0, 
                    (bwd_beta  ? &dB : nullptr),
                    0.0,
                    eps,
                    ws,q);
        }
        else {
            bn->enqueue_backward_direct(
                    train,
                    X,dY,
                    mean,var,
                    dX,0.0,
                    eps,
                    ws,q);

        }
        sync_if_needed(input.device());
        return std::tuple<torch::Tensor,torch::Tensor,torch::Tensor>(x_diff,gamma_diff,beta_diff);
    }

    std::pair<dlprim::Shape,dlprim::Shape> squeeze_dim(dlprim::Shape s,IntArrayRef dim,bool keepdim)
    {
        GUARD;
        std::vector<size_t> full,squeezed;
        std::vector<int> dims(dim.begin(),dim.end());
        for(auto &axis : dims) {
            if (axis < 0) {
                axis = axis + s.size();
            }
        }
        std::sort(dims.begin(),dims.end());
        int pos = 0;
        for(int i=0;i<s.size();i++) {
            if(pos < int(dims.size()) && i==dims[pos]) {
                full.push_back(1);
                if(keepdim)
                    squeezed.push_back(1);
                pos++;
            }
            else {
                full.push_back(s[i]);
                squeezed.push_back(s[i]);
            }
        }
        TORCH_CHECK(pos == int(dims.size()),"Looks like invalid dims");
        return std::make_pair(dlprim::Shape::from_range(full.begin(),full.end()),
                              dlprim::Shape::from_range(squeezed.begin(),squeezed.end()));
    }

    Tensor & sum_mean_out(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> /*dtype*/, Tensor & out,bool mean)
    {
        GUARD;
        dlprim::Tensor X = todp(self.contiguous());
        auto r = squeeze_dim(X.shape(),dim,keepdim);
        dlprim::Tensor Y = todp(out);
        TORCH_CHECK(r.second == Y.shape(),"Invalid output shape");
        Y.reshape(r.first);

        double scale = mean ? double(Y.shape().total_size()) / double(X.shape().total_size()) : 1;

        auto q = getExecutionContext(self);
        dlprim::Context ctx(q);
        auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
                ctx,
                {X.specs()},{Y.specs()},
                0,dlprim::float_data,
                "y0=x0;",
                "reduce_y0 = 0;",
                "reduce_y0 += y0;");

        WSGuard wsg(op->workspace(),self.device());
        op->enqueue({X},{Y},wsg.ws,{},{scale},{0},q);

        sync_if_needed(self.device());
        return out;
    }


    // {"schema": "aten::mean.out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & mean_out(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out)
    {
        GUARD;
        return sum_mean_out(self,dim,keepdim,dtype,out,true);
    }
    
    // {"schema": "aten::sum.IntList_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & sum_out(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out)
    {
        GUARD;
        return sum_mean_out(self,dim,keepdim,dtype,out,false);
    }




    // {"schema": "aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)", "dispatch": "True", "default": "False"} 
    Tensor & hardtanh_(Tensor & self, const Scalar & min_val, const Scalar & max_val)
    {
        GUARD;
        dlprim::Tensor X=todp(self);
        double w0 = min_val.toDouble();
        double w1 = max_val.toDouble();
        dlprim::core::pointwise_operation({X},{X},{w0,w1},"y0=max(w0,min(w1,x0));",getExecutionContext(self));
        sync_if_needed(self.device());
        return self;
    }

    // {"schema": "aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor hardtanh_backward(const Tensor & grad_output, const Tensor & self, const Scalar & min_val, const Scalar & max_val)
    {
        GUARD;
        dlprim::Tensor dY = todp(grad_output);
        dlprim::Tensor X  = todp(self);
        Tensor result = new_tensor_as(X.shape(),self);
        dlprim::Tensor dx = todp(result);
        double w0 = min_val.toDouble();
        double w1 = max_val.toDouble();
        dlprim::core::pointwise_operation({X,dY},{X},{w0,w1},"y0 = (w0 <= x0 && x0 <= w1) ? x1 : 0;",getExecutionContext(self));
        sync_if_needed(self.device());
        return result;
    }

    // {"schema": "aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor abs(const Tensor & self)
    {
        GUARD;
        dlprim::Tensor x=todp(self.contiguous());
        Tensor out = new_tensor_as(x.shape(),self);
        dlprim::Tensor y=todp(out);
        dlprim::core::pointwise_operation({x},{y},{},"y0 = x0 < 0 ? -x0 : x0;",getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::_cat(Tensor[] tensors, int dim=0) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor _cat(TensorList tensors, int64_t dim)
    {
        GUARD;
        std::vector<dlprim::Tensor> list;
        for(auto const &t:tensors) {
            list.push_back(todp(t.contiguous()));
        }
        size_t total_shape = 0;
        dlprim::Shape ref;
        for(size_t i=0;i<list.size();i++) {
            TORCH_CHECK(0<=dim && dim < list[i].shape().size(),"dim does not match shape")
            if(i==0) {
                ref = list[i].shape();
            }
            else {
                dlprim::Shape s1 = ref, s2 = list[i].shape();
                s1[dim]=1; s2[dim]=1;
                TORCH_CHECK(s1==s2,"Shapes do not match");
            }
            total_shape+=list[i].shape()[dim];
        }
        ref[dim]=total_shape;
        Tensor out = new_tensor_as(ref,tensors[0]);
        dlprim::Tensor Y=todp(out);
        dlprim::ExecutionContext q(getExecutionContext(tensors[0]));
        dlprim::Context ctx(q);
        
        dlprim::core::SliceCopy cp(ctx,todp(tensors[0].dtype()));

        for(size_t i=0,pos=0;i<list.size();i++) {
            size_t slice = list[i].shape()[dim];
            cp.tensor_slice_copy(dim,slice,
                                      Y,pos,
                                      list[i],0,
                                      0.0,q);
            pos += slice;
        }
        sync_if_needed(tensors[0].device());
        return out;
    }

    // {"schema": "aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & avg_pool2d_out(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, Tensor & out)
    {
        GUARD;
        TORCH_CHECK(ceil_mode==false,"Ceil mode=true not implemented");
        TORCH_CHECK(!divisor_override,"Divisor override is not implemented");
        int ker[2] = {int(kernel_size[0]),int(kernel_size[1])};
        int pad[2] = {int(padding[0]),    int(padding[1])};
        int strd[2];
        if(stride.empty()) {
            strd[0]=ker[0];
            strd[1]=ker[1]; 
        }
        else {
            strd[0]=stride[0];
            strd[1]=stride[1];
        };
        dlprim::Tensor X=todp(self.contiguous());
        dlprim::Tensor Y=todp(out);
        dlprim::ExecutionContext q(getExecutionContext(self));
        dlprim::Context ctx(q);
        
        auto pool = dlprim::core::Pooling2DForward::create_avg_pooling(
                        ctx,
                        ker,pad,strd,
                        count_include_pad,todp(self.dtype())
                    );                  
        pool->enqueue(X,Y,q);    
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::avg_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & avg_pool2d_backward_out(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, Tensor & grad_input)
    {
        GUARD;
        TORCH_CHECK(ceil_mode==false,"Ceil mode=true not implemented");
        TORCH_CHECK(!divisor_override,"Divisor override is not implemented");
        int ker[2] = {int(kernel_size[0]),int(kernel_size[1])};
        int pad[2] = {int(padding[0]),    int(padding[1])};
        int strd[2];
        if(stride.empty()) {
            strd[0]=ker[0];
            strd[1]=ker[1]; 
        }
        else {
            strd[0]=stride[0];
            strd[1]=stride[1];
        };
        dlprim::Tensor dY=todp(grad_output.contiguous());
        dlprim::Tensor dX=todp(grad_input);
        dlprim::ExecutionContext q(getExecutionContext(self));
        dlprim::Context ctx(q);
        
        auto pool = dlprim::core::AvgPooling2DBackward::create(
                        ctx,
                        ker,pad,strd,
                        count_include_pad,todp(grad_input.dtype())
                    );                  
        pool->enqueue(dX,dY,0,q);    
        sync_if_needed(self.device());
        return grad_input;
    }

    // {"schema": "aten::hardswish_(Tensor(a!) self) 
    Tensor & hardswish_(Tensor & self)
    {
        GUARD;
        dlprim::Tensor x=todp(self.contiguous());
        dlprim::core::pointwise_operation({x},{x},{},"y0 = x0 <= -3 ? 0 : (x0>=3 ? x0 : x0*(x0+3)/6);",getExecutionContext(self));
        sync_if_needed(self.device());
        return self;
    }
    // {"schema": "aten::hardsigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & hardsigmoid_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        dlprim::Tensor x=todp(self.contiguous());
        dlprim::Tensor y=todp(out);
        dlprim::core::pointwise_operation({x},{y},{},"y0 = x0 <= -3 ? 0 : (x0>=3 ? 1 : x0/6 + 0.5);",getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }
    // {"schema": "aten::hardsigmoid_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & hardsigmoid_backward_out(const Tensor & grad_output, const Tensor & self, Tensor & grad_input)
    {
        GUARD;
        dlprim::Tensor x=todp(self.contiguous());
        dlprim::Tensor dx=todp(grad_input);
        dlprim::Tensor dy=todp(grad_output);

        dlprim::core::pointwise_operation({x,dy},{dx},{},"y0 = (-3 < x0 && x0 < 3) ? x1 / 6 : 0;",getExecutionContext(self));
        sync_if_needed(self.device());
        return grad_input;
    }
    
    // {"schema": "aten::hardswish_backward(Tensor grad_output, Tensor self) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor hardswish_backward(const Tensor & grad_output, const Tensor & self)
    {
        GUARD;
        dlprim::Tensor dy=todp(grad_output);
        Tensor out = new_tensor_as(dy.shape(),grad_output);
        dlprim::Tensor dx=todp(out);
        dlprim::Tensor x =todp(self);
        dlprim::core::pointwise_operation({x,dy},{dx},{},
            R"xxx(
                if (x0 < -3) {
                    y0 = 0;
                } else if (x0 <= 3) {
                    y0 =  x1 * ((x0 / 3) + 0.5);
                } else {
                    y0 = x1;
                }
            )xxx",
            getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & sigmoid_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        dlprim::Tensor x=todp(self.contiguous());
        dlprim::Tensor y=todp(out);
        dlprim::core::activation_forward(x,y,dlprim::StandardActivations::sigmoid,getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }
    // {"schema": "aten::sigmoid(Tensor self) -> Tensor", "dispatch": "True", "default": "True"}
    Tensor sigmoid(const Tensor & self)
    {
        GUARD;
        dlprim::Tensor x=todp(self.contiguous());
        Tensor out = new_tensor_as(x.shape(),self);
        dlprim::Tensor y=todp(out);
        dlprim::core::activation_forward(x,y,dlprim::StandardActivations::sigmoid,getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }
    // {"schema": "aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)", "dispatch": "True", "default": "True"}
    Tensor & sigmoid_(Tensor & self)
    {
        GUARD;
        dlprim::Tensor X=todp(self);
        dlprim::core::activation_forward(X,X,dlprim::StandardActivations::sigmoid,getExecutionContext(self));
        sync_if_needed(self.device());
        return self;
    }
    
    // {"schema": "aten::sigmoid_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & sigmoid_backward_out(const Tensor & grad_output, const Tensor & output, Tensor & grad_input)
    {
        GUARD;
        dlprim::Tensor y=todp(output.contiguous());
        dlprim::Tensor dy=todp(grad_output);
        dlprim::Tensor dx=todp(grad_input);
        dlprim::core::activation_backward(dx,dy,y,dlprim::StandardActivations::sigmoid,0,getExecutionContext(grad_output));
        sync_if_needed(grad_output.device());
        return grad_input;
    }

    // {"schema": "aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & tanh_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        dlprim::Tensor x=todp(self.contiguous());
        dlprim::Tensor y=todp(out);
        dlprim::core::activation_forward(x,y,dlprim::StandardActivations::tanh,getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::silu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & silu_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        dlprim::Tensor x=todp(self.contiguous());
        dlprim::Tensor y=todp(out);
        dlprim::core::pointwise_operation({x},{y},{},"y0 = x0 / (1 + exp(-x0));",getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::silu_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & silu_backward_out(const Tensor & grad_output, const Tensor & self, Tensor & grad_input)
    {
        GUARD;
        dlprim::Tensor x=todp(self.contiguous());
        dlprim::Tensor dy=todp(grad_output.contiguous());
        dlprim::Tensor dx=todp(grad_input);
        dlprim::core::pointwise_operation({x,dy},{dx},{},
            R"xxx(
                y0 = x0 / (1 + exp(-x0));
                y0 = x1 * y0 * ( 1 + x0 * (1 - y0));
            )xxx",
            getExecutionContext(self));
        sync_if_needed(self.device());
        return grad_input;
    }

    // {"schema": "aten::tanh(Tensor self) -> Tensor", "dispatch": "True", "default": "True"}
    Tensor tanh(const Tensor & self)
    {
        GUARD;
        dlprim::Tensor x=todp(self.contiguous());
        Tensor out = new_tensor_as(x.shape(),self);
        dlprim::Tensor y=todp(out);
        dlprim::core::activation_forward(x,y,dlprim::StandardActivations::tanh,getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }
    // {"schema": "aten::tanh_(Tensor(a!) self) -> Tensor(a!)", "dispatch": "True", "default": "True"}
    Tensor & tanh_(Tensor & self)
    {
        GUARD;
        dlprim::Tensor X=todp(self);
        dlprim::core::activation_forward(X,X,dlprim::StandardActivations::tanh,getExecutionContext(self));
        sync_if_needed(self.device());
        return self;
    }





} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
      m.impl("aten::empty.memory_format", &ptdlprim::allocate_empty);
      m.impl("aten::_reshape_alias",&ptdlprim::_reshape_alias);
      m.impl("aten::fill_.Scalar",&ptdlprim::fill_);
      m.impl("aten::zero_",&ptdlprim::zero_);
      m.impl("aten::_copy_from",&ptdlprim::_copy_from);
      m.impl("aten::empty_strided",&ptdlprim::empty_strided);
      m.impl("aten::convolution_overrideable",&ptdlprim::convolution_overrideable);
      m.impl("aten::convolution_backward_overrideable",&ptdlprim::convolution_backward_overrideable);
      m.impl("aten::_adaptive_avg_pool2d",&ptdlprim::_adaptive_avg_pool2d);
      m.impl("aten::as_strided",&ptdlprim::as_strided);
      m.impl("aten::_log_softmax.out",&ptdlprim::_log_softmax_out);
      m.impl("aten::nll_loss_forward.output",&ptdlprim::nll_loss_forward_out);
      m.impl("aten::nll_loss_backward.grad_input",&ptdlprim::nll_loss_backward_out);
      m.impl("aten::_log_softmax_backward_data.out",&ptdlprim::_log_softmax_backward_data_out);
      m.impl("aten::relu_",&ptdlprim::relu_);
      m.impl("aten::mul.out",&ptdlprim::mul_out);
      m.impl("aten::mul_.Scalar",&ptdlprim::mul_scalar_);
      m.impl("aten::add.out",&ptdlprim::add_out);
      m.impl("aten::addcmul.out",&ptdlprim::addcmul_out);
      m.impl("aten::sqrt.out",&ptdlprim::sqrt_out);
      m.impl("aten::div.out",&ptdlprim::div_out);
      m.impl("aten::addcdiv.out",&ptdlprim::addcdiv_out);
      m.impl("aten::_local_scalar_dense",&ptdlprim::_local_scalar_dense);
      m.impl("aten::threshold_backward.grad_input",&ptdlprim::threshold_backward_out);
      m.impl("aten::bernoulli_.float",&ptdlprim::bernoulli_);
      m.impl("aten::_adaptive_avg_pool2d_backward",&ptdlprim::_adaptive_avg_pool2d_backward);
      m.impl("aten::native_batch_norm",&ptdlprim::native_batch_norm);
      m.impl("aten::native_batch_norm_backward",&ptdlprim::native_batch_norm_backward);
      m.impl("aten::mean.out",&ptdlprim::mean_out);
      m.impl("aten::sum.IntList_out",&ptdlprim::sum_out);
      m.impl("aten::hardtanh_",&ptdlprim::hardtanh_);
      m.impl("aten::hardtanh_backward",&ptdlprim::hardtanh_backward);
      m.impl("aten::abs",&ptdlprim::abs);
      m.impl("aten::_cat",&ptdlprim::_cat);
      m.impl("aten::avg_pool2d.out",&ptdlprim::avg_pool2d_out);
      m.impl("aten::avg_pool2d_backward.grad_input",&ptdlprim::avg_pool2d_backward_out);
      m.impl("aten::hardswish_",&ptdlprim::hardswish_);
      m.impl("aten::hardsigmoid.out",&ptdlprim::hardsigmoid_out);
      m.impl("aten::hardsigmoid_backward.grad_input",&ptdlprim::hardsigmoid_backward_out);
      m.impl("aten::view",&ptdlprim::view);
      m.impl("aten::sigmoid.out",&ptdlprim::sigmoid_out);
      m.impl("aten::sigmoid",&ptdlprim::sigmoid);
      m.impl("aten::sigmoid_",&ptdlprim::sigmoid_);
      m.impl("aten::sigmoid_backward.grad_input",&ptdlprim::sigmoid_backward_out);
      m.impl("aten::tanh.out",&ptdlprim::tanh_out);
      m.impl("aten::silu.out",&ptdlprim::silu_out);
      m.impl("aten::silu_backward.grad_input",&ptdlprim::silu_backward_out);
      m.impl("aten::tanh",&ptdlprim::tanh);
      m.impl("aten::tanh_",&ptdlprim::tanh_);
      m.impl("aten::hardswish_backward",&ptdlprim::hardswish_backward);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
      m.impl("aten::linear",&ptdlprim::linear);
      m.impl("aten::relu",&ptdlprim::act_autograd<dlprim::StandardActivations::relu>);
      m.impl("aten::max_pool2d",&ptdlprim::max_pool2d_autograd);
}

