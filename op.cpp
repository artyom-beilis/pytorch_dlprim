#include <torch/torch.h>
#include <ATen/ATen.h>
#include "CLTensor.h"

#include <dlprim/core/ip.hpp>
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
    OCLDevImpl() :
        dt_(DeviceType::OPENCL,0),
        s_(c10::Stream::UNSAFE,dt_,0)
    {
    } 
    virtual DeviceType type() const { return c10::DeviceType::OPENCL; }
    virtual Device exchangeDevice(Device d) const {
        Device prev = dt_;
        dt_ = d;
        return prev;
    }
    virtual Device getDevice() const { return dt_; }
    virtual void setDevice(Device d) const { dt_ = d; }
    virtual void uncheckedSetDevice(Device d) const noexcept  { dt_ = d; }
    virtual c10::Stream getStream(Device d) const noexcept { 
        return c10::Stream(c10::Stream::UNSAFE,d,0);
    }
    virtual c10::Stream getDefaultStream(Device d) const { return getStream(d); }
    virtual c10::Stream exchangeStream(Stream) const noexcept { return getStream(dt_); }
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
    mutable Device dt_;
    mutable Stream s_;
} ocl_impl_instance;

// register backend
c10::impl::DeviceGuardImplRegistrar ocl_impl_reg(c10::DeviceType::OPENCL,&ocl_impl_instance);

    using torch::Tensor;
    
    torch::Tensor allocate_empty(torch::IntArrayRef size, c10::optional<ScalarType> dtype, c10::optional<Layout> /*layout*/, c10::optional<Device> device, c10::optional<bool> /*pin_memory*/, c10::optional<MemoryFormat> /*memory_format*/)
    {
        c10::ScalarType st = dtype ? *dtype : c10::kFloat; 
        c10::Device dev = device ? *device : Device(c10::DeviceType::OPENCL,0);
        return ptdlprim::new_ocl_tensor(size,dev,st);
    }

    /// "aten::empty_strided"
    Tensor empty_strided(torch::IntArrayRef size, torch::IntArrayRef /*stride*/, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory) 
    {
        return allocate_empty(size,dtype,layout,device,pin_memory,c10::nullopt);
    }

    bool isCPUScalar(Tensor const &other,float &value)
    {
        if(other.device() == Device(c10::kCPU) && other.numel()==1) {
            switch(other.dtype().toScalarType()){
            case c10::kFloat:
                value = *static_cast<float const *>(other.data_ptr());
                break;
            case c10::kDouble:
                value = *static_cast<double const *>(other.data_ptr());
                break;
            default:
                TORCH_CHECK(false,"Unsupported cpu data type in operations mul_out");
            }
            return true;
        }
        return false;
    }



    torch::Tensor _reshape_alias(const Tensor & self, c10::IntArrayRef size, c10::IntArrayRef stride)
    {
        torch::Tensor data(self);
        data.getIntrusivePtr()->set_sizes_and_strides(size,stride);
        return data;
    }
    Tensor &fill_(Tensor &self, const c10::Scalar &value)
    {
        dlprim::Tensor t(todp(self));
        auto q = getExecutionContext(self);
        dlprim::core::fill_tensor(t,value.to<double>(),q);
        sync_if_needed(self.device());
        return self;
    }
    Tensor &zero_(Tensor &self)
    {
        dlprim::Tensor t(todp(self));
        dlprim::core::fill_tensor(t,0.0,getExecutionContext(self));
        return self;
    }


    Tensor _copy_from(const Tensor & self, const Tensor & dst, bool /*non_blocking*/)
    {
        if(dst.device().type() == c10::DeviceType::CPU && self.device().type() == c10::DeviceType::OPENCL) {
            dlprim::Tensor t(todp(self));
            auto ec = getExecutionContext(self);
            void *ptr = dst.data_ptr();
            t.to_host(ec,ptr);
        }
        else if(self.device().type() == c10::DeviceType::CPU && dst.device().type() == c10::DeviceType::OPENCL) {
            dlprim::Tensor t(todp(dst));
            auto ec = getExecutionContext(dst);
            void *ptr = self.data_ptr();
            t.to_device(ec,ptr);
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
        TORCH_CHECK(!transposed,"Transposed not implemeted yet");
        dlprim::Tensor X = todp(input);
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
        dlprim::Tensor X = todp(self);
        dlprim::ExecutionContext q = getExecutionContext(self);
        dlprim::core::activation_forward(X,X,dlprim::StandardActivations::relu,q);
        sync_if_needed(self.device());
        return self;
    }


    Tensor _adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) // {"schema": "aten::_adaptive_avg_pool2d
    {
        dlprim::Tensor X = todp(self);
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
        return result;
        sync_if_needed(self.device());
    }
    
    // {"schema": "aten::_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor _adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self)
    {
        dlprim::Tensor X = todp(self);
        dlprim::Tensor dy = todp(grad_output);
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
            at::AutoDispatchBelowADInplaceOrView g;

            dlprim::Tensor X = todp(input);
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
            ctx->save_for_backward({input,weight});
            ctx->saved_data["has_bias"]=has_bias;

            sync_if_needed(input.device());
            return result;
        }
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            dlprim::Tensor X = todp(ctx->get_saved_variables()[0]);
            dlprim::Tensor W = todp(ctx->get_saved_variables()[1]);
            Tensor dy_tensor = grad_outputs[0];
            dlprim::Tensor dY = todp(dy_tensor.contiguous());
            auto grad_output = grad_outputs[0].contiguous();

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
        return linear_cls::apply(input,weight,bias);
    }
#if 0
    Tensor as_strided(const Tensor & self, IntArrayRef size, IntArrayRef /*stride*/, c10::optional<int64_t> /*storage_offset*/)
    {
        Tensor result = new_ocl_tensor(size,self.device(),self.dtype().toScalarType());
        dlprim::Tensor X = todp(self);
        dlprim::Tensor Y = todp(result);
        dlprim::ExecutionContext q = getExecutionContext(self);
        q.queue().enqueueCopyBuffer(X.device_buffer(),Y.device_buffer(),X.device_offset(),Y.device_offset(),X.memory_size(),q.events(),q.event("copy"));
        sync_if_needed(self.device());
        return result;

    }
#endif

    // {"schema": "aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & _log_softmax_out(const Tensor & self, int64_t dim, bool /*half_to_float*/, Tensor & out)
    {
        TORCH_CHECK(dim==1,"Only case dim=1 is supported currently");
        dlprim::Tensor x=todp(self);
        dlprim::Tensor y=todp(out);
        dlprim::core::softmax_forward(x,y,true,getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))", "dispatch": "True", "default": "False"}
    ::std::tuple<Tensor &,Tensor &> nll_loss_forward_out(const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, Tensor & output, Tensor & total_weight)
    {
        TORCH_CHECK(!weight || weight->numel()==0,"Weight NLLLoss isn't supported");
        TORCH_CHECK(ignore_index <0,"Ignore index isn't supported");
        dlprim::Tensor x=todp(self);
        dlprim::Tensor lbl=todp(target);
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
        TORCH_CHECK(!weight || weight->numel()==0,"Weight NLLLoss isn't supported");
        TORCH_CHECK(ignore_index <0,"Ignore index isn't supported");
        dlprim::Tensor dx=todp(grad_input);
        dlprim::Tensor lbl=todp(target);
        dlprim::Tensor dy=todp(grad_output);
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
        dlprim::Tensor dx = todp(out);
        dlprim::Tensor y = todp(output);
        dlprim::Tensor dy = todp(grad_output);
        TORCH_CHECK(dim==1,"Only dim=1 is supported");

        dlprim::core::softmax_backward(dx,y,dy,true,0.0f,getExecutionContext(grad_output));
        sync_if_needed(grad_output.device());
        return out;
    }
    
    // {"schema": "aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)", "dispatch": "True", "default": "True"}
    ::std::tuple<Tensor,Tensor,Tensor> convolution_backward_overrideable(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef /*output_padding*/, int64_t groups, ::std::array<bool,3> output_mask)
    {
        TORCH_CHECK(!transposed,"Transposed conv not implemented yet");
        dlprim::Tensor dy = todp(grad_output);
        dlprim::Tensor x  = todp(input);
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

        return std::tuple<torch::Tensor,torch::Tensor,torch::Tensor>(data_diff,filter_diff,bias_diff);
    }

    template<dlprim::StandardActivations Act>
    class act_cls : public torch::autograd::Function<act_cls<Act> > {
    public:
        static torch::Tensor forward(AutogradContext *ctx, torch::Tensor x) 
        {
            at::AutoDispatchBelowADInplaceOrView g;
            
            dlprim::Tensor X = todp(x);
            torch::Tensor result = new_tensor_as(X.shape(),x);
            ctx->save_for_backward({result});
            dlprim::Tensor Y = todp(result);
            dlprim::ExecutionContext q = getExecutionContext(x);
            dlprim::core::activation_forward(X,Y,Act,q);
            sync_if_needed(x.device());
            return result;
        }
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
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
            TORCH_CHECK(x.is_contiguous(),"OpenCL requireds contigonous output");
            dlprim::Tensor X = todp(x);
            ctx->save_for_backward({x});
            dlprim::ExecutionContext q = getExecutionContext(x);
            dlprim::core::activation_forward(X,X,Act,q);
            sync_if_needed(x.device());
            return x;
        }
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            TORCH_CHECK(grad_outputs[0].is_contiguous(),"OpenCL requireds contigonous output");
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
            dlprim::Tensor X = todp(self);
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
            
            ctx->save_for_backward({self});
            ctx->saved_data["kernel_0"]=kernel[0];
            ctx->saved_data["kernel_1"]=kernel[1];
            ctx->saved_data["pad_0"]=pad[0];
            ctx->saved_data["pad_1"]=pad[1];
            ctx->saved_data["strd_0"]=strd[0];
            ctx->saved_data["strd_1"]=strd[1];

            return out;
        }
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            torch::Tensor grad_output = grad_outputs[0].contiguous();
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
        return max_pool2d_cls::apply(self,kernel_size,stride,padding,dilation,ceil_mode);
    }

    Tensor & mul_scalar_(Tensor & self, const Scalar & other)
    {
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
        dlprim::Tensor x0=todp(self);
        dlprim::Tensor y0=todp(out);
        float value=0;
        if(isCPUScalar(other,value)) {
            float w0 = alpha.toDouble() * value;
            dlprim::core::pointwise_operation({x0},{y0},{w0},
                                      "y0 = x0 + w0;",
                                      getExecutionContext(self));
        }
        else {
            dlprim::Tensor x1=todp(other);
            float w0 = alpha.toDouble();
            dlprim::core::pointwise_operation({x0,x1},{y0},{w0},
                                      "y0 = x0 + x1 * w0;",
                                      getExecutionContext(self));
        }
        
        sync_if_needed(self.device());
        return out;
    }
    
    // {"schema": "aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & addcmul_out(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value, Tensor & out)
    {
        dlprim::Tensor x0=todp(self);
        dlprim::Tensor x1=todp(tensor1);
        dlprim::Tensor x2=todp(tensor2);
        dlprim::Tensor y0=todp(out);
        float w0 = value.toDouble();
        dlprim::core::pointwise_operation({x0,x1,x2},{y0},{w0},
                                      "y0 = x0 + w0 * x1 * x2;",
                                      getExecutionContext(self));
        
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & sqrt_out(const Tensor & self, Tensor & out)
    {
        dlprim::Tensor x0=todp(self);
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
        dlprim::Tensor x0=todp(self);
        dlprim::Tensor y0=todp(out);
        float value=0;
        if(isCPUScalar(other,value)) {
            dlprim::core::pointwise_operation({x0},{y0},{1.0f/value},
                                        "y0 = x0*w0;",
                                        getExecutionContext(self));
        }
        else {
            dlprim::Tensor x1=todp(other);
            dlprim::core::pointwise_operation({x0,x1},{y0},{},
                                        "y0 = x0/x1;",
                                        getExecutionContext(self));
        }
        
        sync_if_needed(self.device());
        return out;
    }

   
    Tensor & mul_out(const Tensor & self, const Tensor & other, Tensor & out)
    {
        float scale=0;
        dlprim::Tensor x0=todp(self);
        dlprim::Tensor y0=todp(out);
        
        if(isCPUScalar(other,scale)) {
            dlprim::core::pointwise_operation({x0},{y0},{scale},
                                          "y0 = x0*w0;",
                                          getExecutionContext(self));
        }
        else {
            dlprim::Tensor x1=todp(other);
            dlprim::core::pointwise_operation({x0,x1},{y0},{},
                                          "y0 = x0*x1;",
                                          getExecutionContext(self));
        }
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & addcdiv_out(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value, Tensor & out)
    {
        dlprim::Tensor x0 = todp(self);
        dlprim::Tensor x1 = todp(tensor1);
        dlprim::Tensor x2 = todp(tensor2);
        dlprim::Tensor y0 = todp(out);
        float w0 = value.toDouble();
        dlprim::core::pointwise_operation({x0,x1,x2},{y0},{w0},
                                      "y0 = x0 + w0 * (x1/x2);",
                                      getExecutionContext(self));

        sync_if_needed(self.device());
        return out;
    }
    // {"schema": "aten::_local_scalar_dense(Tensor self) -> Scalar", "dispatch": "True", "default": "False"}
    Scalar _local_scalar_dense(const Tensor & self)
    {
        TORCH_CHECK(self.numel()==1);
        dlprim::Tensor x=todp(self);
        float value=0;
        x.to_host(getExecutionContext(self),&value);
        return value;
    }

    // {"schema": "aten::threshold_backward.grad_input(Tensor grad_output, Tensor self, Scalar threshold, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & threshold_backward_out(const Tensor & grad_output, const Tensor & self, const Scalar & threshold, Tensor & grad_input)
    {
        dlprim::Tensor dy=todp(grad_output);
        dlprim::Tensor dx=todp(grad_input);
        dlprim::Tensor Y=todp(self);
        float th = threshold.toDouble();
        dlprim::core::pointwise_operation({Y,dy},{dx},{th},"y0 = (x0 > w0) ? x1 : 0;",getExecutionContext(self));
        sync_if_needed(self.device());
        return grad_input;
    }
    // {"schema": "aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & bernoulli_(Tensor & self, double p, c10::optional<Generator> generator)
    {
        static dlprim::RandomState state(time(0));
        dlprim::Tensor rnd=todp(self);
        size_t rounds = (rnd.shape().total_size() +  dlprim::philox::result_items - 1) / dlprim::philox::result_items;
        auto seed = state.seed();
        auto seq  = state.sequence_bump(rounds);

        dlprim::core::fill_random(rnd,seed,seq,dlprim::core::rnd_bernoulli,p,0,getExecutionContext(self));
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
      //m.impl("aten::max_pool2d_with_indices.out",&ptdlprim::max_pool2d_with_indices_out);
      m.impl("aten::_adaptive_avg_pool2d",&ptdlprim::_adaptive_avg_pool2d);
      //m.impl("aten::as_strided",&ptdlprim::as_strided);
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
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
      m.impl("aten::linear",&ptdlprim::linear);
      m.impl("aten::relu",&ptdlprim::act_autograd<dlprim::StandardActivations::relu>);
      //m.impl("aten::relu_",&ptdlprim::act_inplace_autograd<dlprim::StandardActivations::relu>);
      m.impl("aten::max_pool2d",&ptdlprim::max_pool2d_autograd);
}

