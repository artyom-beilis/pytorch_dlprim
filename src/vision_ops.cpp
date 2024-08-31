#include "CLTensor.h"
#include "utils.h"

#include <dlprim/core/util.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/core/ip.hpp>
#include <dlprim/core/bn.hpp>
#include <dlprim/core/conv.hpp>
#include <dlprim/core/interpolate.hpp>
#include <dlprim/core/bias.hpp>
#include <dlprim/core/pool.hpp>
#include <dlprim/gpu/gemm.hpp>

#include <ATen/ops/_native_multi_head_attention_cpu_dispatch.h>

#include <iostream>
namespace ptdlprim {

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;


    using torch::Tensor;

    dlprim::core::Conv2DSettings conv_config(bool transposed,dlprim::Tensor &X,dlprim::Tensor &W,
                    IntArrayRef padding,IntArrayRef stride,IntArrayRef dilation,IntArrayRef output_padding,int groups)
    {
        TORCH_CHECK(stride.size()==2 && padding.size() == 2 && dilation.size() == 2,"Expecting size of parameters=2");
        if(transposed)
            TORCH_CHECK(output_padding.size() == 2,"Expecting transposed size == 2")
        dlprim::Convolution2DConfigBase cfg_base;
        if(!transposed) {
            cfg_base.channels_in = W.shape()[1] * groups;
            cfg_base.channels_out = W.shape()[0];
        }
        else {
            cfg_base.channels_out = W.shape()[1] * groups;
            cfg_base.channels_in = W.shape()[0];
        }
        for(int i=0;i<2;i++) {
            cfg_base.kernel[i] = W.shape()[i+2];
            cfg_base.pad[i] = padding[i];
            cfg_base.stride[i] = stride[i];
            cfg_base.dilate[i] = dilation[i];
            cfg_base.groups = groups;
        }
        if(!transposed) {
            return dlprim::core::Conv2DSettings(cfg_base,X.shape(),X.dtype()); 
        }
        else {
            int op[2] = {int(output_padding[0]),int(output_padding[1])};
            return dlprim::core::Conv2DSettings(cfg_base,dlprim::core::Conv2DBase::get_output_shape_transposed(cfg_base,X.shape(),op),X.dtype());
        }
    }

    Tensor convolution_overrideable(const Tensor & input,
                                    const Tensor & weight,
                                    const c10::optional<Tensor> & bias,
                                    IntArrayRef stride,
                                    IntArrayRef padding,
                                    IntArrayRef dilation,
                                    bool transposed,
                                    IntArrayRef output_padding,
                                    int64_t groups)
    {
        GUARD;
        Tensor X_tmp = input.contiguous();
        dlprim::Tensor X = todp(X_tmp);
        dlprim::Tensor W = todp(weight);
        dlprim::Tensor B;
        TORCH_CHECK(X.shape().size() == 4,"Invalid input shape");
        TORCH_CHECK(W.shape().size() == 4,"Invalid input shape");
        bool with_bias = bias && bias->numel() != 0;
        if(with_bias) {
            B=todp(*bias);
        }

        dlprim::core::Conv2DSettings cfg = conv_config(transposed,X,W,padding,stride,dilation,output_padding,groups);

        dlprim::ExecutionContext q = getExecutionContext(input);
        dlprim::Context ctx(q);
        torch::Tensor result;
        if(!transposed) {
            auto conv = dlprim::core::Conv2DForward::create(ctx,cfg,with_bias);
            WSGuard wsg(conv->workspace(),input.device());

            dlprim::Shape rs = dlprim::core::Conv2DForward::get_output_shape(cfg,X.shape());
            result = new_tensor_as(rs,input);
            dlprim::Tensor Y = todp(result);
            conv->enqueue(X,W,(with_bias ? &B : nullptr),Y,wsg.ws,0,q);
        }
        else {
            int opad[2] = {int(output_padding[0]),int(output_padding[1]) };
            dlprim::Shape rs = dlprim::core::Conv2DBase::get_output_shape_transposed(cfg,X.shape(),opad);

            std::swap(cfg.channels_in,cfg.channels_out);
            auto conv = dlprim::core::Conv2DBackwardData::create(ctx,cfg);
            WSGuard wsg(conv->workspace(),input.device());
            result = new_tensor_as(rs,input);
            dlprim::Tensor Y = todp(result);
            conv->enqueue(Y,W,X,wsg.ws,0,q);
            if(with_bias)
                dlprim::core::add_bias(Y,B,q);
        }
        sync_if_needed(input.device());

        return result;
    }

    // {"schema": "aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)", "dispatch": "True", "default": "True"}
    ::std::tuple<Tensor,Tensor,Tensor> convolution_backward_overrideable(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, ::std::array<bool,3> output_mask)
    {
        GUARD;
        Tensor grad_output_c = grad_output.contiguous(), input_c = input.contiguous();
        dlprim::Tensor dy = todp(grad_output_c);
        dlprim::Tensor x  = todp(input_c);
        dlprim::Tensor W  = todp(weight);
        dlprim::core::Conv2DSettings cfg = conv_config(transposed,x,W,padding,stride,dilation,output_padding,groups);
        dlprim::ExecutionContext q = getExecutionContext(input);
        dlprim::Context ctx(q);

        size_t ws_size = 0;
        std::unique_ptr<dlprim::core::Conv2DBackwardData> bwd_data;
        std::unique_ptr<dlprim::core::Conv2DForward> bwd_data_tr;
        std::unique_ptr<dlprim::core::Conv2DBackwardFilter> bwd_filter;
        std::unique_ptr<dlprim::core::BiasBackwardFilter> bwd_bias;

        torch::Tensor data_diff,filter_diff,bias_diff;

        if(transposed)
            std::swap(cfg.channels_out,cfg.channels_in);

        if(output_mask[0]) {
            if(!transposed) {
                bwd_data = std::move(dlprim::core::Conv2DBackwardData::create(ctx,cfg)); 
                ws_size = std::max(ws_size,bwd_data->workspace());
            }
            else {
                bwd_data_tr = std::move(dlprim::core::Conv2DForward::create(ctx,cfg,false));
                ws_size = std::max(ws_size,bwd_data_tr->workspace());
            }
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
            if(!transposed)
                bwd_data->enqueue(dx,W,dy,ws,0,q);
            else 
                bwd_data_tr->enqueue(dy,W,nullptr,dx,ws,0,q);
        }

        if(output_mask[1]) {
            filter_diff = new_tensor_as(W.shape(),weight);
            dlprim::Tensor dW = todp(filter_diff);
            if(!transposed)
                bwd_filter->enqueue(x,dW,dy,ws,0,q);
            else
                bwd_filter->enqueue(dy,dW,x,ws,0,q);
        }

        if(output_mask[2]) {
            bias_diff = new_tensor_as(dlprim::Shape(dy.shape()[1]),weight);
            dlprim::Tensor dB = todp(bias_diff);
            bwd_bias->enqueue(dy,dB,ws,0,q);
        }
        
        sync_if_needed(grad_output.device());

        return std::tuple<torch::Tensor,torch::Tensor,torch::Tensor>(data_diff,filter_diff,bias_diff);
    }

    Tensor _adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size) // {"schema": "aten::_adaptive_avg_pool2d
    {
        GUARD;
        Tensor self_c = self.contiguous();
        dlprim::Tensor X = todp(self_c);
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
        Tensor self_c = self.contiguous();
        Tensor grad_output_c = grad_output.contiguous();
        dlprim::Tensor X = todp(self_c);
        dlprim::Tensor dy = todp(grad_output_c);
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
            return linear_forward(ctx,input,weight,bias);
        }
        static Tensor linear_forward(AutogradContext *ctx,const Tensor & input, const Tensor & weight, const c10::optional<Tensor> & bias)
        {
            GUARD;
            at::AutoDispatchBelowADInplaceOrView g;

            Tensor cinput = input.contiguous();
            dlprim::Tensor X = todp(cinput);
            dlprim::Tensor W = todp(weight);
            dlprim::Shape os = X.shape();
            
            int fi = W.shape()[1];
            int fo = W.shape()[0];
            int batch = X.shape().total_size()/fi;
            
            os[os.size()-1] = fo;

            Tensor result = new_tensor_as(os,input);
            dlprim::Tensor Y = todp(result);
            dlprim::ExecutionContext q = getExecutionContext(input);
            dlprim::Context dlprim_ctx(q);
            dlprim::core::IPSettings cfg;
            cfg.inputs = fi;
            cfg.outputs = fo;
            cfg.optimal_batch_size = batch;
            cfg.dtype = todp(input.dtype());
            bool has_bias = bias && bias->numel() > 0;
            auto ip = dlprim::core::IPForward::create(dlprim_ctx,cfg,has_bias);
            dlprim::Tensor B;
            if(has_bias)
                B=todp(*bias);
            X.reshape(dlprim::Shape(batch,fi));
            Y.reshape(dlprim::Shape(batch,fo));
            ip->enqueue(X,W,(has_bias ? &B : nullptr),Y,q);
            ctx->save_for_backward({cinput,weight});
            ctx->saved_data["has_bias"]=has_bias;

            sync_if_needed(input.device());
            return result;
        }
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            return linear_backward(ctx,grad_outputs);
        }
        static tensor_list linear_backward(AutogradContext *ctx, tensor_list grad_outputs) {
            GUARD;
            dlprim::Tensor X = todp(ctx->get_saved_variables()[0]);
            dlprim::Tensor W = todp(ctx->get_saved_variables()[1]);

            int fi = W.shape()[1];
            int fo = W.shape()[0];
            int batch = X.shape().total_size()/fi;

            Tensor dy_tensor = grad_outputs[0].contiguous();
            dlprim::Tensor dY = todp(dy_tensor);
            auto grad_output = grad_outputs[0];

            torch::Tensor dx_tensor = new_tensor_as(X.shape(),dy_tensor);
            dlprim::Tensor dX = todp(dx_tensor);

            torch::Tensor dW_tensor = new_tensor_as(W.shape(),dy_tensor);
            dlprim::Tensor dW = todp(dW_tensor);

            dlprim::core::IPSettings cfg;
            cfg.inputs = fi;
            cfg.outputs = fo;
            cfg.optimal_batch_size = batch;
            cfg.dtype = todp(dx_tensor.dtype());

            auto q = getExecutionContext(dy_tensor);
            dlprim::Context dlprim_ctx(q);

            dlprim::Shape X_shape(batch,fi);
            dlprim::Shape Y_shape(batch,fo);

            X.reshape(X_shape);
            dX.reshape(X_shape);
            dY.reshape(Y_shape);

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

    class max_pool2d_cls : public torch::autograd::Function<max_pool2d_cls> {
    public:
        static torch::Tensor forward(AutogradContext *ctx,torch::Tensor const &self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode)
        {
            return max_pool2d_forward(ctx,self,kernel_size,stride,padding,dilation,ceil_mode);
        }

        static torch::Tensor max_pool2d_forward(AutogradContext *ctx,torch::Tensor const &self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) 
        {
            GUARD;
            at::AutoDispatchBelowADInplaceOrView g;

            TORCH_CHECK(kernel_size.size()==2,"Invalid sizes");
            TORCH_CHECK(dilation[0]==1 && dilation[1]==1,"Dilation != 1 is not implemented yet");
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
                    dlprim::core::calc_pooling_output_size(x_shape[2],kernel[0],pad[0],strd[0],ceil_mode),
                    dlprim::core::calc_pooling_output_size(x_shape[3],kernel[1],pad[1],strd[1],ceil_mode));

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
            return max_pool_2d_backward(ctx,grad_outputs);
        }
        static tensor_list max_pool_2d_backward(AutogradContext *ctx, tensor_list grad_outputs) {
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
            
            Tensor grad_output_c = grad_output.contiguous(),input_c = input.contiguous();
            dlprim::Tensor dy=todp(grad_output_c);
            dlprim::Tensor x=todp(input_c);
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

    // {"schema": "aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & avg_pool2d_out(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool /*ceil_mode*/, bool count_include_pad, c10::optional<int64_t> divisor_override, Tensor & out)
    {
        GUARD;
        TORCH_CHECK(!divisor_override,"Divisor override is not implemented");
        // note ceil mode calculations are based on output size
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
        Tensor self_c = self.contiguous();
        dlprim::Tensor X=todp(self_c);
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
        Tensor grad_output_c = grad_output.contiguous();
        dlprim::Tensor dY=todp(grad_output_c);
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

    static Tensor get_bmm_mm_valid_tensor(Tensor const &t,bool &transposed,int &ld,bool &copied,int bmm,char /*M*/)
    {
        auto sizes= t.sizes();
        auto strides = t.strides();
        TORCH_CHECK(sizes.size() == 2u + bmm,"Invalid input matrix shape");
        TORCH_CHECK(sizes[0+bmm] > 0 && sizes[1+bmm] > 0,"Invalid matrix size");
        copied = false;
        if(t.is_contiguous())  {
            ld = strides[0+bmm];
            transposed = false;
            return t;
        }
        if(strides[1+bmm] >= sizes[0+bmm] && strides[0+bmm] == 1) {
            ld = strides[1+bmm];
            transposed = true;
            return t;
        }
        if(strides[0+bmm] >= sizes[1+bmm] && strides[1+bmm] == 1) {
            ld = strides[0+bmm];
            transposed = false;
            return t;
        }
        transposed = false;
        copied = true;
        ld = sizes[1+bmm];
        return t.contiguous();
    }

    Tensor & mm_bmm_out(const Tensor & self, const Tensor & mat2, Tensor & out,int bmm)
    {
        GUARD;
        Tensor A,B,C;
        bool At=false,Bt=false,Ct=false,Ac,Bc,Cc;
        int lda,ldb,ldc;
        A = get_bmm_mm_valid_tensor(self,At,lda,Ac,bmm,'A');
        B = get_bmm_mm_valid_tensor(mat2,Bt,ldb,Bc,bmm,'B');
        C = get_bmm_mm_valid_tensor(out, Ct,ldc,Cc,bmm,'C');
        if(Ct) {
            Ct = false;
            A=torch::transpose(A,0+bmm,1+bmm);
            B=torch::transpose(B,0+bmm,1+bmm);
            C=torch::transpose(C,0+bmm,1+bmm);
            At = !At;
            Bt = !Bt;
            std::swap(A,B);
            std::swap(lda,ldb);
            std::swap(At,Bt);
        }
        
        int M  = A.sizes()[0+bmm];
        int Ka = A.sizes()[1+bmm];
        int N  = B.sizes()[1+bmm];
        int Kb = B.sizes()[0+bmm];
        int Mc = C.sizes()[0+bmm];
        int Nc = C.sizes()[1+bmm];
        int K = Ka;

        TORCH_CHECK(M==Mc && N==Nc && Ka == Kb,"Invalid matrix sizes "
                    "A(" + std::to_string(M) + ","+std::to_string(Ka)+")" + (At?".T":"  ") + 
                    "*B(" + std::to_string(Kb) + "," + std::to_string(N) +")=" + (Bt?".T":"  ") +
                    "C("+std::to_string(Mc) + ","+std::to_string(Nc)+")");

        TORCH_CHECK(A.dtype() == B.dtype() && A.dtype() == C.dtype(),"All matrices must have same dtype");
        if(bmm) {
            TORCH_CHECK(A.sizes()[0] == B.sizes()[0] && A.sizes()[0] == C.sizes()[0],"Matrices must have same batch i.e. 0 dimention");
        }


        dlprim::ExecutionContext q(getExecutionContext(self));
        dlprim::Context ctx(q);
        
        cl::Buffer Abuf = buffer_from_tensor(A);
        int64_t    Aoff = A.storage_offset();
        cl::Buffer Bbuf = buffer_from_tensor(B);
        int64_t    Boff = B.storage_offset();
        cl::Buffer Cbuf = buffer_from_tensor(C);
        int64_t    Coff = C.storage_offset();

        if(bmm == 0) {
            auto gemm_op = dlprim::gpu::GEMM::get_optimal_gemm(ctx,todp(A.dtype()),At,Bt,M,N,K);
            gemm_op->gemm(M,N,K,
                    Abuf,Aoff,lda,
                    Bbuf,Boff,ldb,
                    Cbuf,Coff,ldc,
                    nullptr,0,0,M*N,q);
        }
        else {
            int batch = A.sizes()[0];
            int step_A = A.strides()[0];
            int step_B = B.strides()[0];
            int step_C = C.strides()[0];
            dlprim::gpu::GEMM::batch_sgemm(todp(A.dtype()),
                At,Bt,
                batch,M,N,K,
                Abuf,Aoff,step_A,lda,
                Bbuf,Boff,step_B,ldb,
                Cbuf,Coff,step_C,ldc,
                0.0f,q);
        }
        if(Cc)
            out.copy_(C);
        sync_if_needed(self.device());
        return out;
    }
     // {"schema": "aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & mm_out(const Tensor & self, const Tensor & mat2, Tensor & out)
    {
        return mm_bmm_out(self,mat2,out,0);
    }
    // {"schema": "aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & bmm_out(const Tensor & self, const Tensor & mat2, Tensor & out)
    {
        return mm_bmm_out(self,mat2,out,1);
    }

    // {"schema": "aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & addmm_out(const Tensor & self, const Tensor & mat1, const Tensor & mat2, const Scalar &sbeta, const Scalar & salpha, Tensor & out)
    {
        GUARD;
        double alpha = salpha.toDouble();
        double beta = sbeta.toDouble();
        if(alpha == 1 && beta == 0) {
            return mm_out(mat1,mat2,out);
        }
        Tensor p = torch::mm(mat1,mat2);
        if(beta == 0) {
            auto x=todp(p);
            if(out.is_contiguous()) {
                auto y=todp(out);
                dlprim::core::pointwise_operation({x},{y},{alpha},
                                          "y0 = x0*w0;",
                                          getExecutionContext(self));
            }
            else {
                dlprim::core::pointwise_operation({x},{x},{alpha},
                                          "y0 = x0*w0;",
                                          getExecutionContext(self));
                out.copy_(p);
            }
        }
        else {
            dlprim::Tensor x = todp(p);
            Tensor self_c = self.contiguous();
            dlprim::Tensor off = todp(self_c);
            dlprim::Tensor tgt;
            if(out.is_contiguous()) {
                tgt = todp(out);
            }
            else {
                tgt = x;
            }
            dlprim::core::pointwise_operation_broadcast({x,off},{tgt},{alpha,beta},
                    "y0 = x0 * w0 + x1 * w1;",
                    getExecutionContext(self));
            if(!out.is_contiguous()) {
                out.copy_(p);
            }
        }
        sync_if_needed(self.device());
        return out;
    }


    // {"schema": "aten::_native_multi_head_attention.out(Tensor query, Tensor key, Tensor value, int embed_dim, int num_head, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, Tensor? mask=None, bool need_weights=True, bool average_attn_weights=True, int? mask_type=None, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))", "dispatch": "True", "default": "True"}
    ::std::tuple<Tensor &,Tensor &> _native_multi_head_attention_out(const Tensor & query, const Tensor & key, const Tensor & value, int64_t embed_dim, int64_t num_head, const Tensor & qkv_weight, const Tensor & qkv_bias, const Tensor & proj_weight, const Tensor & proj_bias, const ::std::optional<Tensor> & mask, bool need_weights, bool average_attn_weights, ::std::optional<int64_t> mask_type, Tensor & out0, Tensor & out1)
    {
        GUARD;
        auto r = at::cpu::_native_multi_head_attention(query,key,value,embed_dim,num_head,qkv_weight,qkv_bias,proj_weight,proj_bias,mask,need_weights,average_attn_weights,mask_type);
        out0.copy_(std::get<0>(r));
        out1.copy_(std::get<1>(r));
        return ::std::tuple<Tensor &,Tensor &>(out0,out1);
    }
    
    // {"schema": "aten::_native_multi_head_attention(Tensor query, Tensor key, Tensor value, int embed_dim, int num_head, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, Tensor? mask=None, bool need_weights=True, bool average_attn_weights=True, int? mask_type=None) -> (Tensor, Tensor)", "dispatch": "True", "default": "False"} 
    ::std::tuple<Tensor,Tensor> _native_multi_head_attention(const Tensor & query, const Tensor & key, const Tensor & value, int64_t embed_dim, int64_t num_head, const Tensor & qkv_weight, const Tensor & qkv_bias, const Tensor & proj_weight, const Tensor & proj_bias, const ::std::optional<Tensor> & mask, bool need_weights, bool average_attn_weights, ::std::optional<int64_t> mask_type)
    {
        GUARD;
        return at::cpu::_native_multi_head_attention(query,key,value,embed_dim,num_head,qkv_weight,qkv_bias,proj_weight,proj_bias,mask,need_weights,average_attn_weights,mask_type);
    }

  
    // {"schema": "aten::_transform_bias_rescale_qkv(Tensor qkv, Tensor qkv_bias, int num_heads) -> (Tensor, Tensor, Tensor)", "dispatch": "True", "default": "False"} 
    // compute q = (q + q_bias) / sqrt(dim_per_head), k = k + k_bias, v = v + v_bias 
    std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv(const Tensor& qkv,const Tensor& qkv_bias,const int64_t num_head) 
    {
        GUARD;
        auto qkv_ = qkv.is_nested()
            ? c10::MaybeOwned<Tensor>::owned(qkv.to_padded_tensor(0))
            : c10::MaybeOwned<Tensor>::borrowed(qkv);
        auto B = qkv_->size(0);
        auto T = qkv_->size(1);
        auto _3D = qkv_->size(2);
        auto D = _3D / 3;
        TORCH_CHECK(D % num_head == 0);
        TORCH_CHECK(_3D % 3 == 0);
        const auto dim_per_head = D / num_head;
        

        const auto qkv_contig = qkv_->expect_contiguous();
        const auto qkv_bias_contig = qkv_bias.expect_contiguous();

        //QKV[5, 7, 66] @ [66]->[3, 5, 2, 7, 11]
         //     B T  3*D   3*dph*3      3  B  nh T dph

        //auto qkv_ in  qkv_contig = ({B, T, 3, num_head, dim_per_head});
        //auto bias_in                     ({3, num_head, dim_per_head});
        auto q_k_v            = at::empty({3, B, num_head, T, dim_per_head}, qkv_->options());
        auto q_k_v_same_order = at::empty({B,T,3,num_head,dim_per_head},qkv_->options());
        // this is how to transpose the result to QKV (see below)
        //            B,T,3,num_head,dim_per_head
        /// {1,2}  -> B,3,T,num_head,dim_per_head
        //  {2,3}  -> B,3,num_head,T,dim_per_head
        //  {0,1} ->  3,B,num_head,T,dim_per_head}
        
        dlprim::Tensor dp_qkv = todp(*qkv_contig);
        dlprim::Tensor dp_bias = todp(*qkv_bias_contig);
        dlprim::Tensor dp_out = todp(q_k_v_same_order);
        dp_qkv.reshape(dlprim::Shape(B*T,_3D));
        dp_out.reshape(dlprim::Shape(B*T,_3D));
        dp_bias.reshape(dlprim::Shape(_3D));
        double scale = 1.0/std::sqrt(double(dim_per_head));
        dlprim::core::pointwise_operation_broadcast(
                {dp_qkv,dp_bias},{dp_out},
                {scale,double(D)},
                {dp_qkv.dtype(),dlprim::int64_data},
                R"xxx(
                    long position_d1 = index.s[1];
                    typeof_x0 scale = position_d1 < w1 ? w0 : 1;
                    y0 = (x0 + x1)*scale;
                )xxx",
                getExecutionContext(qkv),
                false); // don't shrink broadcast to make sure we have correct dims
        Tensor q_k_v_cont_as_q_k_v = q_k_v_same_order;
        q_k_v_cont_as_q_k_v = torch::transpose(q_k_v_cont_as_q_k_v,1,2);
        q_k_v_cont_as_q_k_v = torch::transpose(q_k_v_cont_as_q_k_v,2,3);
        q_k_v_cont_as_q_k_v = torch::transpose(q_k_v_cont_as_q_k_v,0,1);
        q_k_v.copy_(q_k_v_cont_as_q_k_v);
        auto q_k_v_s =
            at::native::split(q_k_v.view({3 * B, num_head, T, dim_per_head}), B, 0);
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(q_k_v_s.size() == 3);
        return std::make_tuple(q_k_v_s[0], q_k_v_s[1], q_k_v_s[2]);
    }

    // {"schema": "aten::upsample_nearest2d.out(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & interpolate_2d_out_internal(const Tensor & self, c10::SymIntArrayRef /*output_size*/, ::std::optional<double> scales_h, ::std::optional<double> scales_w, Tensor & out,dlprim::InterpolateType method,bool align_c=false)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        Tensor out_c = out.contiguous();
        double scale_h = scales_h ? *scales_h : -1;
        double scale_w = scales_w ? *scales_w : -1;
        dlprim::Tensor x = todp(self_c);
        dlprim::Tensor y = todp(out_c);
        dlprim::core::interpolate2d(x,y,scale_h,scale_w,method,align_c,getExecutionContext(self));
        if(!out.is_contiguous())
            out.copy_(out_c);
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::upsample_nearest2d_backward.grad_input(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & interpolate_2d_backward_out_internal(const Tensor & grad_output, c10::SymIntArrayRef /*output_size*/, c10::SymIntArrayRef /*input_size*/, ::std::optional<double> scales_h, ::std::optional<double> scales_w, Tensor & grad_input,dlprim::InterpolateType method,bool align_c=false)
    {
        GUARD;
        Tensor grad_output_c = grad_output.contiguous();
        Tensor grad_input_c  = grad_input.contiguous();
        double scale_h = scales_h ? *scales_h : -1;
        double scale_w = scales_w ? *scales_w : -1;
        dlprim::Tensor dx = todp(grad_input_c);
        dlprim::Tensor dy = todp(grad_output_c);
        dlprim::core::interpolate2d_backward(dx,dy,scale_h,scale_w,method,align_c,0.0,getExecutionContext(grad_output));
        if(!grad_input.is_contiguous())
            grad_input.copy_(grad_input_c);
        sync_if_needed(grad_output.device());
        return grad_input;
    }

    // {"schema": "aten::upsample_nearest2d.out(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & upsample_nearest2d_out(const Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w, Tensor & out)
    {
        GUARD;
        return interpolate_2d_out_internal(self,output_size,scales_h,scales_w,out,dlprim::InterpolateType::nearest);
    }
    // {"schema": "aten::upsample_nearest2d_backward.grad_input(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & upsample_nearest2d_backward_out(const Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w, Tensor & grad_input)
    {
        GUARD;
        return interpolate_2d_backward_out_internal(grad_output,output_size,input_size,scales_h,scales_w,grad_input,dlprim::InterpolateType::nearest);
    }
    
    // {"schema": "aten::_upsample_nearest_exact2d.out(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}  
    Tensor & _upsample_nearest_exact2d_out(const Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w, Tensor & out)
    {
        GUARD;
        return interpolate_2d_out_internal(self,output_size,scales_h,scales_w,out,dlprim::InterpolateType::nearest_exact);
    }
    
    // {"schema": "aten::_upsample_nearest_exact2d_backward.grad_input(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & _upsample_nearest_exact2d_backward_out(const Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w, Tensor & grad_input)
    {
        GUARD;
        return interpolate_2d_backward_out_internal(grad_output,output_size,input_size,scales_h,scales_w,grad_input,dlprim::InterpolateType::nearest_exact);
    }
    
    // {"schema": "aten::upsample_bilinear2d.out(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}    
    Tensor & upsample_bilinear2d_out(const Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w, Tensor & out)
    {
        GUARD;
        return interpolate_2d_out_internal(self,output_size,scales_h,scales_w,out,dlprim::InterpolateType::bilinear,align_corners);
    }
    
    // {"schema": "aten::upsample_bilinear2d_backward.grad_input(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & upsample_bilinear2d_backward_out(const Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w, Tensor & grad_input)
    {
        GUARD;
        return interpolate_2d_backward_out_internal(grad_output,output_size,input_size,scales_h,scales_w,grad_input,dlprim::InterpolateType::bilinear,align_corners);
    }



} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
      m.impl("aten::convolution_overrideable",&ptdlprim::convolution_overrideable);
      m.impl("aten::convolution_backward_overrideable",&ptdlprim::convolution_backward_overrideable);
      m.impl("aten::_adaptive_avg_pool2d",&ptdlprim::_adaptive_avg_pool2d);
      m.impl("aten::_adaptive_avg_pool2d_backward",&ptdlprim::_adaptive_avg_pool2d_backward);
      m.impl("aten::avg_pool2d.out",&ptdlprim::avg_pool2d_out);
      m.impl("aten::avg_pool2d_backward.grad_input",&ptdlprim::avg_pool2d_backward_out);
      m.impl("aten::mm.out",&ptdlprim::mm_out);
      m.impl("aten::bmm.out",&ptdlprim::bmm_out);
      m.impl("aten::_native_multi_head_attention.out",&ptdlprim::_native_multi_head_attention_out);
      m.impl("aten::_native_multi_head_attention",&ptdlprim::_native_multi_head_attention);
      m.impl("aten::addmm.out",&ptdlprim::addmm_out);
      m.impl("aten::_transform_bias_rescale_qkv",&ptdlprim::transform_bias_rescale_qkv);
      m.impl("aten::upsample_nearest2d.out",&ptdlprim::upsample_nearest2d_out);
      m.impl("aten::upsample_nearest2d_backward.grad_input",&ptdlprim::upsample_nearest2d_backward_out);
      m.impl("aten::_upsample_nearest_exact2d.out",&ptdlprim::_upsample_nearest_exact2d_out);
      m.impl("aten::_upsample_nearest_exact2d_backward.grad_input",&ptdlprim::_upsample_nearest_exact2d_backward_out);
      m.impl("aten::upsample_bilinear2d.out",&ptdlprim::upsample_bilinear2d_out);
      m.impl("aten::upsample_bilinear2d_backward.grad_input",&ptdlprim::upsample_bilinear2d_backward_out);
}
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
      m.impl("aten::linear",&ptdlprim::linear);
      m.impl("aten::max_pool2d",&ptdlprim::max_pool2d_autograd);
}
 
