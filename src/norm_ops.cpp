#include "CLTensor.h"
#include "utils.h"

#include <dlprim/core/util.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/core/bn.hpp>

#include <iostream>
namespace ptdlprim {

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;
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

        Tensor input_c = input.contiguous();
        dlprim::Tensor X = todp(input_c);
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

    // {"schema": "aten::native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)", "dispatch": "True", "default": "True"}
    std::tuple<Tensor,Tensor,Tensor> native_layer_norm(const Tensor & input, c10::SymIntArrayRef normalized_shape, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, double eps)
    {
        int N = 1;
        for(auto v:normalized_shape) {
            N*=v.expect_int();
        }

        bool weight_present = weight && weight->numel()>0; 
        bool bias_present = bias && bias->numel()>0; 

        dlprim::ExecutionContext q=getExecutionContext(input);
        dlprim::Context ctx(q);

        Tensor input_c = input.contiguous();
        dlprim::Tensor X = todp(input_c);
        TORCH_CHECK(X.shape().total_size() % N == 0,"Invalid input shape");
        int B = X.shape().total_size() / N;
        auto bn_shape = dlprim::Shape(1,B,N);
        auto src_shape = X.shape();
        Tensor result = new_tensor_as(X.shape(),input);
        dlprim::Tensor Y = todp(result);
        X.reshape(bn_shape);
        Y.reshape(bn_shape);


        Tensor calc_mean_pt,calc_var_pt,calc_rstd_pt;
        dlprim::Tensor calc_mean,calc_var,calc_rstd;

        calc_mean_pt = new_tensor_as(dlprim::Shape(B),input);
        calc_mean = todp(calc_mean_pt);
        calc_var_pt  = new_tensor_as(dlprim::Shape(B),input);
        calc_var = todp(calc_var_pt);
        calc_rstd_pt  = new_tensor_as(dlprim::Shape(B),input);
        calc_rstd = todp(calc_rstd_pt);

        auto bn = dlprim::core::BatchNormFwdBwd::create(ctx,X.shape(),X.dtype());
        size_t ws_size = bn->workspace();
        
        DataPtr tmp;
        dlprim::Tensor ws = make_workspace(tmp,ws_size,input.device());

        dlprim::Tensor fwd_mean,fwd_var;

        bn->enqueue_calculate_batch_stats(X,calc_mean,calc_var,ws,q);
        bn->enqueue_forward_get_rstd(X,Y,calc_mean,calc_var,eps,calc_rstd,ws,q);

        Y.reshape(src_shape);
        if(weight_present && bias_present) {
            dlprim::Tensor w = todp(*weight);
            dlprim::Tensor b = todp(*bias);
            dlprim::core::pointwise_operation_broadcast({Y,w,b},{Y},{},
                                      "y0 = x0 * x1 + x2;",
                                      q);

        }
        else if(weight_present) {
            dlprim::Tensor w = todp(*weight);
            dlprim::core::pointwise_operation_broadcast({Y,w},{Y},{},
                                      "y0 = x0 * x1;",
                                      q);
        }
        else if(bias_present) {
            dlprim::Tensor b = todp(*bias);
            dlprim::core::pointwise_operation_broadcast({Y,b},{Y},{},
                                      "y0 = x0 + x1;",
                                      q);
        }
        return std::tuple<Tensor,Tensor,Tensor>(result,calc_mean_pt,calc_rstd_pt);
    }
    // {"schema": "aten::native_layer_norm_backward(Tensor grad_out, Tensor input, SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)", "dispatch": "True", "default": "False"}
    std::tuple<Tensor,Tensor,Tensor> native_layer_norm_backward(
            const Tensor & grad_out,
            const Tensor & input,
            c10::SymIntArrayRef normalized_shape,
            const Tensor & save_mean,
            const Tensor & save_rstd,
            const c10::optional<Tensor> & weight,
            const c10::optional<Tensor> & bias,
            ::std::array<bool,3> output_mask)
    {
        GUARD;
        int N = 1;
        std::vector<int> ns;
        for(auto v:normalized_shape) {
            ns.push_back(v.expect_int());
            N*=v.expect_int();
        }
        dlprim::Shape norm_shape = dlprim::Shape::from_range(ns.begin(),ns.end());


        bool weight_present = weight && weight->numel()>0; 
        bool bias_present = bias && bias->numel() > 0;

        dlprim::ExecutionContext q=getExecutionContext(input);
        dlprim::Context ctx(q);
        Tensor grad_out_c = grad_out.contiguous();
        Tensor input_c = input.contiguous();
        dlprim::Tensor dY = todp(grad_out_c);
        dlprim::Tensor X = todp(input_c);
        auto src_shape = X.shape();

        int B = X.shape().total_size() / N;
        auto bn_shape = dlprim::Shape(1,B,N);
        X.reshape(bn_shape);
        dY.reshape(bn_shape);
        
        dlprim::Tensor W;
        
        if(weight_present) {
            W = todp(*weight);
            W.reshape(dlprim::Shape(N));
        }

        Tensor x_diff,gamma_diff,beta_diff;

        bool bwd_data=output_mask[0];
        bool bwd_gamma=output_mask[1] && weight_present;
        bool bwd_beta=output_mask[2] && bias_present;

        
        dlprim::Tensor dX,dG,dB;
        if(bwd_gamma)  {
            gamma_diff = new_tensor_as(norm_shape,input);
            dG = todp(gamma_diff);
            dG.reshape(dlprim::Shape(N));
            auto mean = todp(save_mean);
            auto rstd = todp(save_rstd);
            mean.reshape(dlprim::Shape(1,B,1));
            rstd.reshape(dlprim::Shape(1,B,1));
            auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
                        ctx,
                        {X.specs(),mean.specs(),rstd.specs(),dY.specs()},{dG.specs()},
                        0,dlprim::float_data,
                        "y0=(x0 - x1)*x2*x3;",
                        "reduce_y0 = 0;",
                        "reduce_y0 += y0;");
            WSGuard wsg(op->workspace(),input.device());
            op->enqueue({X,mean,rstd,dY},{dG},wsg.ws,{},{1},{0},q);
        }
        if(bwd_beta) {
            beta_diff = new_tensor_as(norm_shape,input);
            dB = todp(beta_diff);
            dB.reshape(dlprim::Shape(N));
            auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
                        ctx,
                        {dY.specs()},{dB.specs()},
                        0,dlprim::float_data,
                        "y0=x0;",
                        "reduce_y0 = 0;",
                        "reduce_y0 += y0;");
            WSGuard wsg(op->workspace(),input.device());
            op->enqueue({dY},{dB},wsg.ws,{},{1},{0},q);
        }
        if(bwd_data) {
            x_diff = new_tensor_as(src_shape,input);
            dX = todp(x_diff);
            dX.reshape(bn_shape);
            auto bn = dlprim::core::BatchNormFwdBwd::create(ctx,bn_shape,X.dtype());
            size_t ws_size = bn->workspace();
            
            DataPtr tmp;
            dlprim::Tensor ws = make_workspace(tmp,ws_size,input.device());

            dlprim::Tensor mean = todp(save_mean);
            dlprim::Tensor rstd  = todp(save_rstd);
            dlprim::Tensor dYW_diff = dY;

            if(weight_present) {
                auto pt_dYW_diff = new_tensor_as(dY.shape(),input);
                dYW_diff = todp(pt_dYW_diff);
                dlprim::core::pointwise_operation_broadcast({dY,W},{dYW_diff},{},{},
                        "y0 = x0 * x1;", 
                        q);
            }

            bn->enqueue_backward_rstd(
                    X,dYW_diff,
                    mean,rstd,
                    dX,0.0,
                    ws,q);
        }

        sync_if_needed(input.device());
        return std::tuple<torch::Tensor,torch::Tensor,torch::Tensor>(x_diff,gamma_diff,beta_diff);
    }

} // namespace
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
      m.impl("aten::native_batch_norm",&ptdlprim::native_batch_norm);
      m.impl("aten::native_batch_norm_backward",&ptdlprim::native_batch_norm_backward);
      m.impl("aten::native_layer_norm",&ptdlprim::native_layer_norm);
      m.impl("aten::native_layer_norm_backward",&ptdlprim::native_layer_norm_backward);
}

