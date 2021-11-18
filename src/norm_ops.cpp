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
} // namespace
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
      m.impl("aten::native_batch_norm",&ptdlprim::native_batch_norm);
      m.impl("aten::native_batch_norm_backward",&ptdlprim::native_batch_norm_backward);
}

