#include "CLTensor.h"
#include "utils.h"

#include <dlprim/core/util.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/core/loss.hpp>

#include <iostream>
namespace ptdlprim {

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;


    using torch::Tensor;

    // {"schema": "aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))", "dispatch": "True", "default": "False"}
    ::std::tuple<Tensor &,Tensor &> nll_loss_forward_out(const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, Tensor & output, Tensor & total_weight)
    {
        GUARD;
        TORCH_CHECK(!weight || weight->numel()==0,"Weight NLLLoss isn't supported");
        TORCH_CHECK(ignore_index <0,"Ignore index isn't supported");
        Tensor self_c = self.contiguous();
        dlprim::Tensor x=todp(self_c);
        Tensor target_c = target.contiguous();
        dlprim::Tensor lbl=todp(target_c);
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
        Tensor target_c = target.contiguous(), grad_output_c = grad_output.contiguous();
        dlprim::Tensor lbl=todp(target_c);
        dlprim::Tensor dy=todp(grad_output_c);
        bool reduce = false;
        float scale = 1;
        switch(reduction) {
        case 0: reduce=false; break; // None
        case 1: reduce=true; scale = 1.0f/dx.shape()[0]; break; // Mean
        case 2: reduce=true; break; // sum
        }
        dlprim::core::nll_loss_backward(dx,lbl,dy,reduce,scale,0.0f,getExecutionContext(self));
        sync_if_needed(self.device());
        return grad_input;
    }
       
     // {"schema": "aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor binary_cross_entropy(const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction)
    {
        GUARD;
        TORCH_CHECK(!weight || weight->numel()==0,"Weight in binar_cross_entroy isn't supported");
        Tensor self_c = self.contiguous();
        Tensor target_c = target.contiguous();
        dlprim::Tensor x = todp(self_c);
        dlprim::Tensor y = todp(target_c);
        bool reduce = false;
        double scale = 1;
        switch(reduction) {
        case 0: reduce=false; break; // None
        case 1: reduce=true; scale = 1.0/x.shape().total_size(); break; // Mean
        case 2: reduce=true; break; // sum
        }
        dlprim::Shape target_shape;
        if(!reduce)
            target_shape = x.shape();
        Tensor loss_tensor = new_tensor_as(target_shape,self_c);
        dlprim::Tensor loss(todp(loss_tensor));
        auto q = getExecutionContext(self);
        dlprim::Context ctx(q);
        auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(ctx,
                    {x.specs(),y.specs()},{loss.specs()},0,dlprim::float_data,
                    "y0 = - (x1 * max((typeof_x0)(-100),log(x0)) + (1-x1) * max((typeof_x0)(-100),log(1-x0)));",
                    "reduce_y0 = 0;",
                    "reduce_y0 += y0;");
        WSGuard wsg(op->workspace(),self.device());
        op->enqueue({x,y},{loss},wsg.ws,{},{scale},{0},q);
        sync_if_needed(self.device());
        return loss_tensor;
    }

    // {"schema": "aten::binary_cross_entropy_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"} 
    Tensor & binary_cross_entropy_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, Tensor & grad_input)
    {
        GUARD;
        TORCH_CHECK(!weight || weight->numel()==0,"Weight in binar_cross_entroy isn't supported");
        Tensor self_c = self.contiguous();
        Tensor target_c = target.contiguous();
        Tensor grad_output_c = grad_output.contiguous();
        dlprim::Tensor x = todp(self_c);
        dlprim::Tensor y = todp(target_c);
        dlprim::Tensor dloss = todp(grad_output_c);
        double scale = 1;
        if(reduction == 1) // mean
            scale = 1.0/x.shape().total_size(); 
        dlprim::Tensor dx = todp(grad_input);

        // -w (y - x) / (x - x^2)
        dlprim::core::pointwise_operation_broadcast({x,y,dloss},{dx},{scale},
                "y0 = -(x1 - x0) / max(1e-12f,x0 - x0*x0) * x2 * w0;",
                getExecutionContext(self));
        sync_if_needed(self.device());
        return grad_input;

    }

    // {"schema": "aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        Tensor input_grad = new_tensor_as(todp(self_c).shape(),self_c);
        binary_cross_entropy_backward_out(grad_output,self_c,target,weight,reduction,input_grad);
        return input_grad;
    }


    static Tensor & impl_softmax_out(const Tensor & self, int64_t dim, bool is_log, Tensor & out)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        dlprim::Tensor x=todp(self_c);
        dlprim::Tensor y=todp(out);
        TORCH_CHECK(dim==-1 || (0<=dim && dim < int(x.shape().size())),"Invalid value of dim");
        if(x.shape().size()!=2) {
            if(dim == -1)
                dim = x.shape().size() - 1;
            int N=1,M=1;
            int Rd = x.shape()[dim];
            for(int i=0;i<dim;i++)
                N*=x.shape()[i];
            for(int i=dim+1;i<int(x.shape().size());i++)
                M*=x.shape()[i];
            auto new_shape = dlprim::Shape(N,Rd,M);
            x.reshape(new_shape);
            y.reshape(new_shape);
        }
        dlprim::core::softmax_forward(x,y,is_log,getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & _log_softmax_out(const Tensor & self, int64_t dim, bool /*half_to_float*/, Tensor & out)
    {
        return impl_softmax_out(self,dim,true,out);
    }
    
    // {"schema": "aten::_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & _softmax_out(const Tensor & self, int64_t dim, bool /*half_to_float*/, Tensor & out)
    {
        return impl_softmax_out(self,dim,false,out);
    }


    Tensor & impl_softmax_backward_data_out(const Tensor & grad_output, const Tensor & output, int64_t dim, bool is_log, Tensor & out)
    {
        GUARD;
        dlprim::Tensor dx = todp(out);
        Tensor output_c = output.contiguous(),grad_output_c = grad_output.contiguous();
        dlprim::Tensor y = todp(output_c);
        dlprim::Tensor dy = todp(grad_output_c);
        TORCH_CHECK(dim==-1 || (0<=dim && dim < int(y.shape().size())),"Invalid value of dim");
        if(y.shape().size()!=2) {
            if(dim == -1)
                dim = y.shape().size() - 1;
            int N=1,M=1;
            int Rd = y.shape()[dim];
            for(int i=0;i<dim;i++)
                N*=y.shape()[i];
            for(int i=dim+1;i<int(y.shape().size());i++)
                M*=y.shape()[i];
            auto new_shape = dlprim::Shape(N,Rd,M);
            dx.reshape(new_shape);
            dy.reshape(new_shape);
            y.reshape(new_shape);
        }

        dlprim::core::softmax_backward(dx,y,dy,is_log,0.0f,getExecutionContext(grad_output));
        sync_if_needed(grad_output.device());
        return out;
    }

    // {"schema": "aten::_log_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & _log_softmax_backward_data_out(const Tensor & grad_output, const Tensor & output, int64_t dim, ScalarType /*input_dtype*/, Tensor & out)
    {
        return impl_softmax_backward_data_out(grad_output,output,dim,true,out);
    }

    // {"schema": "aten::_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & _softmax_backward_data_out(const Tensor & grad_output, const Tensor & output, int64_t dim, ScalarType /*input_dtype*/, Tensor & grad_input)
    {
        return impl_softmax_backward_data_out(grad_output,output,dim,false,grad_input);
    }

    
    // {"schema": "aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor", "dispatch": "True", "default"
    Tensor mse_loss(const Tensor & self, const Tensor & target, int64_t reduction)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        dlprim::Tensor x=todp(self_c);
        Tensor target_c = target.contiguous();
        dlprim::Tensor lbl=todp(target_c);
        bool reduce = false;
        float scale = 1;
        switch(reduction) {
        case 0: reduce=false; break; // None
        case 1: reduce=true; scale = 1.0f/x.shape().total_size(); break; // Mean
        case 2: reduce=true; break; // sum
        }
        Tensor output = new_tensor_as(reduce ? dlprim::Shape() : x.shape(),self_c);
        dlprim::Tensor y=todp(output);
        auto q = getExecutionContext(self);
        dlprim::Context ctx(q);
        auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(ctx,
                    {x.specs(),lbl.specs()},{y.specs()},0,x.dtype(),
                    "y0 = (x0-x1)*(x0-x1);",
                    "reduce_y0 = 0;",
                    "reduce_y0 += y0;");
        WSGuard wsg(op->workspace(),self.device());
        op->enqueue({x,lbl},{y},wsg.ws,{},{scale},{0},q);
        sync_if_needed(self.device());

        return output;
    }
    // {"schema": "aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction)
    Tensor mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction)
    {
        GUARD;
        Tensor grad_output_c = grad_output.contiguous();
        Tensor self_c = self.contiguous();
        Tensor target_c = target.contiguous();
        dlprim::Tensor x = todp(self_c);
        dlprim::Tensor dy = todp(grad_output_c);
        dlprim::Tensor lbl = todp(target_c);
        Tensor result = new_tensor_as(x.shape(),self_c);
        dlprim::Tensor dx = todp(result);
        double scale = reduction == 1 ? (1.0f/x.shape().total_size()) : 1.0;
        dlprim::core::pointwise_operation_broadcast({dy,x,lbl},{dx},{scale},
            "y0 = 2*(x1 -x2) * x0 * w0;",getExecutionContext(self.device()));
        sync_if_needed(self.device());
        return result;
    }


} // namespace dlprim

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
      m.impl("aten::nll_loss_forward.output",&ptdlprim::nll_loss_forward_out);
      m.impl("aten::nll_loss_backward.grad_input",&ptdlprim::nll_loss_backward_out);
      m.impl("aten::binary_cross_entropy",&ptdlprim::binary_cross_entropy);
      m.impl("aten::binary_cross_entropy_backward",&ptdlprim::binary_cross_entropy_backward);
      m.impl("aten::binary_cross_entropy_backward.grad_input",&ptdlprim::binary_cross_entropy_backward_out);
      m.impl("aten::_log_softmax.out",&ptdlprim::_log_softmax_out);
      m.impl("aten::_log_softmax_backward_data.out",&ptdlprim::_log_softmax_backward_data_out);
      m.impl("aten::_softmax.out",&ptdlprim::_softmax_out);
      m.impl("aten::_softmax_backward_data.out",&ptdlprim::_softmax_backward_data_out);
      m.impl("aten::mse_loss",&ptdlprim::mse_loss);
      m.impl("aten::mse_loss_backward",&ptdlprim::mse_loss_backward);
} 
