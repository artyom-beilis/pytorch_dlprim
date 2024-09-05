#include <torch/torch.h>
#include <ATen/ATen.h>
#include "CLTensor.h"
#include "utils.h"

#include <dlprim/core/ip.hpp>
#include <dlprim/core/bn.hpp>
#include <dlprim/core/util.hpp>
#include <dlprim/core/conv.hpp>
#include <dlprim/core/bias.hpp>
#include <dlprim/core/pool.hpp>
#include <dlprim/core/loss.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/core/activation.hpp>

#include <iostream>
namespace ptdlprim {

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;

    bool is_integer(Tensor const &t,bool include_bool)
    {
        return c10::isIntegralType(t.dtype().toScalarType(),include_bool);
    }


    bool isCPUScalar(Tensor const &other, double &value)
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


    Tensor & relu_(Tensor & self)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        dlprim::Tensor X = todp(self_c);
        dlprim::ExecutionContext q = getExecutionContext(self);
        dlprim::core::activation_forward(X,X,dlprim::StandardActivations::relu,q);
        
        if (!self.is_contiguous())
            self.copy_(self_c);
        
        sync_if_needed(self.device());
        return self;
    }
    
    template<dlprim::StandardActivations Act>
    class act_cls : public torch::autograd::Function<act_cls<Act> > {
    public:
        static torch::Tensor std_activation_forward(AutogradContext *ctx, torch::Tensor x) 
        {
            GUARD;
            at::AutoDispatchBelowADInplaceOrView g;
           
            Tensor x_c = x.contiguous(); 
            dlprim::Tensor X = todp(x_c);
            torch::Tensor result = new_tensor_as(X.shape(),x);
            ctx->save_for_backward({result});
            dlprim::Tensor Y = todp(result);
            dlprim::ExecutionContext q = getExecutionContext(x);
            dlprim::core::activation_forward(X,Y,Act,q);
            sync_if_needed(x.device());
            return result;
        }
        static torch::Tensor forward(AutogradContext *ctx, torch::Tensor x) 
        {
            return std_activation_forward(ctx,x);
        }
        static tensor_list std_activation_backward(AutogradContext *ctx, tensor_list grad_outputs) {
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
        static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
            return std_activation_backward(ctx,grad_outputs);
        }
    };

    template<dlprim::StandardActivations Act>
    torch::Tensor act_autograd(torch::Tensor const &x) {
        GUARD;
        return act_cls<Act>::apply(x);
    }

    Tensor & mul_scalar_(Tensor & self, const Scalar & other)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        dlprim::Tensor x0=todp(self_c);
        float scale = other.to<double>();
        dlprim::core::pointwise_operation({x0},{x0},{scale},
                                          "y0 = x0*w0;",
                                          getExecutionContext(self));
        
        if (!self.is_contiguous())
            self.copy_(self_c);
        
        sync_if_needed(self.device());
        return self;
    }
    
    // {"schema": "aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & add_out(const Tensor & self, const Tensor & other, const Scalar & alpha, Tensor & out)
    {
        GUARD;
        Tensor out_c = out.contiguous();
        dlprim::Tensor y0=todp(out_c);
        double value=0;
        auto dev_to_sync = self.device();
        if(isCPUScalar(other,value)) {
            Tensor self_c = self.contiguous();
            dlprim::Tensor x0=todp(self_c);
            float w0 = alpha.toDouble() * value;
            dlprim::core::pointwise_operation({x0},{y0},{w0},
                                      "y0 = x0 + w0;",
                                      getExecutionContext(self));
        }
        else if(isCPUScalar(self,value)) {
            dev_to_sync = other.device();
            Tensor other_c = other.contiguous();
            dlprim::Tensor x0=todp(other_c);
            float w0 = value;
            float w1 = alpha.toDouble();
            dlprim::core::pointwise_operation({x0},{y0},{w0,w1},
                                      "y0 = w0 + x0 * w1;",
                                      getExecutionContext(other));
        }
        else {
            Tensor self_c = self.contiguous();
            dlprim::Tensor x0=todp(self_c);
            Tensor other_c = other.contiguous();
            dlprim::Tensor x1=todp(other_c);
            float w0 = alpha.toDouble();
            dlprim::core::pointwise_operation_broadcast({x0,x1},{y0},{w0},
                                      "y0 = x0 + x1 * w0;",
                                      getExecutionContext(self));
        }
        if (!out.is_contiguous())
            out.copy_(out_c);
        sync_if_needed(dev_to_sync);
        return out;
    }

    Tensor &unitary_op(const Tensor &self,Tensor &out,std::string const &op)
    {
        GUARD;
        Tensor self_c = self.contiguous(), out_c = out.contiguous();
        
        dlprim::Tensor x=todp(self_c);
        dlprim::Tensor y=todp(out_c);
        dlprim::core::pointwise_operation({x},{y},{},op,getExecutionContext(self));
        
        if (!out.is_contiguous())
            out.copy_(out_c);
        
        sync_if_needed(self.device());
        return out;
    }
    

    // {"schema": "aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & exp_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        return unitary_op(self,out,"y0 = exp(x0);");
    }

    // {"schema": "aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & log_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        return unitary_op(self,out,"y0 = log(x0);");
    }

    // {"schema": "aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & sub_out(const Tensor & self, const Tensor & other, const Scalar & alpha, Tensor & out)
    {
        GUARD;
        return add_out(self,other,Scalar(alpha.toDouble()*-1),out);
    }

    
    // {"schema": "aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & addcmul_out(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value, Tensor & out)
    {
        GUARD;
        Tensor self_c = self.contiguous(), out_c = out.contiguous(),
               tensor1_c = tensor1.contiguous(), tensor2_c = tensor2.contiguous();
        
        dlprim::Tensor x0=todp(self_c);
        dlprim::Tensor x1=todp(tensor1_c);
        dlprim::Tensor x2=todp(tensor2_c);
        dlprim::Tensor y0=todp(out_c);
        float w0 = value.toDouble();
        dlprim::core::pointwise_operation_broadcast({x0,x1,x2},{y0},{w0},
                                      "y0 = x0 + w0 * x1 * x2;",
                                      getExecutionContext(self));
        
        if (!out.is_contiguous())
            out.copy_(out_c);
        
        sync_if_needed(self.device());
        return out;
    }
    
    Tensor & comp_out(const Tensor & self, const Scalar & other, Tensor & out,std::string const &op)
    {
        GUARD;
        Tensor self_c = self.contiguous(), out_c = out.contiguous();
        
        dlprim::Tensor x0=todp(self_c);
        dlprim::Tensor y0=todp(out_c);
        float w0 = other.toDouble();
        dlprim::core::pointwise_operation_broadcast({x0},{y0},{w0},{dlprim::float_data},
                                      "y0 = (x0 " + op + " w0) ? 1 : 0;" ,
                                      getExecutionContext(self));
        
        if (!out.is_contiguous())
            out.copy_(out_c);
        
        sync_if_needed(self.device());
        return out;
    }
    // {"schema": "aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & le_out(const Tensor & self, const Scalar & other, Tensor & out)
    {
        return comp_out(self,other,out,"<=");
    }
    // {"schema": "aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & ge_out(const Tensor & self, const Scalar & other, Tensor & out)
    {
        return comp_out(self,other,out,">=");
    }

    // {"schema": "aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & lt_out(const Tensor & self, const Scalar & other, Tensor & out)
    {
        return comp_out(self,other,out,"<");
    }
    // {"schema": "aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & gt_out(const Tensor & self, const Scalar & other, Tensor & out)
    {
        return comp_out(self,other,out,">");
    }



    // {"schema": "aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & neg_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        return unitary_op(self,out,"y0=-x0;");
    }
    // {"schema": "aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & reciprocal_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        return unitary_op(self,out,"y0=1.0f/x0;");
    }

    // {"schema": "aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & sqrt_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        return unitary_op(self,out,"y0 = sqrt(x0);");
    }

    // {"schema": "aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & pow_out(const Tensor & self, const Scalar & exponent, Tensor & out)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        Tensor out_c = out.contiguous();
        double val = exponent.toDouble();
        dlprim::Tensor x = todp(self_c);
        dlprim::Tensor y = todp(out_c);
        
        dlprim::core::pointwise_operation({x},{y},{val},"y0=pow(x0,w0);",getExecutionContext(self));

        if(!out.is_contiguous())
            out.copy_(out_c);
        sync_if_needed(self.device());
        return out;
    }


    // {"schema": "aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & div_out(const Tensor & self, const Tensor & other, Tensor & out)
    {
        GUARD;
        Tensor self_c = self.contiguous(), out_c = out.contiguous();
        
        dlprim::Tensor x0=todp(self_c);
        dlprim::Tensor y0=todp(out_c);
        double value=0;
        if(isCPUScalar(other,value)) {
            dlprim::core::pointwise_operation({x0},{y0},{double(1.0/value)},
                                        "y0 = x0*w0;",
                                        getExecutionContext(self));
        }
        else {
            Tensor other_c = other.contiguous();
            dlprim::Tensor x1=todp(other_c);
            dlprim::core::pointwise_operation_broadcast({x0,x1},{y0},{},
                                        "y0 = x0/x1;",
                                        getExecutionContext(self));
        }
        
        if (!out.is_contiguous())
            out.copy_(out_c);
        
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & lerp_out(const Tensor & self, const Tensor & end, const Scalar & weight, Tensor & out)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        Tensor end_c = end.contiguous();
        dlprim::Tensor x0=todp(self_c);
        dlprim::Tensor x1=todp(end_c);
        dlprim::Tensor y0 = todp(out);
        float w = weight.toDouble();

        dlprim::core::pointwise_operation_broadcast({x0,x1},{y0},{w},
                                      "y0 = x0 + w0 * (x1 - x0 );",
                                      getExecutionContext(self));
        sync_if_needed(self.device());
        return out;

    }

    Tensor & binary_op_out_tensor(const Tensor & self, const Tensor & other, Tensor & out,std::string const &op,std::string op_builder="")
    {
        GUARD;
        Tensor self_c  = self.contiguous(), out_c = out.contiguous(),
               other_c = other.contiguous();
        
        if(op_builder.empty()) {
            op_builder = "y0 = left " + op + " right;";
        }

        dlprim::Tensor y(todp(out_c));
        double value;
        if(isCPUScalar(other,value)) {
            dlprim::Tensor x0(todp(self_c));
            dlprim::core::pointwise_operation_broadcast({x0},{y},{value},{x0.dtype()},
                        "typeof_x0 left = x0; typeof_w0 right = w0;" + op_builder,
                        getExecutionContext(self));
            sync_if_needed(self.device());
        }
        else if(isCPUScalar(self,value)) {
            dlprim::Tensor x0(todp(other_c));
            dlprim::core::pointwise_operation_broadcast({x0},{y},{value},{x0.dtype()},
                        "typeof_w0 left = w0; typeof_x0 right = x0;" + op_builder,
                        getExecutionContext(other));
            sync_if_needed(other.device());
        }
        else {
            dlprim::Tensor x0(todp(self_c));
            dlprim::Tensor x1(todp(other_c));
            dlprim::core::pointwise_operation_broadcast({x0,x1},{y},{},
                    "typeof_x0 left = x0; typeof_x1 right = x1;" + op_builder,
                    getExecutionContext(self));
            sync_if_needed(self.device());
        }
        
        if (!out.is_contiguous())
            out.copy_(out_c);

        return out;
    }


    Tensor & mul_out(const Tensor & self, const Tensor & other, Tensor & out)
    {
        GUARD;
        return binary_op_out_tensor(self,other,out,"*");
    }

    // {"schema": "aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & addcdiv_out(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, const Scalar & value, Tensor & out)
    {
        GUARD;
        Tensor self_c = self.contiguous(), out_c = out.contiguous(),
               tensor1_c = tensor1.contiguous(), tensor2_c = tensor2.contiguous();
        
        dlprim::Tensor x0 = todp(self_c);
        dlprim::Tensor x1 = todp(tensor1_c);
        dlprim::Tensor x2 = todp(tensor2_c);
        dlprim::Tensor y0 = todp(out_c);
        float w0 = value.toDouble();
        dlprim::core::pointwise_operation_broadcast({x0,x1,x2},{y0},{w0},
                                      "y0 = x0 + w0 * (x1/x2);",
                                      getExecutionContext(self));
        
        if (!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::threshold_backward.grad_input(Tensor grad_output, Tensor self, Scalar threshold, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & threshold_backward_out(const Tensor & grad_output, const Tensor & self, const Scalar & threshold, Tensor & grad_input)
    {
        GUARD; 
        Tensor self_c = self.contiguous(),
               grad_input_c = grad_input.contiguous(),
               grad_output_c = grad_output.contiguous();
               
        dlprim::Tensor dy=todp(grad_output_c);
        dlprim::Tensor dx=todp(grad_input_c);
        dlprim::Tensor Y=todp(self_c);
        float th = threshold.toDouble();
        dlprim::core::pointwise_operation({Y,dy},{dx},{th},"y0 = (x0 > w0) ? x1 : 0;",getExecutionContext(self));
        
        if(!grad_input.is_contiguous())
            grad_input.copy_(grad_input_c);
        
        sync_if_needed(self.device());
        return grad_input;
    }

    std::pair<dlprim::Shape,dlprim::Shape> squeeze_dim(dlprim::Shape s,OptionalIntArrayRef odim,bool keepdim)
    {
        GUARD;
        std::vector<size_t> full,squeezed;
        std::vector<int> dims;
        if(odim)
            dims.assign(odim->begin(),odim->end());
        if(dims.empty()) {
            for(int i=0;i<s.size();i++)
                dims.push_back(i);
        }

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
        auto full_shape = dlprim::Shape::from_range(full.begin(),full.end());
        auto squeezed_shape = dlprim::Shape::from_range(squeezed.begin(),squeezed.end());
        if(squeezed_shape.size() == 0) {
            squeezed_shape = dlprim::Shape(1);
        }
        return std::make_pair(full_shape,squeezed_shape);
    }
    
    enum class RedOp  {
        sum,mean,prod
    };

    Tensor & red_op_out(const Tensor & self, OptionalIntArrayRef dim, bool keepdim, c10::optional<ScalarType> /*dtype*/, Tensor & out, RedOp rop)
    {
        GUARD;
        Tensor self_c = self.contiguous(), out_c = out.contiguous();
        
        dlprim::Tensor X = todp(self_c);
        auto r = squeeze_dim(X.shape(),dim,keepdim);
        dlprim::Tensor Y = todp(out_c);
        TORCH_CHECK(r.second == Y.shape(),"Invalid output shape");
        Y.reshape(r.first);

        double scale = (rop == RedOp::mean) ? double(Y.shape().total_size()) / double(X.shape().total_size()) : 1;

        auto q = getExecutionContext(self);
        dlprim::Context ctx(q);
        auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
                ctx,
                {X.specs()},{Y.specs()},
                0,dlprim::float_data,
                "y0=x0;",
                (rop == RedOp::prod ? "reduce_y0 =   1;" : "reduce_y0 =   0;"),
                (rop == RedOp::prod ? "reduce_y0 *= y0;" : "reduce_y0 += y0;")
        );

        WSGuard wsg(op->workspace(),self.device());
        op->enqueue({X},{Y},wsg.ws,{},{scale},{0},q);
        
        if (!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(self.device());
        return out;
    }


    // {"schema": "aten::mean.out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & mean_out(const Tensor & self, OptionalIntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out)
    {
        GUARD;
        return red_op_out(self,dim,keepdim,dtype,out,RedOp::mean);
    }
    
    // {"schema": "aten::sum.IntList_out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & sum_out(const Tensor & self, OptionalIntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype, Tensor & out)
    {
        GUARD;
        return red_op_out(self,dim,keepdim,dtype,out,RedOp::sum);
    }
    // {"schema": "aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}    
    Tensor & prod_out(const Tensor & self, int64_t dim, bool keepdim, ::std::optional<ScalarType> dtype, Tensor & out)
    {
        GUARD;
        std::vector<int64_t> dims({dim});
        return red_op_out(self,dims,keepdim,dtype,out,RedOp::prod);
    }


    // {"schema": "aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)", "dispatch": "True", "default": "False"} 
    Tensor hardtanh(Tensor const &self, const Scalar & min_val, const Scalar & max_val)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        dlprim::Tensor X = todp(self_c);
        Tensor out = new_tensor_as(X.shape(),self);
        dlprim::Tensor Y(todp(out));
        double w0 = min_val.toDouble();
        double w1 = max_val.toDouble();
        dlprim::core::pointwise_operation({X},{Y},{w0,w1},"y0=max(w0,min(w1,x0));",getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }


    // {"schema": "aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)", "dispatch": "True", "default": "False"} 
    Tensor & hardtanh_(Tensor & self, const Scalar & min_val, const Scalar & max_val)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        dlprim::Tensor X=todp(self_c);
        double w0 = min_val.toDouble();
        double w1 = max_val.toDouble();
        dlprim::core::pointwise_operation({X},{X},{w0,w1},"y0=max(w0,min(w1,x0));",getExecutionContext(self));
        if(!self.is_contiguous())
            self.copy_(self_c);
        sync_if_needed(self.device());
        return self;
    }

    // {"schema": "aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor hardtanh_backward(const Tensor & grad_output, const Tensor & self, const Scalar & min_val, const Scalar & max_val)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        dlprim::Tensor X  = todp(self_c);
        dlprim::Tensor dY = todp(grad_output);
        Tensor result = new_tensor_as(X.shape(),self);
        dlprim::Tensor dX = todp(result);
        double w0 = min_val.toDouble();
        double w1 = max_val.toDouble();
        dlprim::core::pointwise_operation({X,dY},{dX},{w0,w1},"y0 = (w0 <= x0 && x0 <= w1) ? x1 : 0;",getExecutionContext(self));
        sync_if_needed(self.device());
        return result;
    }


    // {"schema": "aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor abs(const Tensor & self)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        dlprim::Tensor x=todp(self_c);
        Tensor out = new_tensor_as(x.shape(),self);
        dlprim::Tensor y=todp(out);
        dlprim::core::pointwise_operation({x},{y},{},"y0 = x0 < 0 ? -x0 : x0;",getExecutionContext(self));
        sync_if_needed(self.device());
        return out;
    }
     // {"schema": "aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & round_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        return unitary_op(self,out,"y0=round(x0);");
    }
    
    // {"schema": "aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & atan_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        return unitary_op(self,out,"y0=atan(x0);");
    }


     // {"schema": "aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & abs_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        return unitary_op(self,out,"y0 = x0 < 0 ? -x0 : x0;");
    }

    // {"schema": "aten::sgn.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & sgn_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        return unitary_op(self,out,"y0 = x0 < 0 ? -1 : (x0 > 0 ? 1 : 0) ;");
    }

    template<typename TL>
    Tensor &cat_internal(TL const &tensors, int64_t dim, Tensor &out, bool reuse)
    {
        GUARD;
        std::vector<dlprim::Tensor> list;
        std::vector<Tensor> list_c;
        for(auto const &t:tensors) {
            list_c.push_back(t.contiguous());
            list.push_back(todp(list_c.back()));
        }
        TORCH_CHECK(!list_c.empty());
        Tensor &ref_tensor=list_c.front();
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
        dlprim::Tensor Y;
        Tensor out_c;
        if(reuse) {
            out_c = out.contiguous();
            Y = todp(out_c);
            TORCH_CHECK(Y.shape() == ref,"Output shape is not correct for concatenation");
        }
        else {
            out = new_tensor_as(ref,ref_tensor);
            Y = todp(out);
        }
        dlprim::ExecutionContext q(getExecutionContext(ref_tensor));
        dlprim::Context ctx(q);
        

        dlprim::core::SliceCopy cp(ctx,todp(out.dtype()));
        for(size_t i=0,pos=0;i<list.size();i++) {
            Tensor new_tensor;
            dlprim::Tensor x;
            // handle casting
            if(list_c[i].dtype() != out.dtype()) {
                new_tensor = list_c[i].to(out.dtype());
                x=todp(new_tensor);
            }
            else
                x=list[i];
            size_t slice = list[i].shape()[dim];
            cp.tensor_slice_copy(dim,slice,
                                      Y,pos,
                                      x,0,
                                      0.0,q);
            pos += slice;
        }
        
        if (reuse && !out.is_contiguous())
            out.copy_(out_c);
        
        sync_if_needed(ref_tensor.device());
        return out;
    }

    // {"schema": "aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & cat_out(const ITensorListRef & tensors, int64_t dim, Tensor & out)
    {
        GUARD;
        cat_internal(tensors,dim,out,true);
		return out;
    }
    


    // {"schema": "aten::_cat(Tensor[] tensors, int dim=0) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor _cat(TensorList tensors, int64_t dim)
    {
        GUARD;
        Tensor out;
        cat_internal(tensors,dim,out,false);
        return out;
    }


    // {"schema": "aten::hardswish_(Tensor(a!) self) 
    Tensor & hardswish_(Tensor & self)
    {
        GUARD;
        
        Tensor self_c = self.contiguous();
        dlprim::Tensor x=todp(self_c);
        dlprim::core::pointwise_operation({x},{x},{},"y0 = x0 <= -3.0f ? 0 : (x0>=3.0f ? x0 : x0*(x0+3.0f)/6.0f);",getExecutionContext(self));
        
        if (!self.is_contiguous())
            self.copy_(self_c);
        
        sync_if_needed(self.device());
        return self;
    }
    // {"schema": "aten::hardsigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & hardsigmoid_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        return unitary_op(self,out,"y0 = x0 <= -3.0f ? 0 : (x0>=3.0f ? 1.0f : x0/6.0f + 0.5f);");
    }
    // {"schema": "aten::hardsigmoid_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & hardsigmoid_backward_out(const Tensor & grad_output, const Tensor & self, Tensor & grad_input)
    {
        GUARD;
        Tensor self_c = self.contiguous(),
               grad_input_c = grad_input.contiguous(),
               grad_output_c = grad_output.contiguous();
        
        dlprim::Tensor x=todp(self_c);
        dlprim::Tensor dx=todp(grad_input_c);
        dlprim::Tensor dy=todp(grad_output_c);

        dlprim::core::pointwise_operation({x,dy},{dx},{},"y0 = (-3 < x0 && x0 < 3) ? x1 / 6 : 0;",getExecutionContext(self));
        
        if(!grad_input.is_contiguous())
            grad_input.copy_(grad_input_c);
        
        sync_if_needed(self.device());
        return grad_input;
    }
    
    // {"schema": "aten::hardswish_backward(Tensor grad_output, Tensor self) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor hardswish_backward(const Tensor & grad_output, const Tensor & self)
    {
        GUARD;
        Tensor grad_output_c = grad_output.contiguous();
        dlprim::Tensor dy=todp(grad_output_c);
        
        Tensor out = new_tensor_as(dy.shape(),grad_output);
        dlprim::Tensor dx=todp(out);
        
        Tensor self_c = self.contiguous();
        dlprim::Tensor x =todp(self_c);
        dlprim::core::pointwise_operation({x,dy},{dx},{},
            R"xxx(
                if (x0 < -3.0f) {
                    y0 = 0;
                } else if (x0 <= 3.0f) {
                    y0 =  x1 * ((x0 / 3.0f) + 0.5f);
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
        Tensor self_c = self.contiguous(), out_c = out.contiguous();
        
        dlprim::Tensor x=todp(self_c);
        dlprim::Tensor y=todp(out_c);
        dlprim::core::activation_forward(x,y,dlprim::StandardActivations::sigmoid,getExecutionContext(self));
        
        if (!out.is_contiguous())
            out.copy_(out_c);
        
        sync_if_needed(self.device());
        return out;
    }
    // {"schema": "aten::sigmoid(Tensor self) -> Tensor", "dispatch": "True", "default": "True"}
    Tensor sigmoid(const Tensor & self)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        
        dlprim::Tensor x=todp(self_c);
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
        Tensor self_c = self.contiguous();
        
        dlprim::Tensor X=todp(self_c);
        dlprim::core::activation_forward(X,X,dlprim::StandardActivations::sigmoid,getExecutionContext(self));
        
        if(!self.is_contiguous())
          self.copy_(self_c);
        
        sync_if_needed(self.device());
        return self;
    }
    
    // {"schema": "aten::sigmoid_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & sigmoid_backward_out(const Tensor & grad_output, const Tensor & output, Tensor & grad_input)
    {
        GUARD;
        Tensor output_c = output.contiguous(),
               grad_output_c = grad_output.contiguous(),
               grad_input_c  = grad_input.contiguous();
               
        dlprim::Tensor y=todp(output_c);
        dlprim::Tensor dy=todp(grad_output_c);
        
        dlprim::Tensor dx=todp(grad_input_c);
        dlprim::core::activation_backward(dx,dy,y,dlprim::StandardActivations::sigmoid,0,getExecutionContext(grad_output));
        
        if(!grad_input.is_contiguous())
            grad_input.copy_(grad_input_c);
        
        sync_if_needed(grad_output.device());
        return grad_input;
    }

    // {"schema": "aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & tanh_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        Tensor self_c = self.contiguous(), out_c = out.contiguous();
        
        dlprim::Tensor x=todp(self_c);
        dlprim::Tensor y=todp(out_c);
        dlprim::core::activation_forward(x,y,dlprim::StandardActivations::tanh,getExecutionContext(self));
        
        if (!out.is_contiguous())
            out.copy_(out_c);
        
        sync_if_needed(self.device());
        return out;
    }
    
    // {"schema": "aten::tanh_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & tanh_backward_out(const Tensor & grad_output, const Tensor & output, Tensor & grad_input)
    {
        GUARD;                
        Tensor grad_input_c  = grad_input.contiguous(),
               grad_output_c = grad_output.contiguous(),
               output_c      = output.contiguous();
                
        dlprim::Tensor dY=todp(grad_output_c);
        dlprim::Tensor Y=todp(output_c);
        dlprim::Tensor dX=todp(grad_input_c);
        dlprim::core::activation_backward(dX,dY,Y,dlprim::StandardActivations::tanh,0.0,getExecutionContext(grad_output));
        
        if(!grad_input.is_contiguous())
            grad_input.copy_(grad_input_c);
        
        sync_if_needed(grad_output.device());
        return grad_input;
}

    // {"schema": "aten::silu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & silu_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        return unitary_op(self,out,"y0 = x0 / (1.0f + exp(-x0));");
    }

    // {"schema": "aten::silu_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & silu_backward_out(const Tensor & grad_output, const Tensor & self, Tensor & grad_input)
    {
        GUARD; 
        Tensor grad_output_c = grad_output.contiguous(),
               grad_input_c  = grad_input.contiguous(),
               self_c = self.contiguous();
               
        dlprim::Tensor x=todp(self_c);
        dlprim::Tensor dy=todp(grad_output_c);
        dlprim::Tensor dx=todp(grad_input);
        dlprim::core::pointwise_operation({x,dy},{dx},{},
            R"xxx(
                y0 = 1.0f / (1.0f + exp(-x0));
                y0 = x1 * y0 * ( 1.0f + x0 * (1.0f - y0));
            )xxx",
            getExecutionContext(self));
        
        if(!grad_input.is_contiguous())
            grad_input.copy_(grad_input_c);
        
        sync_if_needed(self.device());
        return grad_input;
    }

    // {"schema": "aten::tanh(Tensor self) -> Tensor", "dispatch": "True", "default": "True"}
    Tensor tanh(const Tensor & self)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        dlprim::Tensor x=todp(self_c);
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
        Tensor self_c = self.contiguous();
        dlprim::Tensor X=todp(self_c);
        dlprim::core::activation_forward(X,X,dlprim::StandardActivations::tanh,getExecutionContext(self));
        if(!self.is_contiguous())
            self.copy_(self_c);
        sync_if_needed(self.device());
        return self;
    }

    // {"schema": "aten::leaky_relu.out(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & leaky_relu_out(const Tensor & self, const Scalar & negative_slope, Tensor & out)
    {
        GUARD;
        double slope = negative_slope.to<double>();
        Tensor self_c = self.contiguous(), out_c = out.contiguous();
        
        dlprim::Tensor x=todp(self_c);
        dlprim::Tensor y=todp(out_c);
        dlprim::core::pointwise_operation({x},{y},{slope},"y0 = x0 > 0 ? x0 : w0 * x0;",getExecutionContext(self));
        
        if (!out.is_contiguous())
            out.copy_(out_c);
        
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::leaky_relu_backward.grad_input(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & leaky_relu_backward_out(const Tensor & grad_output, const Tensor & self, const Scalar & negative_slope, bool /*self_is_result*/, Tensor & grad_input)
    {
        GUARD;
        double slope = negative_slope.to<double>();
        Tensor self_c = self.contiguous(),
               grad_input_c  = grad_input.contiguous(),
               grad_output_c = grad_output.contiguous();
        
        dlprim::Tensor y=todp(self_c);
        dlprim::Tensor dy=todp(grad_output_c);
        dlprim::Tensor dx=todp(grad_input_c);
        dlprim::core::pointwise_operation({y,dy},{dx},{slope},"y0 = x0 > 0 ? x1 : w0 * x1;",getExecutionContext(self));
        
        if(!grad_input.is_contiguous())
            grad_input.copy_(grad_input_c);
        
        sync_if_needed(self.device());
        return grad_input;
    }

    // {"schema": "aten::amax.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & amin_amax_out(const Tensor & self, IntArrayRef dim, bool keepdim, Tensor & out,bool is_max)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        Tensor out_c = out.contiguous();
        
        dlprim::Tensor X = todp(self_c);
        dlprim::Tensor Yval = todp(out_c);
        std::vector<int64_t> dims;
        for(int64_t d :dim) {
            dims.push_back(d);
        }
        if(dims.empty()) {
            for(int i=0;i<X.shape().size();i++)
                dims.push_back(i);
        }
        c10::IntArrayRef sqdims(dims.data(),dims.size());
        auto r = squeeze_dim(X.shape(),sqdims,keepdim);
        TORCH_CHECK(r.second == Yval.shape(),"Invalid output shape");
        Yval.reshape(r.first);

        dlprim::ExecutionContext q=getExecutionContext(self);
        dlprim::Context ctx(q);
        std::string ext_val = dlprim::data_type_to_opencl_numeric_limit(X.dtype(),(is_max ? dlprim::dt_min_val : dlprim::dt_max_val));
        auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
                    ctx,
                    {X.specs()},{Yval.specs()},
                    0,X.dtype(),
                    "y0=x0;",
                    "reduce_y0 = " + ext_val + ";",
                    std::string("reduce_y0 = ") + (is_max?"max":"min") + "(reduce_y0,y0);"
                    );
        WSGuard ws_guard(op->workspace(),self.device());
        op->enqueue({X},{Yval},ws_guard.ws,{},{1,1},{0,0},q);
        
        if (!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(self.device());
        return out;
    }
    // {"schema": "aten::amax.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & amax_out(const Tensor & self, IntArrayRef dim, bool keepdim, Tensor & out)
    {
        return amin_amax_out(self,dim,keepdim,out,true);
    }
    // {"schema": "aten::amin.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & amin_out(const Tensor & self, IntArrayRef dim, bool keepdim, Tensor & out)
    {
        return amin_amax_out(self,dim,keepdim,out,false);
    }



    // {"schema": "aten::argmax.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & argmax_out(const Tensor & self, c10::optional<int64_t> dim, bool keepdim, Tensor & out)
    {
        GUARD;
        Tensor self_c = self.contiguous(), out_c = out.contiguous();
        
        dlprim::Tensor X = todp(self_c);
        dlprim::Tensor Yind = todp(out_c);
        std::vector<int64_t> dims;
        if(dim) {
            dims.push_back(*dim);
        }
        else {
            for(int i=0;i<X.shape().size();i++)
                dims.push_back(i);
        }
        c10::IntArrayRef sqdims(dims.data(),dims.size());
        auto r = squeeze_dim(X.shape(),sqdims,keepdim);
        TORCH_CHECK(r.second == Yind.shape(),"Invalid output shape");
        Yind.reshape(r.first);

        WSGuard tmp_guard(Yind.shape().total_size()*dlprim::size_of_data_type(X.dtype()),
                         self.device());
        dlprim::Tensor Yval = tmp_guard.ws.sub_tensor(0,Yind.shape(),X.dtype());

        dlprim::ExecutionContext q=getExecutionContext(self);
        dlprim::Context ctx(q);
        std::string min_val = dlprim::data_type_to_opencl_numeric_limit(X.dtype(),dlprim::dt_min_val);
        auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
                    ctx,
                    {X.specs()},{Yval.specs(),Yind.specs()},
                    0,dlprim::float_data,
                    "y0=x0; y1=reduce_item;",
                    "reduce_y0 = " + min_val + "; reduce_y1 = -1;",
                    R"xxx(
                        if(y0 > reduce_y0) {
                            reduce_y0 = y0; 
                            reduce_y1 = y1; 
                        }
                    )xxx"
                    );
        WSGuard ws_guard(op->workspace(),self.device());
        op->enqueue({X},{Yval,Yind},ws_guard.ws,{},{1,1},{0,0},q);
        
        if (!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(self.device());
        return out;
    }
    
    static Tensor min_or_max(const Tensor & self, bool is_min)
    {
        GUARD;
        Tensor self_c = self.contiguous();
        dlprim::Tensor X = todp(self_c);
        Tensor result = new_tensor_as(dlprim::Shape(),self);
        dlprim::Tensor Y = todp(result);
        std::string y0 = dlprim::data_type_to_opencl_numeric_limit(X.dtype(),(is_min ? dlprim::dt_max_val : dlprim::dt_min_val));
        dlprim::ExecutionContext q=getExecutionContext(self);
        dlprim::Context ctx(q);
        auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
                    ctx,
                    {X.specs()},{Y.specs()},
                    0,X.dtype(),
                    "y0=x0;",
                    std::string("reduce_y0 = ") + y0 + ";",
                    std::string("reduce_y0 = y0 ") + (is_min ? "<" : ">") +  " reduce_y0 ? y0 : reduce_y0;"
                    );
        WSGuard ws_guard(op->workspace(),self.device());
        op->enqueue({X},{Y},ws_guard.ws,{},{1},{0},q);
        
        if (!self.is_contiguous())
            self.copy_(self_c);

        sync_if_needed(self.device());
        return result;
    }

    // {"schema": "aten::min(Tensor self) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor min(const Tensor & self)
    {
        return min_or_max(self,true);
    }

    // {"schema": "aten::max(Tensor self) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor max(const Tensor & self)
    {
        return min_or_max(self,false);
    }

    // {"schema": "aten::dot(Tensor self, Tensor tensor) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor dot(const Tensor & self, const Tensor & tensor)
    {
        GUARD;
        Tensor self_c = self.contiguous(), tensor_c = tensor.contiguous();
        
        dlprim::Tensor x0=todp(self_c);
        dlprim::Tensor x1=todp(tensor_c);
        Tensor result = new_tensor_as(dlprim::Shape(),self_c);
        dlprim::Tensor y=todp(result);
        auto q = getExecutionContext(self);
        dlprim::Context ctx(q);
        auto op = dlprim::core::PointwiseOperationBroadcastReduce::create(
                ctx,
                {x0.specs(),x1.specs()},{y.specs()},
                0,dlprim::float_data,
                "y0=x0*x1;",
                "reduce_y0 = 0;",
                "reduce_y0 += y0;");

        WSGuard wsg(op->workspace(),self.device());
        op->enqueue({x0,x1},{y},wsg.ws,{},{1},{0},q);
        sync_if_needed(self.device());
        return result;
    }


    // {"schema": "aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"} 
    Tensor & ne_out(const Tensor & self, const Scalar & other, Tensor & out)
    {
        GUARD;
        Tensor self_c = self.contiguous(), out_c = out.contiguous();
        
        dlprim::Tensor x(todp(self_c));
        dlprim::Tensor y(todp(out_c));
        dlprim::core::pointwise_operation_broadcast({x},{y},{other.to<double>()},{x.dtype()},
                    "y0 = x0 != w0;", 
                    getExecutionContext(self));
        
        if (!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(self.device());
        return out;

    }


    Tensor & ne_out_tensor(const Tensor & self, const Tensor & other, Tensor & out)
    {
        return binary_op_out_tensor(self,other,out,"!=");
    }
    Tensor & eq_out_tensor(const Tensor & self, const Tensor & other, Tensor & out)
    {
        return binary_op_out_tensor(self,other,out,"==");
    }
    Tensor & lt_out_tensor(const Tensor & self, const Tensor & other, Tensor & out)
    {
        return binary_op_out_tensor(self,other,out,"<");
    }
    Tensor & le_out_tensor(const Tensor & self, const Tensor & other, Tensor & out)
    {
        return binary_op_out_tensor(self,other,out,"<=");
    }
    Tensor & gt_out_tensor(const Tensor & self, const Tensor & other, Tensor & out)
    {
        return binary_op_out_tensor(self,other,out,">");
    }
    Tensor & ge_out_tensor(const Tensor & self, const Tensor & other, Tensor & out)
    {
        return binary_op_out_tensor(self,other,out,">=");
    }

    // {"schema": "aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & eq_out(const Tensor & self, const Scalar & other, Tensor & out)
    {
        GUARD;
        Tensor self_c = self.contiguous(), out_c = out.contiguous();
        
        dlprim::Tensor x(todp(self_c));
        dlprim::Tensor y(todp(out_c));
        dlprim::core::pointwise_operation_broadcast({x},{y},{other.to<double>()},{x.dtype()},
                    "y0 = x0 == w0;",
                    getExecutionContext(self));
        
        if (!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(self.device());
        return out;

    }

    // {"schema": "aten::bitwise_and.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & bitwise_and_out(const Tensor & self, const Tensor & other, Tensor & out)
    {
        GUARD;
        TORCH_CHECK(is_integer(self,true) && is_integer(other,true),"& is not valid for floating point");
        return binary_op_out_tensor(self,other,out,(self.dtype() == c10::kBool ? "&&" : "&"));
    }
    // {"schema": "aten::bitwise_or.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & bitwise_or_out(const Tensor & self, const Tensor & other, Tensor & out)
    {
        GUARD;
        TORCH_CHECK(is_integer(self,true) && is_integer(other,true),"| is not valid for floating point");
        return binary_op_out_tensor(self,other,out,(self.dtype() == c10::kBool ? "||" : "|"));
    }
    // {"schema": "aten::bitwise_xor.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & bitwise_xor_out(const Tensor & self, const Tensor & other, Tensor & out)
    {
        GUARD;
        TORCH_CHECK(is_integer(self,true) && is_integer(other,true),"^ is not valid for floating point");
        return binary_op_out_tensor(self,other,out,(self.dtype() == c10::kBool ? "!=" : "^"));
    }
    // {"schema": "aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & bitwise_not_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        TORCH_CHECK(is_integer(self,true),"~ is valid for integer types");
        return unitary_op(self,out,(self.dtype() == c10::kBool ? "y0 = !x0;" : "y0 = ~x0;"));
    }


    // {"schema": "aten::clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & clamp_out(const Tensor & self, const c10::optional<Scalar> & min, const c10::optional<Scalar> & max, Tensor & out)
    {
        GUARD;
        Tensor self_c = self.contiguous(), out_c = out.contiguous();
        dlprim::Tensor Y = todp(out_c);
        dlprim::Tensor X = todp(self_c);
        auto q = getExecutionContext(self);
        if(min && max)
            dlprim::core::pointwise_operation({X},{Y},{min->to<double>(),max->to<double>()},"y0 = max(w0,min(w1,x0));",q);
        else if(min)
            dlprim::core::pointwise_operation({X},{Y},{min->to<double>()},"y0 = max(w0,x0);",q);
        else if(max)
            dlprim::core::pointwise_operation({X},{Y},{max->to<double>()},"y0 = min(w0,x0);",q);
        else
            dlprim::core::pointwise_operation({X},{Y},{},"y0 = x0;",q);
        
        if (!out.is_contiguous())
            out.copy_(out_c);
        
        sync_if_needed(self.device());
        return out;
    }
    
    // {"schema": "aten::clamp.out(Tensor self, Scalar min, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & clamp_min_out(const Tensor & self, const Scalar & min, Tensor & out)
    {
        GUARD;
        Tensor self_c = self.contiguous(), out_c = out.contiguous();
        dlprim::Tensor Y = todp(out_c);
        dlprim::Tensor X = todp(self_c);
        auto q = getExecutionContext(self);
        
        dlprim::core::pointwise_operation({X},{Y},{min.to<double>()},"y0 = max(w0,x0);",q);
        
        if (!out.is_contiguous())
            out.copy_(out_c);
        
        sync_if_needed(self.device());
        return out;
    }
    
    // {"schema": "aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & ceil_out(const Tensor & self, Tensor & out)
    {
        GUARD;
        return unitary_op(self,out,"y0 = ceil(x0);");
    }

    // {"schema": "aten::gelu.out(Tensor self, *, str approximate='none', Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & gelu_out(const Tensor & self, c10::string_view approximate, Tensor & out)
    {
        GUARD;
        Tensor self_c = self.contiguous(), out_c = out.contiguous();
        dlprim::Tensor Y = todp(out_c);
        dlprim::Tensor X = todp(self_c);
        auto q = getExecutionContext(self);
        TORCH_CHECK(approximate == "none" || approximate == "tanh","Unsupported variant")
        if(approximate == "tanh")
            dlprim::core::pointwise_operation({X},{Y},{},"y0 = 0.5f * x0 * (1.0f + tanh(0.7978845608028654f * x0 * (1.0f + 0.044715f * x0 * x0)));",q); // 0.7978845608028654 = sqrt(2/pi)
        else {
            dlprim::core::pointwise_operation({X},{Y},{},"y0 = x0 * (1.0f + erf(x0 * 0.7071067811865475f  )) / 2.0f;",q); // 0.7071067811865475 = 1/sqrt(2)
        }
            
        if (!out.is_contiguous())
            out.copy_(out_c);
        
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::gelu_backward.grad_input(Tensor grad_output, Tensor self, *, str approximate='none', Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & gelu_backward_out(const Tensor & grad_output, const Tensor & self, c10::string_view approximate, Tensor & grad_input)
    {
        GUARD;
        Tensor grad_output_c = grad_output.contiguous(),
               grad_input_c  = grad_input.contiguous(),
               self_c = self.contiguous();

        dlprim::Tensor dX = todp(grad_input_c);
        dlprim::Tensor dY = todp(grad_output_c);
        dlprim::Tensor X  = todp(self_c);

        auto q = getExecutionContext(self);
        TORCH_CHECK(approximate == "none" || approximate == "tanh","Unsupported variant")

        char const *eq;
        // 1.128379167095512558561 = 2/ sqrt(pi)
        // 0.7071067811865475 = 1/sqrt(2)
        
        if(approximate == "tanh")
            eq = R"xxx(
                dtype alpha = 1.128379167095512558561f * 0.7071067811865475f;
                dtype koeff = 0.044715f;
                dtype beta  = alpha * koeff * 3.0f;
                dtype Y = tanh(alpha * fma(koeff,x0*x0*x0,x0));
                y0 = 0.5f * x1 * fma(
                    fma(-x0,Y * Y, x0),
                    fma(beta,x0*x0,alpha),
                    1 + Y                  
                );
            )xxx";
        else
            eq = R"xxx(
                dtype alpha = 1.128379167095512558561f * 0.7071067811865475f * 0.5f; 
                dtype cdf = 0.5f * (1.0f + erf(x0 * 0.7071067811865475f));
                y0 = x1 * fma(
                    alpha * x0,
                    exp(-0.5f * x0*x0),
                    cdf);
            )xxx";

        dlprim::core::pointwise_operation({X,dY},{dX},{},eq,q);
        
        if (!grad_input.is_contiguous())
            grad_input.copy_(grad_input_c);

        sync_if_needed(self.device());
        return grad_input;
    }

    // {"schema": "aten::logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"} 
    Tensor & logit_out(const Tensor & self, ::std::optional<double> eps, Tensor & out)
    {
        GUARD;
        Tensor self_c = self.contiguous(), out_c = out.contiguous();
        dlprim::Tensor X = todp(self_c);
        dlprim::Tensor Y = todp(out_c);
        auto q = getExecutionContext(self);
        if(eps) {
            double e = *eps;
            dlprim::core::pointwise_operation({X},{Y},{e},
                "dtype z = min(1.0f-w0,max(w0,x0)); "
                "y0 = log(z / (z-1.0f)); ",q);
        }
        else {
            dlprim::core::pointwise_operation({X},{Y},{},
                "y0 = log(x0 / (x0-1.0f));",q);
        }
        if(!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(self.device());
        return out;
    }
    // {"schema": "aten::logit(Tensor self, float? eps=None) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor logit(const Tensor & self, ::std::optional<double> eps)
    {
        Tensor self_c = self.contiguous();
        dlprim::Tensor X = todp(self_c);
        
        torch::Tensor result = new_tensor_as(X.shape(),self);
        logit_out(self_c,eps,result);
        return result;
    }

    // {"schema": "aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False")
    Tensor & arange_out(const Scalar & start, const Scalar & end, const Scalar & step, Tensor & out) 
    {
        GUARD;
        double dstart = start.to<double>();
        double dend   = end.to<double>();
        double dstep = step.to<double>();
        int64_t size = std::ceil((dend - dstart) / dstep);
        int64_t numel = out.numel();
        
        if(numel  != size) {
            if(numel!=0) {
                 TORCH_WARN("The number of elements in the out tensor of shape ", out.sizes(),
                    " is ", numel, " which does not match the computed number of elements ", size,
                    ". Note that this may occur as a result of rounding error. "
                    "The out tensor will be resized to a tensor of shape (", size, ",).");
            }
            //Tensor tmp = new_tensor_as(dlprim::Shape(size),out);
            //out = std::move(tmp);
            out.resize_({size});
        }
        Tensor out_c = out.contiguous();
        dlprim::Tensor Y = todp(out_c);
        auto q = getExecutionContext(out);
        dlprim::core::pointwise_operation({},{Y},{dstart,dstep},
            "y0 = w0 + index*w1;",q);
        if(!out.is_contiguous())
            out.copy_(out_c);

        sync_if_needed(out.device());
        return out;
    }

    // {"schema": "aten::maximum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & maximum_out(const Tensor & self, const Tensor & other, Tensor & out)
    {
        GUARD;
        return binary_op_out_tensor(self,other,out,"","y0 = max(left,right); ");
    }
    // {"schema": "aten::minimum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & minimum_out(const Tensor & self, const Tensor & other, Tensor & out)
    {
        GUARD;
        return binary_op_out_tensor(self,other,out,"","y0 = min(left,right); ");
    }

   // {"schema": "aten::log_sigmoid_forward.output(Tensor self, *, Tensor(a!) output, Tensor(b!) buffer) -> (Tensor(a!), Tensor(b!))", "dispatch": "True", "default": "False"
    ::std::tuple<Tensor &,Tensor &> log_sigmoid_forward_out(const Tensor & self, Tensor & output, Tensor & buffer)
    {
        GUARD;
        Tensor self_c = self.contiguous(), output_c = output.contiguous(), buffer_c = buffer.contiguous();
        dlprim::Tensor x=todp(self_c), out = todp(output_c), buf = todp(buffer_c);
        dlprim::core::pointwise_operation({x},{out,buf},{},
                    R"xxx(
                    y1 = exp(-fabs(x0));
                    y0 = min((dtype)(0),x0) - log1p(y1);
                    )xxx",
                    getExecutionContext(self));
        
        if(!output.is_contiguous())
            output.copy_(output_c);

        if(!buffer.is_contiguous())
            buffer.copy_(buffer_c);

        sync_if_needed(self.device());
        return ::std::tuple<Tensor &,Tensor &>(output,buffer);

    }

    /// {"schema": "aten::log_sigmoid_forward(Tensor self) -> (Tensor output, Tensor buffer)", "dispatch": "True", "default": "False"}
    ::std::tuple<Tensor,Tensor> log_sigmoid_forward(const Tensor & self)
    {
        GUARD;
        Tensor out = at::empty_like(self);
        Tensor buffer = at::empty_like(self);
        
        log_sigmoid_forward_out(self,out,buffer);
        return ::std::tuple<Tensor,Tensor>(out,buffer);
    }
    
    // {"schema": "aten::log_sigmoid_backward.grad_input(Tensor grad_output, Tensor self, Tensor buffer, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & log_sigmoid_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & buffer, Tensor & grad_input)
    {
        GUARD;
        Tensor grad_output_c = grad_output.contiguous();
        Tensor self_c = self.contiguous();
        Tensor buffer_c = buffer.contiguous();
        Tensor grad_input_c = grad_input.contiguous();

        dlprim::Tensor dy = todp(grad_output_c);
        dlprim::Tensor x = todp(self_c);
        dlprim::Tensor buf = todp(buffer_c);
        dlprim::Tensor dx = todp(grad_input_c);

        dlprim::core::pointwise_operation({x,buf,dy},{dx},{},
                    R"xxx(
                    int is_negative = x0 < 0;
                    dtype maxd = is_negative ? 1.0f: 0.0f;
                    dtype s = is_negative ? 1.0f: -1.0f;
                    y0 = (maxd - s * (x1 / ((dtype)(1) + x1))) * x2;
                    )xxx",
                    getExecutionContext(self));
        
        if(!grad_input.is_contiguous())
            grad_input.copy_(grad_input_c);;

        sync_if_needed(self.device());
        return grad_input;
    }
    // {"schema": "aten::log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer)
    {
        GUARD;
        Tensor grad_input = at::empty_like(grad_output);
        log_sigmoid_backward_out(grad_output,self,buffer,grad_input);
        return grad_input;
    }


  

    

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
      m.impl("aten::relu_",&ptdlprim::relu_);
      m.impl("aten::mul.out",&ptdlprim::mul_out);
      m.impl("aten::mul_.Scalar",&ptdlprim::mul_scalar_);
      m.impl("aten::add.out",&ptdlprim::add_out);
      m.impl("aten::sub.out",&ptdlprim::sub_out);
      m.impl("aten::exp.out",&ptdlprim::exp_out);
      m.impl("aten::log.out",&ptdlprim::log_out);
      m.impl("aten::addcmul.out",&ptdlprim::addcmul_out);
      m.impl("aten::sqrt.out",&ptdlprim::sqrt_out);
      m.impl("aten::div.out",&ptdlprim::div_out);
      m.impl("aten::addcdiv.out",&ptdlprim::addcdiv_out);
      m.impl("aten::threshold_backward.grad_input",&ptdlprim::threshold_backward_out);
      
      m.impl("aten::mean.out",&ptdlprim::mean_out);
      m.impl("aten::sum.IntList_out",&ptdlprim::sum_out);
      m.impl("aten::prod.int_out",&ptdlprim::prod_out);
      
      m.impl("aten::hardtanh",&ptdlprim::hardtanh);
      m.impl("aten::hardtanh_",&ptdlprim::hardtanh_);
      m.impl("aten::hardtanh_backward",&ptdlprim::hardtanh_backward);
      m.impl("aten::abs",&ptdlprim::abs);
      m.impl("aten::abs.out",&ptdlprim::abs_out);
      m.impl("aten::sgn.out",&ptdlprim::sgn_out);
      m.impl("aten::_cat",&ptdlprim::_cat);
      m.impl("aten::cat.out",&ptdlprim::cat_out);
      m.impl("aten::hardswish_",&ptdlprim::hardswish_);
      m.impl("aten::hardsigmoid.out",&ptdlprim::hardsigmoid_out);
      m.impl("aten::hardsigmoid_backward.grad_input",&ptdlprim::hardsigmoid_backward_out);
      m.impl("aten::sigmoid.out",&ptdlprim::sigmoid_out);
      m.impl("aten::sigmoid",&ptdlprim::sigmoid);
      m.impl("aten::sigmoid_",&ptdlprim::sigmoid_);
      m.impl("aten::sigmoid_backward.grad_input",&ptdlprim::sigmoid_backward_out);
      m.impl("aten::tanh.out",&ptdlprim::tanh_out);
      m.impl("aten::tanh_backward.grad_input",&ptdlprim::tanh_backward_out);
      m.impl("aten::silu.out",&ptdlprim::silu_out);
      m.impl("aten::silu_backward.grad_input",&ptdlprim::silu_backward_out);
      m.impl("aten::tanh",&ptdlprim::tanh);
      m.impl("aten::tanh_",&ptdlprim::tanh_);
      m.impl("aten::leaky_relu.out",&ptdlprim::leaky_relu_out);
      m.impl("aten::leaky_relu_backward.grad_input",&ptdlprim::leaky_relu_backward_out);
      m.impl("aten::hardswish_backward",&ptdlprim::hardswish_backward);
      m.impl("aten::argmax.out",&ptdlprim::argmax_out);
      
      m.impl("aten::ne.Tensor_out",&ptdlprim::ne_out_tensor);
      m.impl("aten::eq.Tensor_out",&ptdlprim::eq_out_tensor);
      m.impl("aten::lt.Tensor_out",&ptdlprim::lt_out_tensor);
      m.impl("aten::gt.Tensor_out",&ptdlprim::gt_out_tensor);
      m.impl("aten::le.Tensor_out",&ptdlprim::le_out_tensor);
      m.impl("aten::ge.Tensor_out",&ptdlprim::ge_out_tensor);

      m.impl("aten::ne.Scalar_out",&ptdlprim::ne_out);
      m.impl("aten::eq.Scalar_out",&ptdlprim::eq_out);
      m.impl("aten::le.Scalar_out",&ptdlprim::le_out);
      m.impl("aten::ge.Scalar_out",&ptdlprim::ge_out);
      m.impl("aten::lt.Scalar_out",&ptdlprim::lt_out);
      m.impl("aten::gt.Scalar_out",&ptdlprim::gt_out);

      m.impl("aten::bitwise_and.Tensor_out",&ptdlprim::bitwise_and_out);
      m.impl("aten::bitwise_or.Tensor_out",&ptdlprim::bitwise_or_out);
      m.impl("aten::bitwise_xor.Tensor_out",&ptdlprim::bitwise_xor_out);
      m.impl("aten::bitwise_not.out",&ptdlprim::bitwise_not_out);

      m.impl("aten::min",&ptdlprim::min);
      m.impl("aten::max",&ptdlprim::max);
      m.impl("aten::clamp.out",&ptdlprim::clamp_out);
      m.impl("aten::clamp_min.out",&ptdlprim::clamp_min_out);
      m.impl("aten::neg.out",&ptdlprim::neg_out);
      m.impl("aten::reciprocal.out",&ptdlprim::reciprocal_out);
      m.impl("aten::dot",&ptdlprim::dot);
      m.impl("aten::ceil.out",&ptdlprim::ceil_out);
      m.impl("aten::gelu.out",&ptdlprim::gelu_out);
      m.impl("aten::gelu_backward.grad_input",&ptdlprim::gelu_backward_out);
      m.impl("aten::lerp.Scalar_out",&ptdlprim::lerp_out);
      m.impl("aten::logit.out",&ptdlprim::logit_out);
      m.impl("aten::logit",&ptdlprim::logit);
      m.impl("aten::arange.start_out",&ptdlprim::arange_out);
      m.impl("aten::amax.out",&ptdlprim::amax_out);
      m.impl("aten::amin.out",&ptdlprim::amin_out);
      m.impl("aten::round.out",&ptdlprim::round_out);
      m.impl("aten::atan.out",&ptdlprim::atan_out);
      m.impl("aten::maximum.out",&ptdlprim::maximum_out);
      m.impl("aten::minimum.out",&ptdlprim::minimum_out);
      m.impl("aten::pow.Tensor_Scalar_out",&ptdlprim::pow_out);
      m.impl("aten::log_sigmoid_forward",&ptdlprim::log_sigmoid_forward);
      m.impl("aten::log_sigmoid_forward.output",&ptdlprim::log_sigmoid_forward_out);
      m.impl("aten::log_sigmoid_backward.grad_input", &ptdlprim::log_sigmoid_backward_out);
      m.impl("aten::log_sigmoid_backward", &ptdlprim::log_sigmoid_backward);
}

TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
      m.impl("aten::relu",&ptdlprim::act_autograd<dlprim::StandardActivations::relu>);
}

