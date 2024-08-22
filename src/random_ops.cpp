#include "CLTensor.h"
#include "utils.h"

#include <dlprim/core/util.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/core/loss.hpp>
#include <dlprim/random.hpp>

#include <iostream>
namespace ptdlprim {

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;


    struct SeqState {
         dlprim::RandomState::seed_type seed;
         dlprim::RandomState::sequence_type sequence;
    };

    SeqState get_random_seq(c10::Device const &d,int64_t items,c10::optional<Generator> generator)
    {
        dlprim::RandomState &state = CLContextManager::instance().rng_state(d.index());
        size_t rounds = (items +  dlprim::philox::result_items - 1) / dlprim::philox::result_items;
        SeqState s;
        s.seed = state.seed();
        s.sequence  = state.sequence_bump(rounds);
        return  s;
    }

    // {"schema": "aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & bernoulli_(Tensor & self, double p, c10::optional<Generator> generator)
    {
        GUARD;
        dlprim::Tensor rnd=todp(self);
        auto seq = get_random_seq(self.device(),rnd.shape().total_size(),generator);
        dlprim::core::fill_random(rnd,seq.seed,seq.sequence,dlprim::core::rnd_bernoulli,p,0,getExecutionContext(self));
        sync_if_needed(self.device());
        return self;
    }

    // {"schema": "aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & normal_(Tensor & self, double mean, double std, c10::optional<Generator> generator)
    {
        GUARD;
        dlprim::Tensor rnd=todp(self);
        auto seq = get_random_seq(self.device(),rnd.shape().total_size(),generator);
        dlprim::core::fill_random(rnd,seq.seed,seq.sequence,dlprim::core::rnd_normal,mean,std*std,getExecutionContext(self));
        sync_if_needed(self.device());
        return self;
    }

    // {"schema": "aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & uniform_(Tensor & self, double from, double to, c10::optional<Generator> generator)
    {
        GUARD;
        dlprim::Tensor rnd=todp(self);
        auto seq = get_random_seq(self.device(),rnd.shape().total_size(),generator);
        dlprim::core::fill_random(rnd,seq.seed,seq.sequence,dlprim::core::rnd_uniform,from,to,getExecutionContext(self));
        sync_if_needed(self.device());
        return self;
    }
    // {"schema": "aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)", "dispatch": "True", "default": "False"}
    ::std::tuple<Tensor,Tensor> native_dropout(const Tensor & input, double p, ::std::optional<bool> train)
    {
        GUARD;
        Tensor input_c = input.contiguous();
        dlprim::Tensor X = todp(input_c);
        Tensor mask = new_tensor_as(X.shape(),input_c);
        Tensor res =  new_tensor_as(X.shape(),input_c);
        dlprim::Tensor Y = todp(res);
        dlprim::Tensor M = todp(mask);
        if(train && *train && p > 0) {
            bernoulli_(mask,1-p,c10::nullopt);
            dlprim::core::pointwise_operation({X,M},{Y},{1/(1-p)},
                                          "y0 = x0*x1*w0;",
                                          getExecutionContext(input));
        }
        else {
            torch::fill_(mask,1);
        }
        sync_if_needed(input.device());
        return std::make_pair(res,mask);
    }
    // {"schema": "aten::native_dropout_backward(Tensor grad_output, Tensor mask, float scale) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor native_dropout_backward(const Tensor & grad_output, const Tensor & mask, double scale)
    {
        GUARD;
        Tensor grad_output_c=grad_output.contiguous();
        Tensor mask_c=mask.contiguous();
        dlprim::Tensor dy = todp(grad_output_c);
        dlprim::Tensor m = todp(mask_c);
        Tensor res =  new_tensor_as(dy.shape(),grad_output);
        dlprim::Tensor dx = todp(res);
        dlprim::core::pointwise_operation({dy,m},{dx},{scale},
                "y0 = x0*x1*w0;",
                getExecutionContext(grad_output));
        sync_if_needed(grad_output.device());
        return res;
    }


} // namespace
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
      m.impl("aten::bernoulli_.float",&ptdlprim::bernoulli_);
      m.impl("aten::normal_",&ptdlprim::normal_);
      m.impl("aten::uniform_",&ptdlprim::uniform_);
      m.impl("aten::native_dropout",&ptdlprim::native_dropout);
      m.impl("aten::native_dropout_backward",&ptdlprim::native_dropout_backward);
}

