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

    SeqState get_random_seq(int64_t items,c10::optional<Generator> generator)
    {
        static dlprim::RandomState state(time(0));
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
        auto seq = get_random_seq(rnd.shape().total_size(),generator);
        dlprim::core::fill_random(rnd,seq.seed,seq.sequence,dlprim::core::rnd_bernoulli,p,0,getExecutionContext(self));
        sync_if_needed(self.device());
        return self;
    }

    // {"schema": "aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & normal_(Tensor & self, double mean, double std, c10::optional<Generator> generator)
    {
        GUARD;
        dlprim::Tensor rnd=todp(self);
        auto seq = get_random_seq(rnd.shape().total_size(),generator);
        dlprim::core::fill_random(rnd,seq.seed,seq.sequence,dlprim::core::rnd_normal,mean,std*std,getExecutionContext(self));
        sync_if_needed(self.device());
        return self;
    }

    // {"schema": "aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor & uniform_(Tensor & self, double from, double to, c10::optional<Generator> generator)
    {
        GUARD;
        dlprim::Tensor rnd=todp(self);
        auto seq = get_random_seq(rnd.shape().total_size(),generator);
        dlprim::core::fill_random(rnd,seq.seed,seq.sequence,dlprim::core::rnd_uniform,from,to,getExecutionContext(self));
        sync_if_needed(self.device());
        return self;
    }

} // namespace
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
      m.impl("aten::bernoulli_.float",&ptdlprim::bernoulli_);
      m.impl("aten::normal_",&ptdlprim::normal_);
      m.impl("aten::uniform_",&ptdlprim::uniform_);
}

