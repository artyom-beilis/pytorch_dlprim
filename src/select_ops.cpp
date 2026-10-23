#include <ATen/ATen.h>
#include <Python.h>
#include "CLTensor.h"
#include "utils.h"

#include <dlprim/core/select.hpp>
#include <dlprim/core/pointwise.hpp>

extern "C"  int pt_dlpprim_pycallback(void* p) {
    PyErr_SetString(PyExc_IndexError, "out of bound");
    return -1;
}

extern "C"  void CL_CALLBACK pt_dplprim__event_err(cl_event evt, cl_int status, void* dat) {
    clReleaseEvent(evt);
    if (*(cl_int*)dat == -1) {
        Py_AddPendingCall(pt_dlpprim_pycallback, nullptr);
    }
}

namespace ptdlprim {
    using namespace at;

    using c10::Device;
    using c10::DeviceType;
    using dlprim::core::Select;

    Tensor& _select(const Tensor& self, int64_t dim, const Tensor& index, Tensor& out, Select::Op op, Select::Target tg, bool scal, double alpha)
    {
        GUARD;
        auto x = todp(self), i = todp(index), y = todp(out);
        WSGuard gd(sizeof(cl_int), self.device());
        dlprim::Tensor &ws = gd.ws;
        auto e = getExecutionContext(self);
        static cl_int err;
        err = 0;
        e.queue().enqueueWriteBuffer(ws.device_buffer(),false,ws.device_offset(),sizeof(cl_int),&err);
        dlprim::Context ctx(e);
        auto sel = Select::create(ctx, self.dim(), x.dtype(), i.dtype(), op, tg, scal);
        auto xstrds = self.strides(), istrds = index.strides(), ystrds = out.strides();
        auto xst = dlprim::Shape::from_range(xstrds.begin(), xstrds.end());
        auto ist = dlprim::Shape::from_range(istrds.begin(), istrds.end());
        auto yst = dlprim::Shape::from_range(ystrds.begin(), ystrds.end());
        sel->enqueue(x, y, i, xst, yst, ist, dim, &ws, alpha, e);
        cl::Event ev;
        e.queue().enqueueReadBuffer(ws.device_buffer(),false,ws.device_offset(),sizeof(cl_int),&err,nullptr,&ev);
        ev.setCallback(CL_COMPLETE, pt_dplprim__event_err, &err);
        sync_if_needed(self.device());
        return out;
    }

    // {"schema": "aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor& gather_out(const Tensor& self, int64_t dim, const Tensor& index, bool sparse_grad, Tensor& out)
    {
        GUARD;
        TORCH_CHECK(!sparse_grad, "sparse_grad=True not implemented");
        return _select(self, dim, index, out, Select::ASSIGN, Select::X, false, 0.);
    }
    // "aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor";
    Tensor gather(const Tensor& self, int64_t dim, const Tensor& index, bool sparse_grad)
    {
        GUARD;
        auto out = new_ocl_tensor(index.sizes(), self.device(), self.dtype().toScalarType());
        return gather_out(self, dim, index, sparse_grad, out);
    }

    // {"schema": "aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor& index_select_out(const Tensor& self, int64_t dim, const Tensor& index, Tensor& out)
    {
        GUARD;
        // TODO: split and merge contiguous dims
        std::vector<int64_t> strds(self.dim(), 0);
        strds[dim] = index.stride(0);
        Tensor idx = index.as_strided(out.sizes(), strds);
        return _select(self, dim, idx, out, Select::ASSIGN, Select::X, false, 0.);
    }

    // {"schema": "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index)
    {
        GUARD;
        auto sizs(self.sizes().vec());
        sizs[dim] = index.size(0);
        Tensor out = new_ocl_tensor(sizs, self.device(), self.dtype().toScalarType());
        return index_select_out(self, dim, index, out);
    }

    // {"schema": "aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor(a!)", "dispatch": "True", "defualt": "False"}
    Tensor& index_add_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Scalar& alpha)
    {
        GUARD;
        // TODO: split and merge contiguous dims
        std::vector<int64_t> strds(source.dim(), 0);
        strds[dim] = index.stride(0);
        Tensor idx = index.as_strided(source.sizes(), strds);
        return _select(source, dim, idx, self, Select::ADD, Select::Y, true, alpha.toFloat());
    }
    // {"schema": "aten::index_add.out(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor& index_add(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Scalar& alpha, Tensor& out)
    {
        GUARD;
        out.copy_(self);
        index_add_(out, dim, index, source, alpha);
        return out;
    }

    // {"schema": "aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor& scatter_add_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src)
    {
        GUARD;
        return _select(src, dim, index, self, Select::ADD, Select::Y, false, 0.);
    }

    // {"schema": "aten::scatter_add.out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor& scatter_add(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src, Tensor& out)
    {
        GUARD;
        out.copy_(self);
        scatter_add_(out, dim, index, src);
        return out;
    }

    // {"schema": "aten::embedding_dense_backward.out(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor& embedding_dense_backward_out(const Tensor& grad_output, const Tensor& indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq, Tensor& out)
    {
        GUARD;
        Tensor indc = indices.contiguous();
        dlprim::Tensor dind = todp(indc);
        dlprim::Shape sp(dind.shape().total_size(),1);
        dind.reshape(sp);
        dlprim::Tensor dcnt;
        auto e = getExecutionContext(grad_output);
        out.zero_();
        if (scale_grad_by_freq) {
            auto tp = dind.dtype();
            int64_t W = num_weights.as_int_unchecked();
            WSGuard gd(W*sizeof(cl_uint), indices.device());
            dcnt = gd.ws.workspace_as_type(dlprim::uint32_data);
            dlprim::core::fill_tensor(dcnt, 0.0, e);
            dlprim::Tensor vcnt = dlprim::Tensor(dcnt.device_buffer(),dcnt.device_offset(),sp,tp);
            dlprim::core::pointwise_operation({dind},{vcnt},
                {(double)W}, "if(x0<w0)atomic_inc(((volatile __global uint*)py0)+py0_offset+x0);return;", e);
        }
        int64_t n[2] = {sp[0],grad_output.size(-1)};
        _select(grad_output.reshape(n), 0, indc.as_strided(n, {1,0}), out, Select::ADD, Select::Y, false, 0.);
        int64_t pd = padding_idx.as_int_unchecked();
        if (scale_grad_by_freq) {
            sp[1] = n[1]; // [w,d]->[n,d]
            dlprim::Tensor vout(buffer_from_tensor(out),out.storage_offset(),sp,todp(out.scalar_type())),
                vcnt(dcnt.device_buffer(),dcnt.device_offset(),sp,dlprim::uint32_data);
            dlprim::core::pointwise_operation_broadcast({dind},{vout,vcnt},
                {(double)pd,(double)out.stride(0),(double)out.stride(1)},
                {dind.dtype(),dlprim::int64_data,dlprim::int64_data},
                "py0+=py0_offset+x0*w1+index.s[1]*w2;*py0=x0==w0?0:*py0/py1[py1_offset+x0];return;", e);
        } else if (pd>=0) {
            out.index_put_({pd,indexing::Slice()},0);
        }
        return out;
    }

    // {"schema": "aten::embedding_dense_backward(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq) -> Tensor", "dispatch": "True"
    Tensor embedding_dense_backward(const Tensor& grad_output, const Tensor& indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq) {
        GUARD;
        Tensor out = new_ocl_tensor({num_weights.as_int_unchecked(), grad_output.size(-1)}, grad_output.device(), grad_output.scalar_type());
        return embedding_dense_backward_out(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq, out);
    }
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::index_select.out", &ptdlprim::index_select_out);
    m.impl("aten::index_select", &ptdlprim::index_select);
    m.impl("aten::index_add_", &ptdlprim::index_add_);
    m.impl("aten::index_add.out", &ptdlprim::index_add);
    m.impl("aten::gather.out", &ptdlprim::gather_out);
    m.impl("aten::gather", &ptdlprim::gather);
    m.impl("aten::scatter_add_", &ptdlprim::scatter_add_);
    m.impl("aten::scatter_add.out", &ptdlprim::scatter_add);
    m.impl("aten::embedding_dense_backward.out",&ptdlprim::embedding_dense_backward_out);
    m.impl("aten::embedding_dense_backward",&ptdlprim::embedding_dense_backward);
}
