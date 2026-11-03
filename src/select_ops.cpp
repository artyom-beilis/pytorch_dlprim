#include <ATen/ATen.h>
#include "CLTensor.h"
#include "utils.h"

#include <dlprim/core/select.hpp>
#include <dlprim/core/pointwise.hpp>
#define STR(...) #__VA_ARGS__
#define OOB_CALLBACK 0 // experimental
#if OOB_CALLBACK
#include <Python.h>
#endif

namespace ptdlprim {
    using namespace at;

    using dlprim::core::Select;

#if OOB_CALLBACK
    static int _pycallback(void* p) {
        PyErr_SetString(PyExc_IndexError, "out of bound");
        return -1;
    }

    static void CL_CALLBACK _event_err(cl_event evt, cl_int status, void* dat) {
        clReleaseEvent(evt);
        if (*(cl_int*)dat == -1) {
            Py_AddPendingCall(_pycallback, nullptr);
        }
    }
#endif

    static Tensor& _select(const Tensor& self, int64_t dim, const Tensor& index, Tensor& out, Select::Op op, Select::Target tg, bool scal=false, double alpha=0.)
    {
        GUARD;
        Tensor selfc = self.contiguous(), indexc = index.contiguous(), outc = out.contiguous();
        auto x = todp(selfc), i = todp(indexc), y = todp(outc);
        auto e = getExecutionContext(self);
        dlprim::Context ctx(e);
        auto sel = Select::create(ctx, (int)self.dim(), x.dtype(), i.dtype(), op, tg, scal);
        if (dim < 0) dim += self.dim();
#if OOB_CALLBACK
        WSGuard gd(sizeof(cl_int), self.device());
        dlprim::Tensor &flag = gd.ws;
        static cl_int err;
        err = 0;
        e.queue().enqueueWriteBuffer(flag.device_buffer(), false, flag.device_offset(), sizeof(cl_int), &err);
        sel->enqueue(x, y, i, dim, &flag, alpha, e, false);
        cl::Event evt;
        e.queue().enqueueReadBuffer(flag.device_buffer(),false,flag.device_offset(),sizeof(cl_int),&err,nullptr,&evt);
        evt.setCallback(CL_COMPLETE, _event_err, &err);
        evt() = nullptr; // avoid clReleaseEvent on destruction
#else
        sel->enqueue(x, y, i, dim, nullptr, alpha, e, false);
#endif
        if (!out.is_contiguous()) {
            out.copy_(outc);
        }
        sync_if_needed(self.device());
        return out;
    }

    static void _check_dim(int64_t dims, int64_t dim) {
        TORCH_CHECK_INDEX(-dims<=dim && dim<dims, 
            "Dimension out of range (expected to be in range of [",-dims,", ",dims-1,"], but got ",dim,")");
    }

    // {"schema": "aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor& gather_out(const Tensor& self, int64_t dim, const Tensor& index, bool sparse_grad, Tensor& out)
    {
        GUARD;
        _check_dim(self.dim(), dim);
        TORCH_CHECK(!sparse_grad, "sparse_grad=True not implemented");
        return _select(self, dim, index, out, Select::ASSIGN, Select::X);
    }
    // "aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor";
    Tensor gather(const Tensor& self, int64_t dim, const Tensor& index, bool sparse_grad)
    {
        GUARD;
        auto out = new_ocl_tensor(index.sizes(), self.device(), self.scalar_type());
        return gather_out(self, dim, index, sparse_grad, out);
    }

    // {"schema": "aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor& index_select_out(const Tensor& self, int64_t dim, const Tensor& index, Tensor& out)
    {
        GUARD;
        _check_dim(self.dim(), dim);
        // TODO: split and merge contiguous dims
        return _select(self, dim, index, out, Select::ASSIGN, Select::X);
    }

    // {"schema": "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor", "dispatch": "True", "default": "False"}
    Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index)
    {
        GUARD;
        _check_dim(self.dim(), dim);
        auto sizs(self.sizes().vec());
        if (dim < 0) dim += self.dim();
        sizs[dim] = index.size(0);
        Tensor out = new_ocl_tensor(sizs, self.device(), self.scalar_type());
        return _select(self, dim, index, out, Select::ASSIGN, Select::X);
    }

    // {"schema": "aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor(a!)", "dispatch": "True", "defualt": "False"}
    Tensor& index_add_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Scalar& alpha)
    {
        GUARD;
        _check_dim(self.dim(), dim);
        // TODO: split and merge contiguous dims
        return _select(source, dim, index, self, Select::ADD, Select::Y, true, alpha.toDouble());
    }
    // {"schema": "aten::index_add.out(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor& index_add(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source, const Scalar& alpha, Tensor& out)
    {
        GUARD;
        _check_dim(self.dim(), dim);
        return _select(source, dim, index, out.copy_(self), Select::ADD, Select::Y, true, alpha.toDouble());
    }

    // {"schema": "aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor& scatter_add_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src)
    {
        GUARD;
        _check_dim(self.dim(), dim);
        return _select(src, dim, index, self, Select::ADD, Select::Y);
    }

    // {"schema": "aten::scatter_add.out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
    Tensor& scatter_add(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src, Tensor& out)
    {
        GUARD;
        return scatter_add_(out.copy_(self), dim, index, src);
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
        if (scale_grad_by_freq) {
            auto tp = dind.dtype();
            int64_t W = num_weights.as_int_unchecked();
            WSGuard gd(W*sizeof(cl_uint), indices.device());
            dcnt = gd.ws.workspace_as_type(dlprim::uint32_data);
            dlprim::core::fill_tensor(dcnt, 0.0, e);
            dlprim::Tensor vcnt(dcnt.device_buffer(),dcnt.device_offset(),sp,tp);
            dlprim::core::pointwise_operation({dind},{vcnt},{(double)W},
                "if(x0<w0)atomic_inc(((volatile __global uint*)py0)+py0_offset+x0);return;", e);
        }
        Tensor outc = out.is_contiguous() ? out.zero_() : at::zeros(out.sizes(),out.options());
        // index_add
        _select(grad_output.reshape({(int64_t)sp[0],grad_output.size(-1)}), 0, indc.reshape(-1), outc, Select::ADD, Select::Y);
        int64_t pd = padding_idx.as_int_unchecked();
        // TODO merged kernel
        if (scale_grad_by_freq) {
            dcnt.reshape({dcnt.shape()[0],1});
            dlprim::Tensor vout = todp(outc);
            dlprim::core::pointwise_operation_broadcast({dcnt},{vout},
                {(double)pd}, {dind.dtype()},
                STR(
                    if (x0!=0) {
                        py0+=py0_offset+index.s[0]*limit.s[1]+index.s[1];
                        *py0=index.s[0]==w0?0:*py0/x0;
                    }
                    return;
                ), e);
        } else if (pd>=0) {
            outc.index_put_({pd,indexing::Slice()},0);
        }
        if (!out.is_contiguous()) {
            out.copy_(outc);
        }
        sync_if_needed(out.device());
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
