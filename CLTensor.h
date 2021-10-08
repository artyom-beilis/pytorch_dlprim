#pragma once
#include <dlprim/core/common.hpp>

#define OpenCL PrivateUse1
#define AutogradOpenCL AutogradPrivateUse1
#include <c10/core/Storage.h>

#include <mutex>

namespace ptdlprim {
    class CLContextManager {
    public: 
        static CLContextManager &instance()
        {
            static std::once_flag once;
            static CLContextManager inst;
            std::call_once(once,init,&inst);
            return inst;
        }
        static unsigned count()
        {
            return instance().data_.size();
        }
        static dlprim::Context getContext(int id)
        {
            return instance().data(id).ctx;
        }
        static dlprim::ExecutionContext getCommandQueue(int id)
        {
            return instance().data(id).queue;
        }
    private:

        struct DevData {
            bool ready = false; // FIXME make thread safe
            std::string name;
            dlprim::Context ctx;
            dlprim::ExecutionContext queue;
        };



        static void init(CLContextManager *self)
        {
            self->allocate();
        }
        void allocate()
        {
            std::vector<cl::Platform> platforms;
            try {
                cl::Platform::get(&platforms);
            }
            catch(cl::Error &) {
                return;
            }
            for(size_t i=0;i<platforms.size();i++) {
                std::vector<cl::Device> devices;
                try{
                    platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
                }
                catch(cl::Error &)
                {
                    continue;
                }
                for(size_t j=0;j<devices.size();j++) {
                    data_.push_back(DevData());
                    data_.back().name = std::to_string(i) + ":" + std::to_string(j);
                }
            }
        }

        DevData &data(int i)
        {
            if(i < 0)
                i = 0;
            if(i >= int(data_.size()))
                throw std::runtime_error("Invalid Device #" + std::to_string(i));
            DevData &res = data_[i];
            if(res.ready)
                return res;
            res.ctx=dlprim::Context(res.name);
            res.queue = res.ctx.make_execution_context();
            res.ready = true;
            std::cout << "Accessing device #" << i << ":" << res.ctx.name() << std::endl;
            return res;
        }

        
        std::vector<DevData> data_;
    };




    struct OCLAllocator : public c10::Allocator {

        OCLAllocator(cl::Context ctx,c10::Device dev) :
            context_(ctx),
            device_(dev)
        {
        }
        virtual ~OCLAllocator() {
        }

        virtual at::DataPtr allocate(size_t n) const
        {
            cl_int err_code = 0;
            cl_mem ptr = clCreateBuffer(
                context_(),
                CL_MEM_READ_WRITE,
                n,
                nullptr,
                &err_code
            );
            if(!ptr) {
                throw cl::Error(err_code,"Failed to allocate memory");
            }
            return at::DataPtr(ptr,ptr,&OCLAllocator::free_ptr,device_);
        }

        static void free_ptr(void *p)
        {
            cl_mem ptr=static_cast<cl_mem>(p);
            if(ptr != nullptr)
                clReleaseMemObject(ptr);
        }
        cl::Context context_;
        c10::Device device_; 
    };

    inline dlprim::DataType todp(c10::ScalarType tp)
    {
        switch(tp) {
        case c10::kFloat:
            return dlprim::float_data;
        case c10::kLong:
            return dlprim::int64_data;
        case c10::kByte:
            return dlprim::uint8_data;
        default:
            throw std::runtime_error(std::string("Unsupported data type:") + c10::toString(tp));
        }
    }

    inline dlprim::DataType todp(caffe2::TypeMeta meta)
    {
        return todp(meta.toScalarType());

    }

    dlprim::Tensor todp(torch::Tensor const &tt)
    {
        auto ct = tt.contiguous();
        cl_mem p=static_cast<cl_mem>(ct.data_ptr());
        auto sizes = ct.sizes();
        auto offset = ct.storage_offset();
        auto dtype = ct.dtype();
        cl::Buffer buf(p,true);
        dlprim::Shape sp;
        if(sizes.empty())
            sp = dlprim::Shape(1); // scalar
        else
            sp = dlprim::Shape::from_range(sizes.begin(),sizes.end());
        dlprim::Tensor res(buf,offset,sp,todp(dtype));
        return res;
    }


    torch::Tensor new_ocl_tensor(torch::IntArrayRef size,c10::Device dev,c10::ScalarType type=c10::kFloat)
    {
        auto ctx = CLContextManager::getContext(dev.index());
        OCLAllocator alloc(ctx.context(),dev);
        size_t n = 1;
        for(auto const &v:size)
            n*=v;
        dlprim::DataType dt = todp(type);
        size_t mem = n*dlprim::size_of_data_type(dt);
        c10::Storage storage(c10::Storage::use_byte_size_t(),mem,alloc.allocate(mem));

        c10::DispatchKeySet ks = c10::DispatchKeySet{c10::DispatchKey::OpenCL, c10::DispatchKey::AutogradOpenCL};
        
        c10::intrusive_ptr<c10::TensorImpl> impl=c10::make_intrusive<c10::TensorImpl>(
            std::move(storage),
            ks,
            caffe2::TypeMeta::fromScalarType(type));

        impl->set_sizes_contiguous(size);


        return torch::Tensor::wrap_tensor_impl(impl);

    }

    inline dlprim::ExecutionContext getExecutionContext(c10::Device dev)
    {
        return CLContextManager::getCommandQueue(dev.index());
    }
    inline dlprim::ExecutionContext getExecutionContext(torch::Tensor const &t)
    {
        return getExecutionContext(t.device());
    }


} // namespace ptdlprim
