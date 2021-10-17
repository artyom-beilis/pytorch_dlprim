#pragma once
#include <dlprim/core/common.hpp>

#define OpenCL PrivateUse1
#define AutogradOpenCL AutogradPrivateUse1
#include <c10/core/Storage.h>

#include <mutex>
#include <cstdint>

namespace ptdlprim {
    struct CLMemAllocation {

        CLMemAllocation(CLMemAllocation const &) = delete;
        void operator=(CLMemAllocation const &) = delete;
        CLMemAllocation(CLMemAllocation &&) = default;
        CLMemAllocation &operator=(CLMemAllocation &&) = default;
        ~CLMemAllocation() {}

        CLMemAllocation(int id,cl::Context &ctx,std::int64_t length) :
            device_id(id),
            size(length),
            buffer(ctx,CL_MEM_READ_WRITE,length)
        {
        }

        int device_id;
        std::int64_t size;
        cl::Buffer buffer;
    };

    class CLCache {
    public:
        typedef std::unordered_map<std::int64_t,std::list<std::unique_ptr<CLMemAllocation> > > allocation_type;
        allocation_type allocation;

        void clear()
        {
            {
                allocation_type tmp;
                tmp.swap(allocation);
            }
        }
        static std::uint64_t round(uint64_t v)
        {
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v |= v >> 32;
            v++;
            return v;
        }


        std::unique_ptr<CLMemAllocation> allocate(int id,cl::Context &ctx,int64_t size)
        {
            size = round(size);
            std::unique_ptr<CLMemAllocation> res;
            auto p=allocation.find(size);
            if(p==allocation.end() || p->second.empty()) {
                res.reset(new CLMemAllocation(id,ctx,size));
            }
            else {
                res = std::move(p->second.back());
                p->second.pop_back();
            }
            return res;
        }
        void release(std::unique_ptr<CLMemAllocation> &&mem)
        {
            int64_t size = mem->size;
            allocation[size].push_back(std::move(mem));
        }
    };

    class CLContextManager {
    public: 
        static CLContextManager &instance()
        {
            static std::once_flag once;
            static CLContextManager *inst=nullptr;
            std::call_once(once,init,inst);
            return *inst;
        }
        ~CLContextManager()
        {
            {
                std::vector<DevData> tmp;
                tmp.swap(data_);
                for(DevData &data:tmp)
                    data.cache.clear();
            }
            no_cache_ = true;
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
        static std::unique_ptr<CLMemAllocation> alloc(int id,int64_t size)
        {
            auto &d = instance().data(id);
            return d.cache.allocate(id,d.ctx.context(),size);
        }
        static void release(std::unique_ptr<CLMemAllocation> &&mem)
        {
            auto &inst = instance();
            if(inst.no_cache_) {
                mem.reset();
                return;
            }
            auto &d = instance().data(mem->device_id);
            d.cache.release(std::move(mem));
        }
        static at::DataPtr allocate(c10::Device const &dev,size_t n)
        {
            std::unique_ptr<CLMemAllocation> ptr=alloc(dev.index(),n);
            cl_mem buffer = ptr->buffer();
            return at::DataPtr(buffer,ptr.release(),&CLContextManager::free_ptr,dev);
        }

        static void sync_if_needed(int index)
        {
            auto &inst = instance();
            if(inst.no_cache_) {
                inst.data(index).queue.finish();
            }
        }

        static void free_ptr(void *ctx)
        {
            if(ctx == nullptr)
                return;
            std::unique_ptr<CLMemAllocation> ptr(static_cast<CLMemAllocation *>(ctx));
            release(std::move(ptr));
        }

    private:

        struct DevData {
            bool ready = false; // FIXME make thread safe
            std::string name;
            dlprim::Context ctx;
            dlprim::ExecutionContext queue;
            CLCache cache;
        };



        static void init(CLContextManager *&self)
        {
            self = new CLContextManager();
            self->allocate();
        }
        void allocate()
        {
            char *no_cache=getenv("OPENCL_NO_MEM_CACHE");
            no_cache_ = no_cache && atoi(no_cache);
                
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
        bool no_cache_;
    };


    void sync_if_needed(c10::Device const &d)
    {
        CLContextManager::sync_if_needed(d.index());
    }

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

    inline cl::Buffer buffer_from_tensor(torch::Tensor const &tt)
    {
        TORCH_CHECK(tt.device().type() == c10::DeviceType::OPENCL,"OpenCL device is required for tensor");
        cl_mem p=static_cast<cl_mem>(tt.data_ptr());
        cl::Buffer buf(p,true);
        return buf;
    }
    inline dlprim::Tensor todp(torch::Tensor const &tt)
    {
        TORCH_CHECK(tt.device().type() == c10::DeviceType::OPENCL,"OpenCL device is required for tensor");
        TORCH_CHECK(tt.is_contiguous(),"dlprim::Tensor must be contiguous");
        auto sizes = tt.sizes();
        auto offset = tt.storage_offset();
        auto dtype = tt.dtype();
        cl::Buffer buf = buffer_from_tensor(tt);
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
        size_t n = 1;
        for(auto const &v:size)
            n*=v;
        if(n == 0)
            return torch::Tensor();
        dlprim::DataType dt = todp(type);
        size_t mem = n*dlprim::size_of_data_type(dt);
        c10::Storage storage(c10::Storage::use_byte_size_t(),mem,CLContextManager::allocate(dev,mem));

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
