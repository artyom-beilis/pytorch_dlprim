#pragma once
#include <dlprim/core/common.hpp>

#define OpenCL PrivateUse1
#define AutogradOpenCL AutogradPrivateUse1
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/core/Storage.h>

#include <mutex>
#include <cstdint>
#include <list>

namespace ptdlprim {

#ifdef USE_PATCHED_TORCH
    constexpr c10::DeviceType OpenCLDeviceType = c10::DeviceType::OPENCL;
#else
    constexpr c10::DeviceType OpenCLDeviceType = c10::DeviceType::PrivateUse1;
#endif    

    struct CLMemAllocation {

        CLMemAllocation(CLMemAllocation const &) = delete;
        void operator=(CLMemAllocation const &) = delete;
        CLMemAllocation(CLMemAllocation &&) = default;
        CLMemAllocation &operator=(CLMemAllocation &&) = default;
        ~CLMemAllocation() {}

        CLMemAllocation(int id,cl::Context &ctx,std::int64_t length,std::int64_t os) :
            device_id(id),
            size(length),
            orig_size(os),
            buffer(ctx,CL_MEM_READ_WRITE,length)
        {
        }

        int device_id;
        std::int64_t size;
        std::int64_t orig_size;
        cl::Buffer buffer;
    };

    class CLCache {
    public:
        CLCache() {}

        CLCache(CLCache const &) = delete;
        void operator=(CLCache const &) = delete;
        typedef std::map<std::int64_t,std::list<std::unique_ptr<CLMemAllocation> > > allocation_type;
        std::mutex lock;
        allocation_type allocation;

        std::int64_t allocated_size = 0;
        std::int64_t peak_requested_size = 0;
        std::int64_t requested_size = 0;
        std::int64_t cached_size = 0;

        bool reuse_oversized_chunks = getenv("OPENCL_CACHE_OVERSIZED") && atoi(getenv("OPENCL_CACHE_OVERSIZED"));
        bool debug_allocator = (getenv("OPENCL_DEBUG_CACHE") && atoi(getenv("OPENCL_DEBUG_CACHE")));

        void clear();
        static std::uint64_t round(uint64_t v);
        std::unique_ptr<CLMemAllocation> allocate(int id,cl::Context &ctx,int64_t orig_size);
        void release(std::unique_ptr<CLMemAllocation> &&mem);
        void prepare(dlprim::Context &ctx);
    };

    class CLContextManager {
    public: 
        static CLContextManager &instance()
        {
            static std::once_flag once;
            static std::unique_ptr<CLContextManager> inst;
            std::call_once(once,init,inst);
            return *inst;
        }
        ~CLContextManager()
        {
            {
                for(auto &data:data_)
                    data->cache.clear();
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



        static void init(std::unique_ptr<CLContextManager> &self)
        {
            self.reset(new CLContextManager());
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
                    std::unique_ptr<DevData> d(new DevData());
                    data_.push_back(std::move(d));
                    data_.back()->name = std::to_string(i) + ":" + std::to_string(j);
                }
            }
        }

        DevData &data(int i)
        {
            if(i < 0)
                i = 0;
            if(i >= int(data_.size()))
                throw std::runtime_error("Invalid Device #" + std::to_string(i));
            DevData &res = *data_[i];
            if(res.ready)
                return res;
            res.ctx=dlprim::Context(res.name);
            res.queue = res.ctx.make_execution_context();
            res.cache.prepare(res.ctx);
            res.ready = true;
            std::cout << "Accessing device #" << i << ":" << res.ctx.name() << std::endl;
            return res;
        }

        
        std::vector<std::unique_ptr<DevData> > data_;
        bool no_cache_;
    };




} // namespace ptdlprim
