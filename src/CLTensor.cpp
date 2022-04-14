#include "CLTensor.h"

#ifdef DLPRIM_USE_CL1_HPP
#error "DLPrimitives need to be compiled agaist cl2.hpp in order to work with pytorch. cl.hpp is not supported and known to fail"
#endif


namespace ptdlprim {
    std::uint64_t CLCache::round(uint64_t v)
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

    std::unique_ptr<CLMemAllocation> CLCache::allocate(int id,cl::Context &ctx,int64_t orig_size)
    {
        if(allocated_size == 0 && debug_allocator)
            setlocale(LC_ALL,"");

        std::unique_lock<std::mutex> g(lock);

        std::int64_t size = round(orig_size);
        std::unique_ptr<CLMemAllocation> res;
        
        auto p=allocation.find(size);
        if(reuse_oversized_chunks) {
            int times=0;
            while(p!=allocation.end()) {
                if(!p->second.empty() || times==2) {
                    break;
                }
                ++p;
                times++;
            }
        }

        if(p==allocation.end() || p->second.empty()) {
            res.reset(new CLMemAllocation(id,ctx,size,orig_size));
            allocated_size += res->size;
        }
        else {
            res = std::move(p->second.back());
            TORCH_CHECK(res->size >= orig_size,"Internal validation");
            res->orig_size = orig_size;
            cached_size -= res->size;
            p->second.pop_back();
        }
        requested_size += res->orig_size;
        peak_requested_size = std::max(requested_size,peak_requested_size);
        if(debug_allocator)
            printf("malloc: allocated: %'16ld  requested %'16ld peak-req %'16ld cached %'16ld\n",allocated_size,requested_size,peak_requested_size,cached_size);
        return res;
    }
    void CLCache::release(std::unique_ptr<CLMemAllocation> &&mem)
    {
        std::unique_lock<std::mutex> g(lock);

        int64_t size = mem->size;
        cached_size += mem->size;
        requested_size -= mem->orig_size;
        if(debug_allocator)
            printf("free  : allocated: %'16ld  requested %'16ld peak-req %'16ld cached %'16ld\n",allocated_size,requested_size,peak_requested_size,cached_size);
        allocation[size].push_back(std::move(mem));
    }
    
    void CLCache::clear()
    {
        std::unique_lock<std::mutex> g(lock);
        {
            allocation_type tmp;
            tmp.swap(allocation);
        }
    }
    void CLCache::prepare(dlprim::Context &ctx)
    {
        int64_t mem_size = ctx.device().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
        int64_t rounded_mem_size = round(mem_size);
        for(int64_t size=1;size<=rounded_mem_size;size*=2) {
            allocation[size]; // create empty list
        }
        if(debug_allocator) {
            setlocale(LC_ALL,"");
            printf("GPU max memory allocation size %'15ld creating tables up to %'ld\n",mem_size,rounded_mem_size);
        }
    }
} // namespace

