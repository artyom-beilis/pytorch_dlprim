#include "utils.h"

namespace ptdlprim {
    typedef void (*enter_type)(char const *name,char const*);
    typedef void (*leave_type)(char const *name);
    static dlprim::ExecutionContext *profiling_queue = nullptr;
    static void enter_default(char const *,char const *name) {
       if(profiling_queue)
            profiling_queue->enter(name);
    }
    static void leave_default(char const *) {
        if(profiling_queue)
            profiling_queue->leave();
    }
    static void enter_log_exception(char const *,char const*){}
    static void leave_log_exception(char const *name)
    {
        if(std::uncaught_exceptions()) {
            std::cerr << "Exception from " << name << std::endl;
        }
    }

    std::atomic<int> indent;

    static void enter_trace(char const *name,char const *)
    {
        int v=indent++;
        for(int i=0;i<v;i++)
            std::cout << "  ";
        std::cout << "in:  " << name << std::endl;
    }
    static void leave_trace(char const *)
    {
        indent--;
    }

    static enter_type enter;
    static leave_type leave;

    static void select_default_calls()
    {
        char *smode = getenv("OPENCL_DEBUG_MODE");
        int mode = smode ? atoi(smode) : 0;
        switch(mode) {
        case 1:
            enter = enter_log_exception;
            leave = leave_log_exception;
            break;
        case 2:
            enter = enter_trace;
            leave = leave_trace;
            break;
        default:
            enter = enter_default;
            leave = leave_default;
        }
    }

    static struct SelectCalls {
        SelectCalls() {
            select_default_calls();
        }
    } select;

    ExecGuard::ExecGuard(char const *name,char const *short_name) : name_(name)
    {
        enter(name_,short_name);
    }
    ExecGuard::~ExecGuard()
    {
        leave(name_);
    }
    void ExecGuard::set_profiling_context(dlprim::ExecutionContext *queue)
    {
        profiling_queue = queue;
    }
};
