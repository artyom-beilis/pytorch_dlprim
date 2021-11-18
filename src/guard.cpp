#include "utils.h"

namespace ptdlprim {
    typedef void (*enter_type)(char const *name);
    typedef void (*leave_type)(char const *name);
    static void enter_none(char const *) {}
    static void leave_none(char const *) {}

    static void leave_log_exception(char const *name)
    {
        if(std::uncaught_exception()) {
            std::cerr << "Exception from " << name << std::endl;
        }
    }

    std::atomic<int> indent;

    static void enter_trace(char const *name)
    {
        int v=indent++;
        for(int i=0;i<v;i++)
            std::cout << "  ";
        std::cout << "in:  " << name << std::endl;
    }
    static void leave_tace(char const *)
    {
        indent--;
    }

    static enter_type enter;
    static leave_type leave;

    static struct SelectCalls {
        SelectCalls() {
            char *smode = getenv("OPENCL_DEBUG_MODE");
            int mode = smode ? atoi(smode) : 0;
            switch(mode) {
            case 1:
                enter = enter_none;
                leave = leave_log_exception;
                break;
            case 2:
                enter = enter_trace;
                leave = leave_tace;
                break;
            default:
                enter = enter_none;
                leave = leave_none;
            }
        }
    } select;

    ExecGuard::ExecGuard(char const *name) : name_(name)
    {
        enter(name_);
    }
    ExecGuard::~ExecGuard()
    {
        leave(name_);
    }
};
