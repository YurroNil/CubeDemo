// include/core/monitor.h
#pragma once
#ifdef _WIN32
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace CubeDemo {

    class Monitor {
    public:
        static float GetMemoryUsageMB(); // 获取当前进程内存使用(MB)
    };
}