// include/core/monitor.h
#pragma once

#include <iostream>
#ifdef _WIN32
#include <windows.h>
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