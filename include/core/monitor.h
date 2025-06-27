// include/core/monitor.h
#pragma once

#include <set>

// 操作系统特定头文件
#ifdef _WIN32
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
    #include <intrin.h>
#elif defined(__linux__) || defined(__unix__)
    #include <unistd.h>
    #include <sys/sysinfo.h>
    #include <cstring>
#elif defined(__APPLE__)
    #include <sys/sysctl.h>
    #include <sys/types.h>
#endif

namespace CubeDemo::System {

struct CPUInfo {
    string vendor;
    string brand;
    int physicalCores;
    int logicalCores;
    float clockSpeed; // GHz
    string cacheInfo;
};

class MONITOR {
public:
    // 获取当前进程内存使用(MB)
    static float GetMemoryUsageMB();
    // 获取CPU信息
    static CPUInfo GetCPUInfo();
};
}   // namespace CubeDemo::System 
