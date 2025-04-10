// src/core/systemMonitor.cpp
#include "core/systemMonitor.h"
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif
namespace CubeDemo {


    float SystemMonitor::GetMemoryUsageMB() {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            return pmc.WorkingSetSize / (1024.0f * 1024.0f);
        }
#else
        long rss = 0;
        FILE* fp = nullptr;
        if ((fp = fopen("/proc/self/statm", "r")) == nullptr) return -1;
        if (fscanf(fp, "%*s%ld", &rss) != 1) {
            fclose(fp);
            return -1;
        }
        fclose(fp);
        return (rss * sysconf(_SC_PAGESIZE)) / (1024.0f * 1024.0f);
#endif
        return -1.0f;
    }
}