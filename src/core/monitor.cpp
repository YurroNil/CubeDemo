// src/core/MONITOR.cpp
#include "pch.h"
#include "core/monitor.h"

namespace CubeDemo::System {
// 获取内存用量
float MONITOR::GetMemoryUsageMB() {
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

// 获取CPU信息(型号, 时钟频率, 核心数)
CPUInfo MONITOR::GetCPUInfo() {
    CPUInfo info;

#if defined(_WIN32)
    // Windows 实现
    char cpuVendor[13] = {0};
    int cpuInfo[4] = {-1};
    
    // 获取CPU厂商
    __cpuid(cpuInfo, 0);
    *reinterpret_cast<int*>(cpuVendor) = cpuInfo[1];
    *reinterpret_cast<int*>(cpuVendor+4) = cpuInfo[3];
    *reinterpret_cast<int*>(cpuVendor+8) = cpuInfo[2];
    info.vendor = cpuVendor;
    
    // 获取CPU品牌
    char cpuBrand[0x40] = {0};
    __cpuid(cpuInfo, 0x80000000);
    unsigned int nExIds = cpuInfo[0];
    
    for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
        __cpuid(cpuInfo, i);
        if (i == 0x80000002)
            memcpy(cpuBrand, cpuInfo, sizeof(cpuInfo));
        else if (i == 0x80000003)
            memcpy(cpuBrand + 16, cpuInfo, sizeof(cpuInfo));
        else if (i == 0x80000004)
            memcpy(cpuBrand + 32, cpuInfo, sizeof(cpuInfo));
    }
    info.brand = cpuBrand;
    
    // 获取核心信息
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    info.logicalCores = sysInfo.dwNumberOfProcessors;
    
    // 物理核心需要特殊处理
    DWORD length = 0;
    GetLogicalProcessorInformation(NULL, &length);
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION* buffer = 
        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*)malloc(length);
    GetLogicalProcessorInformation(buffer, &length);
    
    info.physicalCores = 0;
    for (DWORD i = 0; i < length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); i++) {
        if (buffer[i].Relationship == RelationProcessorCore) {
            info.physicalCores++;
        }
    }
    free(buffer);
    
    // 获取时钟速度 - 修复字符串问题
    HKEY hKey;
    DWORD speedMHz = 0;
    DWORD size = sizeof(speedMHz);
    
    // 使用 ANSI 字符串版本
    const char* subKey = "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0";
    const char* valueName = "~MHz";
    
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, 
        subKey, 
        0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        if (RegQueryValueExA(hKey, valueName, NULL, NULL, 
            reinterpret_cast<LPBYTE>(&speedMHz), &size) == ERROR_SUCCESS) {
            info.clockSpeed = speedMHz / 1000.0f;
        }
        RegCloseKey(hKey);
    } else {
        // 如果注册表读取失败，使用默认值
        info.clockSpeed = 3.0f;
    }
    
#elif defined(__linux__)
    // Linux 实现
    std::ifstream cpuinfo("/proc/cpuinfo");
    string line;
    std::set<string> physicalIds;
    std::set<string> coreIds;
    
    while (std::getline(cpuinfo, line)) {
        if (line.find("vendor_id") == 0) {
            info.vendor = line.substr(line.find(":") + 2);
        } else if (line.find("model name") == 0) {
            info.brand = line.substr(line.find(":") + 2);
        } else if (line.find("physical id") == 0) {
            physicalIds.insert(line);
        } else if (line.find("core id") == 0) {
            coreIds.insert(line);
        } else if (line.find("cpu MHz") == 0) {
            string speedStr = line.substr(line.find(":") + 2);
            float speedMHz = std::stof(speedStr);
            info.clockSpeed = speedMHz / 1000.0f;
        }
    }
    
    info.physicalCores = physicalIds.size() * coreIds.size();
    info.logicalCores = sysconf(_SC_NPROCESSORS_ONLN);
    
    // 如果未能获取时钟速度，使用默认值
    if (info.clockSpeed == 0) {
        info.clockSpeed = 2.5f;
    }
    
#elif defined(__APPLE__)
    // macOS 实现
    char buffer[128];
    size_t size = sizeof(buffer);
    
    // 获取品牌
    if (sysctlbyname("machdep.cpu.brand_string", &buffer, &size, NULL, 0) == 0) {
        info.brand = buffer;
    }
    
    // 获取厂商
    size = sizeof(buffer);
    if (sysctlbyname("machdep.cpu.vendor", &buffer, &size, NULL, 0) == 0) {
        info.vendor = buffer;
    }
    
    // 获取物理核心数
    int physicalCores;
    size = sizeof(physicalCores);
    sysctlbyname("hw.physicalcpu", &physicalCores, &size, NULL, 0);
    info.physicalCores = physicalCores;
    
    // 获取逻辑核心数
    int logicalCores;
    size = sizeof(logicalCores);
    sysctlbyname("hw.logicalcpu", &logicalCores, &size, NULL, 0);
    info.logicalCores = logicalCores;
    
    // 获取时钟速度
    uint64_t freq;
    size = sizeof(freq);
    sysctlbyname("hw.cpufrequency", &freq, &size, NULL, 0);
    info.clockSpeed = freq / 1000000000.0f;
    
#endif
    
    return info;
}

void Runfn() {
    // 获取CPU信息
    CPUInfo cpuInfo = MONITOR::GetCPUInfo();
    
    // 打印系统信息
    std::cout << "===== System Information =====\n";
    std::cout << "CPU Vendor: " << cpuInfo.vendor << "\nCPU: " << cpuInfo.brand << "\n";

    std::cout
        << "核心数: " << cpuInfo.physicalCores << "核"
        << cpuInfo.logicalCores << "线程; 时钟频率: " << cpuInfo.clockSpeed
        << " GHz\n============================="
    << std::endl;
}
}   // namespace CubeDemo::System
