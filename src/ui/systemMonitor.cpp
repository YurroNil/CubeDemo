// src/ui/systemMonitor.cpp
#include "ui/systemMonitor.h"
#include <iostream>
#include <vector>
#include <intrin.h>

SystemMonitor::SystemMonitor() {
    Init();
}

SystemMonitor::~SystemMonitor() {}

void SystemMonitor::Init() { }

// 获取系统版本
std::wstring SystemMonitor::GetOSVersion() {
    wchar_t buffer[256];
    DWORD bufferSize = 256; // 创建DWORD变量
    if (GetComputerNameW(buffer, &bufferSize)) { // 传递地址
        return buffer;
    }
    return L"Unknown";
}
// 获取内存用量
DWORD SystemMonitor::GetMemoryUsage() {
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    GlobalMemoryStatusEx(&statex);
    return statex.dwMemoryLoad;
}
// 获取CPU信息
std::wstring SystemMonitor::GetProcessorName() {
    wchar_t name[256] = {0};
    int cpuInfo[4] = {-1};
    
    __cpuid(cpuInfo, 0x80000002);
    memcpy(name, cpuInfo, sizeof(cpuInfo));
    
    __cpuid(cpuInfo, 0x80000003);
    memcpy(name + 16, cpuInfo, sizeof(cpuInfo));
    
    __cpuid(cpuInfo, 0x80000004);
    memcpy(name + 32, cpuInfo, sizeof(cpuInfo));
    
    return name;
}

// 示例窗口枚举回调函数（可根据需要扩展）
BOOL CALLBACK SystemMonitor::MonitorEnumProc(HWND hwnd, LPARAM lParam) {
    // 这里可以添加窗口监控逻辑
    return TRUE;
}
