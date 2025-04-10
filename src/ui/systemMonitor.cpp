// src/ui/systemMonitor.cpp
#include "ui/systemMonitor.h"
#include <intrin.h>
namespace CubeDemo {

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


// 示例窗口枚举回调函数（可根据需要扩展）
BOOL CALLBACK SystemMonitor::MonitorEnumProc(HWND hwnd, LPARAM lParam) {
    // 这里可以添加窗口监控逻辑
    return TRUE;
}

}