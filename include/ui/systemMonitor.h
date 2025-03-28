// include/ui/systemMonitor.h
#pragma once
#include "root.h"
#include "streams.h"
#include <windows.h>

class SystemMonitor {
public:
    SystemMonitor();
    ~SystemMonitor();

    // 获取设备信息
    std::wstring GetOSVersion();
    
    // 获取内存使用率（百分比）
    DWORD GetMemoryUsage();
    
    std::wstring GetProcessorName();


private:
    void Init();
    static BOOL CALLBACK MonitorEnumProc(HWND hwnd, LPARAM lParam);
};
