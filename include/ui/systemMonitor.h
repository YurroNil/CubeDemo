// include/ui/systemMonitor.h
#pragma once
#include <windows.h>
#include <string>

class SystemMonitor {
public:
    SystemMonitor();
    ~SystemMonitor();

    // 获取操作系统版本
    std::wstring GetOSVersion();
    
    // 获取内存使用率（百分比）
    DWORD GetMemoryUsage();
    
    // 获取CPU型号
    std::wstring GetProcessorName();


private:
    void Init();
    static BOOL CALLBACK MonitorEnumProc(HWND hwnd, LPARAM lParam);
};
