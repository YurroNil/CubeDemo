// include/ui/systemMonitor.h

#pragma once
#include "utils/streams.h"
#include <windows.h>
namespace CubeDemo {

class SystemMonitor {
public:
    SystemMonitor();
    ~SystemMonitor();

    // 获取设备信息
    std::wstring GetOSVersion();
    
    // 获取内存使用率（百分比）
    DWORD GetMemoryUsage();


private:
    void Init();
    static BOOL CALLBACK MonitorEnumProc(HWND hwnd, LPARAM lParam);
};


}