// include/core/diagnostic.h

#pragma once
#include <atomic>
#include <map>
#include <mutex>
#include <thread>

namespace CubeDemo {
class Diagnostic {
public:
    // 线程状态枚举
    enum class ThreadState { 
        Created,    // 已创建
        Running,    // 执行任务中
        Waiting,    // 等待任务
        Terminated  // 已终止
    };

    static Diagnostic& Get();
    void ReportThreadState(std::thread::id tid, ThreadState state);
    auto GetThreadStates() const;

    // 资源加载统计
    struct ResourceStats {
        std::atomic<int> texturesLoaded{0};
        std::atomic<int> modelsLoaded{0};
        std::atomic<int> tasksQueued{0};
    } stats;

private:
    mutable std::mutex mutex_;
    std::map<std::thread::id, ThreadState> threadStates_;
};
}