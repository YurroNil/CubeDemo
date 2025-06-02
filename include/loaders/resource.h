// include/loaders/resource.h

#pragma once
#include "loaders/base.h"
#include "threads/taskQueue.h"

namespace CubeDemo {

class Loaders::Resource {
public:
    static void Init(int ioThreads = 2);
    static void Shutdown();
    static void RunningLoop(Diagnostic& diag, const TID& tid);

    // CPU任务
    template<typename F>
    static auto EnqueueIOJob(F&& f) -> std::future<decltype(f())>;

    // GPU任务专用（需在主线程执行）
    template<typename F>
    static auto EnqueueGPUJob(F&& f) -> std::future<decltype(f())>;
    
private:
    static std::atomic<bool> s_Running;
    static std::vector<std::thread> s_IOThreads;
    static TaskQueue s_IOQueue;  // 用于IO密集型任务
    static TaskQueue s_GPUQueue; // 用于GPU资源创建
};
}
// 模板实现
#include "loaders/resource.inl"
