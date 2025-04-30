// include/threads/resourceLoader.h

#pragma once
#include "threads/taskQueue.h"
#include "threads/diagnostic.h"

namespace CubeDemo {

class ResourceLoader {
public:
    static void Init(int ioThreads = 2);
    static void Shutdown();
    static void RunningLoop(Diagnostic& diag, const std::thread::id& tid);

    // CPU任务
    template<typename F>
    static auto EnqueueIOJob(F&& f) -> std::future<decltype(f())> {

        using result_type = decltype(f());
        auto promise = std::make_shared<std::promise<result_type>>();

        // 执行TaskQueue::Push()
        s_IOQueue.Push([promise, func = std::forward<F>(f)]() mutable {
            try {
                if constexpr (std::is_void_v<result_type>) {
                    func(); promise->set_value(); }
                else { promise->set_value(func()); }

            } catch(...) {
                auto eptr = std::current_exception();
                promise->set_exception(eptr);
            }
        }, false);

        return promise->get_future();
    }

    // GPU任务专用（需在主线程执行）
    template<typename F>
    static auto EnqueueGPUJob(F&& f) -> std::future<decltype(f())> {
        using result_type = decltype(f());
        using task_type = std::packaged_task<result_type()>;
        
        auto task = std::make_shared<task_type>(std::forward<F>(f));
        auto future = task->get_future();
        
        TaskQueue::AddTasks([task]() mutable {
            (*task)();
        });
        
        return future;
    }
    
private:
    static std::atomic<bool> s_Running;
    static std::vector<std::thread> s_IOThreads;
    static TaskQueue s_IOQueue;  // 用于IO密集型任务
    static TaskQueue s_GPUQueue; // 用于GPU资源创建
};
}
