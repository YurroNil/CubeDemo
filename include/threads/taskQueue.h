// include/threads/taskQueue.h

#pragma once
#include <deque>
#include <mutex>
#include <functional>
#include <future>
#include <condition_variable>

namespace CubeDemo {
using Task = std::function<void()>;


class TaskQueue {
public:
    // 记录运行状态
    static std::atomic<bool> s_Running;

    // 静态方法
    static void ProcTasks(int& processed);
    static void AddTasks(std::function<void()> task, bool isHighPriority);
    static bool IsMainThread();

    // 实例方法
    void Push(Task task, bool isUrgent);
    Task Pop();
    bool Empty() const;
    float GetQueuePressure() const;
    void Shutdown();
    
    // 无返回值重载
    template<typename Func>
    static auto PushTaskSync(Func&& func) {
        using R = std::invoke_result_t<Func>;
        // 主线程直接执行
        if(IsMainThread()) { return func(); }
        auto promise = std::make_shared<std::promise<R>>();
        auto future = promise->get_future();
        
        TaskQueue::AddTasks([promise, func = std::forward<Func>(func)]() mutable {
            try {
                if constexpr (std::is_void_v<R>) {
                    func();
                    promise->set_value();
                } else { promise->set_value(func()); }
            } catch(...) { promise->set_exception(std::current_exception()); }
        }, true); // 高优先级

        return future.get();
    }

    // 记录主线程ID
    inline static std::thread::id s_MainThreadId;

private:
    std::deque<Task> queue_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::chrono::steady_clock::time_point lastEnqueueTime_;
};

}   // namespace CubeDemo