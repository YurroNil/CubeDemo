// include/threads/task_queue.h
#pragma once
#include <condition_variable>

// 乱七八糟的别名
using csclock = std::chrono::steady_clock;
using Task = std::function<void()>;
using millisec = std::chrono::milliseconds;
using TID = std::thread::id;

namespace CubeDemo {
class TaskQueue {
public:
    // 记录运行状态
    static std::atomic<bool> s_Running;

    // 静态方法
    static void ProcTasks(int& processed);
    static void AddTasks(Task task, bool is_high_priority);
    static bool IsMainThread();

    // 实例方法
    void Push(Task task, bool isUrgent);
    Task Pop();
    bool Empty() const;
    float GetQueuePressure() const;
    void Shutdown();
    size_t GetQueueSize() const;
    
    // 无返回值重载
    template<typename Func>
    static auto PushTaskSync(Func&& func);

    // 记录主线程ID
    inline static TID s_MainThreadId;

private:
    std::deque<Task> queue_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    csclock::time_point lastEnqueueTime_;
};

}   // namespace CubeDemo

// 模板具体实现
#include "threads/task_queue.inl"
