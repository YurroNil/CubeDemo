// src/threads/task_queue.cpp
#include "pch.h"
#include "threads/task_queue.h"

namespace CubeDemo {

constexpr int MAX_TASKS_PER_PROCESS = 100; // 处理限制
static TaskQueue s_MainThreadQueue; // 添加全局任务队列

std::atomic<bool> TaskQueue::s_Running{true};

bool TaskQueue::IsMainThread() { return std::this_thread::get_id() == s_MainThreadId; }

// 每帧任务处理
void TaskQueue::ProcTasks(int& processed) {
    processed = 0;
    while (!s_MainThreadQueue.Empty() && processed++ < MAX_TASKS_PER_PROCESS) {
        auto task = s_MainThreadQueue.Pop();
        task();
    }
}
// 发出任务指令
void TaskQueue::AddTasks(Task task, bool is_high_priority) {
    s_MainThreadQueue.Push(std::move(task), is_high_priority);
    std::cout << "[Queue] 添加任务类型: " << typeid(task).name() << " 队列深度: " << sizeof(queue_) << std::endl;
}
// 任务优先级处理
void TaskQueue::Push(Task task, bool is_high_priority) {
    auto& diag = Diagnostic::Get();
    {
        std::lock_guard lock(mutex_);

        if(is_high_priority) queue_.push_front(std::move(task));
        else queue_.push_back(std::move(task));
        

        diag.stats.tasksQueued = queue_.size();
        std::cout << "[TASK] 添加任务 类型:" << typeid(task).name()<< " 优先级:" << (is_high_priority ? "高" : "低") << " 队列深度:" << queue_.size() << "\n";
    }
    condition_.notify_all(); // 必须使用notify_all
}

Task TaskQueue::Pop() {
    auto& diag = Diagnostic::Get();

    std::unique_lock lock(mutex_);
    
    // 记录队列压力
    diag.stats.tasksQueued = queue_.size();

    // 使用条件变量+超时双重机制
    condition_.wait_for(lock, millisec(100), [&]{ return !queue_.empty() || !s_Running; });
    
    if(!s_Running) return nullptr;
    
    if(!queue_.empty()){
        Task task = std::move(queue_.front());
        queue_.pop_front();
        std::cout << "[QUEUE] 取出任务，剩余: " << queue_.size() << "\n";
        return task;
    }
    return nullptr;
}

bool TaskQueue::Empty() const {
    std::lock_guard lock(mutex_);
    return queue_.empty();
}

float TaskQueue::GetQueuePressure() const {
    std::lock_guard lock(mutex_);
    const auto now = csclock::now();
    const auto elapsed = std::chrono::duration_cast<millisec>(
        now - lastEnqueueTime_
    ).count();
    return queue_.size() * 1000.0f / (elapsed + 1); // +1防止除零
}

void TaskQueue::Shutdown() {
    std::lock_guard lock(mutex_);
    while(!queue_.empty()) queue_.pop_front();
    condition_.notify_all();
}

size_t TaskQueue::GetQueueSize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

}   // namespace CubeDemo