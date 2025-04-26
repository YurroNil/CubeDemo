// src/threads/taskQueue.cpp

#include <iostream>
#include "threads/taskQueue.h"

namespace CubeDemo {
constexpr int MAX_TASKS_PER_PROCESS = 100; // 处理限制
static TaskQueue s_MainThreadQueue; // 添加全局任务队列

std::atomic<bool> TaskQueue::s_Running{true};

bool TaskQueue::IsMainThread() { return std::this_thread::get_id() == s_MainThreadId; }

// 每帧任务处理
void TaskQueue::ProcTasks() {
    int processed = 0;
    while (!s_MainThreadQueue.Empty() && processed++ < MAX_TASKS_PER_PROCESS) {
        auto task = s_MainThreadQueue.Pop();
        task();
    }
}
// 发出任务指令
void TaskQueue::AddTasks(std::function<void()> task, bool isHighPriority) {
    s_MainThreadQueue.Push(std::move(task), isHighPriority);
}
// 任务优先级处理
void TaskQueue::Push(Task task, bool isHighPriority) {
    {
        std::cout << "[断点1]" << std::endl;

        std::lock_guard lock(mutex_);
        // 修改为双端队列实现优先级
        if(isHighPriority) {
            queue_.emplace_front(std::move(task));  // 任务放到队头
            std::cout << "[断点2]" << std::endl;
        } else {
            queue_.emplace_back(std::move(task));   // 任务放到队尾
            std::cout << "[断点3]" << std::endl;
        }
        lastEnqueueTime_ = std::chrono::steady_clock::now();
        std::cout << "[DEBUG] 添加任务，当前队列: " << queue_.size()+1 << std::endl;
    }
    condition_.notify_all();// 通知所有的等待线程
    std::cout << "[断点4]" << std::endl;
}

Task TaskQueue::Pop() {
    std::unique_lock lock(mutex_);
    
    // 修改为主动式等待，防止虚假唤醒
    while(s_Running) {
        if(!queue_.empty()) {
            Task task = std::move(queue_.front());
            queue_.pop_front();
            std::cout << "[DEBUG] 取出任务，剩余: " << queue_.size() << std::endl;
            return task;
        }
        
        // 带超时的等待避免永久阻塞
        condition_.wait_for(lock, std::chrono::milliseconds(100), [&]{
            return !queue_.empty() || !s_Running;
        });
    }
    return nullptr; // 正常关闭时返回空任务
}

bool TaskQueue::Empty() const {
    std::lock_guard lock(mutex_);
    return queue_.empty();
}

float TaskQueue::GetQueuePressure() const {
    std::lock_guard lock(mutex_);
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - lastEnqueueTime_
    ).count();
    return queue_.size() * 1000.0f / (elapsed + 1); // +1防止除零
}

void TaskQueue::Shutdown() {
    std::lock_guard lock(mutex_);
    while(!queue_.empty()) queue_.pop_front();
    condition_.notify_all();
}
}   // namespace CubeDemo