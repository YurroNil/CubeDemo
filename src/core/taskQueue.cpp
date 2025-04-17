// src/core/taskQueue.cpp

#include <iostream>
#include "core/taskQueue.h"

namespace CubeDemo {


void TaskQueue::Push(Task task) {
    std::cout << "提交任务到队列（当前深度：" << queue_.size() << "）\n";

    {
        std::lock_guard lock(mutex_);
        queue_.push(std::move(task));
    }
    condition_.notify_one();
}

Task TaskQueue::Pop() {
    std::unique_lock lock(mutex_);
    condition_.wait(lock, [&]{ return !queue_.empty(); });
    Task task = std::move(queue_.front());
    queue_.pop();
    std::cout << "处理队列任务（剩余：" << queue_.size() << "）\n";
    return task;
}

bool TaskQueue::Empty() const {
    std::lock_guard lock(mutex_);
    return queue_.empty();
}


}   // namespace CubeDemo