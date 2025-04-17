// include/core/taskQueue.h

#pragma once
#include <queue>
#include <mutex>
#include <functional>
#include <condition_variable>

namespace CubeDemo {
using Task = std::function<void()>;


class TaskQueue {
public:

    void Push(Task task);
    Task Pop();
    bool Empty() const;

private:
    std::queue<Task> queue_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
};

}   // namespace CubeDemo