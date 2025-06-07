// include/threads/task_queue.inl

#pragma once

namespace CubeDemo {

template<typename Func>
auto TaskQueue::PushTaskSync(Func&& func) {
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

}   // namespace CubeDemo
