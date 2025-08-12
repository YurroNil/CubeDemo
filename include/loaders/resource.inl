// include/loaders/resource.inl
#pragma once

// 类型别名简化
using RL = CubeDemo::Loaders::Resource;

namespace CubeDemo::Loaders {

// 将IO任务（如文件加载）加入任务队列
template<typename F>
auto RL::EnqueueIOJob(F&& f) -> std::future<decltype(f())> {

    // 推导任务函数的返回类型
    using result_type = decltype(f());
    
    // 创建共享的promise用于异步结果传递
    auto promise = std::make_shared<std::promise<result_type>>();

    // 将任务包装后推入IO任务队列
    s_IOQueue.Push([promise, func = std::forward<F>(f)]() mutable {
        try {
            // 处理void返回类型的特化
            if constexpr (std::is_void_v<result_type>) {
                func(); // 执行无返回值的任务
                promise->set_value(); // 设置promise完成状态
            } else { 
                // 执行有返回值的任务并设置结果
                promise->set_value(func()); 
            }
        } catch(...) {
            // 捕获异常并传递给promise
            auto eptr = std::current_exception();
            promise->set_exception(eptr);
        }
    }, false);  // false表示普通优先级任务

    // 返回future供调用者获取异步结果
    return promise->get_future();
}

// 将GPU任务（需在主线程执行）加入任务队列
template<typename F>
auto RL::EnqueueGPUJob(F&& f) -> std::future<decltype(f())> {
    using result_type = decltype(f());
    // 使用packaged_task自动管理future
    using task_type = std::packaged_task<result_type()>;
    
    // 创建共享的任务包装器
    auto task = std::make_shared<task_type>(std::forward<F>(f));
    // 获取关联的future对象
    auto future = task->get_future();
    
    // 将任务添加到主线程任务队列
    TaskQueue::AddTasks([task]() mutable {
        (*task)();  // 在主线程执行任务
    });
    
    return future;
}
}   // namespace
