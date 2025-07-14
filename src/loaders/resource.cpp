// src/loaders/resource.cpp
#include "pch.h"
#include "loaders/resource.h"

// 别名
using RL = CubeDemo::Loaders::Resource;
using DTS = CubeDemo::Diagnostic::ThreadState;

namespace CubeDemo {

extern unsigned int DEBUG_INFO_LV;

// 静态成员初始化
std::atomic<bool> RL::s_Running = false;
std::vector<std::thread> RL::s_IOThreads;
TaskQueue RL::s_IOQueue;
TaskQueue RL::s_GPUQueue;

// 初始化
void RL::Init(int ioThreads) {
    s_Running = true;
    if(DEBUG_INFO_LV > 1) std::cout << "[THREAD] 正在启动" << ioThreads << "个IO线程...\n";
    for(int i=0; i<ioThreads; ++i) {

        s_IOThreads.emplace_back([]{
            // Diagnostic记录
            auto& diag = Diagnostic::Get();
            const auto tid = std::this_thread::get_id();
            diag.ReportThreadState(tid, DTS::Created);

            // 线程启动运行循环
            RunningLoop(diag, tid);
        });
    }
}
// s_Running = true时循环执行
void RL::RunningLoop(Diagnostic& diag, const TID& tid) {

// 循环体
while(s_Running) {
    // 报告等待状态
    diag.ReportThreadState(tid, DTS::Waiting);
    diag.stats.tasksQueued = s_IOQueue.GetQueueSize();
    auto task = s_IOQueue.Pop();
    if(!task) continue;

    // 处理任务
    diag.ReportThreadState(tid, DTS::Running);
    try {
        if(DEBUG_INFO_LV > 1) std::cout << "IO线程ID: " << std::this_thread::get_id() << "处理任务...";
        task();
    } catch(const std::exception& e) { if(DEBUG_INFO_LV > 0)  std::cerr<<"[THREAD] 任务处理失败: "<<e.what()<< std::endl; }
    // 线程让步
    std::this_thread::yield();
}
    // 退出循环后报告
    diag.ReportThreadState(tid, DTS::Terminated);
    if(DEBUG_INFO_LV > 1) std::cout << "[THREAD] IO线程退出 ID:" << tid << std::endl;
}

void RL::Shutdown() {
    s_Running = false;
    
    // 清空任务队列
    s_IOQueue.Shutdown();
    s_GPUQueue.Shutdown();
    
    // 等待IO线程退出
    for(auto& thread : s_IOThreads) {
        if(thread.joinable()) thread.join();
    }
    s_IOThreads.clear();
    
    // 处理剩余主线程任务
    TaskQueue::PushTaskSync([] {});
}
}
