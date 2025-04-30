// src/threads/resourceLoader.cpp

#include <thread>
#include "threads/resourceLoader.h"
#include <iostream>

namespace CubeDemo {
// 静态成员初始化
std::atomic<bool> ResourceLoader::s_Running{false};
std::vector<std::thread> ResourceLoader::s_IOThreads;
TaskQueue ResourceLoader::s_IOQueue;
TaskQueue ResourceLoader::s_GPUQueue;

// 初始化
void ResourceLoader::Init(int ioThreads) {
    s_Running = true;
    std::cout << "正在启动" << ioThreads << "个IO线程...\n";
    for(int i=0; i<ioThreads; ++i){

        s_IOThreads.emplace_back([]{
            // Diagnostic记录
            auto& diag = Diagnostic::Get();
            const auto tid = std::this_thread::get_id();
            diag.ReportThreadState(tid, Diagnostic::ThreadState::Created);

            std::cout << "IO线程ID: " << std::this_thread::get_id() << "启动, ";

            // 线程启动运行循环
            RunningLoop(diag, tid);
        
        });
    }
}

// s_Running = true时循环执行
void ResourceLoader::RunningLoop(Diagnostic& diag, const std::thread::id& tid) {

int temp_counter = 1;
// 循环体
while(s_Running) {
    // 报告等待状态
    diag.ReportThreadState(tid, Diagnostic::ThreadState::Waiting);
    diag.stats.tasksQueued = s_IOQueue.GetQueueSize();
    auto task = s_IOQueue.Pop();
    if(!task) continue;

    // 处理任务
    diag.ReportThreadState(tid, Diagnostic::ThreadState::Running);
    try {
        std::cout << "IO线程ID: " << std::this_thread::get_id() << "处理任务...";
        task();
    } catch(const std::exception& e) { std::cerr<<"[WARNING] 任务处理失败: "<<e.what()<< std::endl; }
    // 线程让步
    std::this_thread::yield();
    // 调试部分
    std::cout << "[ResourceLoader:RunningLoop] 运行中, 当前帧号: "<< temp_counter << std::endl;
    temp_counter++;
}
    // 退出循环后报告
    diag.ReportThreadState(tid, Diagnostic::ThreadState::Terminated);
    std::cout << "[THREAD] IO线程退出 ID:" << tid << std::endl;
}


void ResourceLoader::Shutdown() {
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