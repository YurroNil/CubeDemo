// src/threads/resourceLoader.cpp

#include <thread>
#include "threads/resourceLoader.h"

namespace CubeDemo {
// 静态成员初始化
std::atomic<bool> ResourceLoader::s_Running{false};
std::vector<std::thread> ResourceLoader::s_IOThreads;
TaskQueue ResourceLoader::s_IOQueue;
TaskQueue ResourceLoader::s_GPUQueue;


void ResourceLoader::Init(int ioThreads) {
    s_Running = true;
    std::cout << "正在启动" << ioThreads << "个IO线程...\n";
    for(int i=0; i<ioThreads; ++i){

        s_IOThreads.emplace_back([]{
            std::cout << "IO线程ID: " << std::this_thread::get_id() << "启动, ";

            while(s_Running) {
                auto task = s_IOQueue.Pop();

                if(task) { try {
                    std::cout << "IO线程ID: " << std::this_thread::get_id() << "处理任务...";
                    
                    task(); } catch(const std::exception& e) { std::cerr<<"[WARNING] 任务处理失败: "<<e.what()<< std::endl; }

                    std::this_thread::yield();  // 增加线程让步
                }
            }
            std::cout << std::endl;
        });
    }
    
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