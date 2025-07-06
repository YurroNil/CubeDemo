// src/threads/diagnostic.cpp
#include "pch.h"

namespace CubeDemo {
// 单例访问
Diagnostic& Diagnostic::Get() {
    static Diagnostic instance;
    return instance;
}
// 记录线程状态
void Diagnostic::ReportThreadState(TID tid, ThreadState state) {
    std::lock_guard lock(mutex_);
    threadStates_[tid] = state;
}
// 获取所有线程状态
auto Diagnostic::GetThreadStates() const {
    std::lock_guard lock(mutex_);
    return threadStates_;
}
}
