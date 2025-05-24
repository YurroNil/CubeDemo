// include/resources/texture.h
#pragma once

#include <atomic>
#include <mutex>
#include "resources/textureBase.h"

namespace CubeDemo {

class Texture {
public:
    explicit Texture();

    enum class LoadState {
        Init,           // 初始化
        Uninitialized,  // 初始状态
        Placeholder,    // 使用占位纹理
        Loading,        // 正在异步加载
        Ready,          // 加载完成可用
        Failed          // 最终加载失败
    };

    ~Texture();
    // 禁用构造函数拷贝
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
    // 立即绑定纹理到指定纹理单元
    void Bind(unsigned int slot) const;
    void MarkValid(bool valid);


    std::atomic<bool> m_Valid = true;     // 有效性标志（原子）
    std::atomic<int> m_RetryCount = 0;    // 当前重试次数
    string Type;                          //纹理类型（diffuse/normal等）
    std::atomic<unsigned int> ID = 0;     //OpenGL纹理ID
    //加载任务数
    std::atomic<LoadState> State{LoadState::Uninitialized}; // 当前状态
    string Path;    // 原始文件路径

}; }    // namespace CubeDemo

// 关于LoadState结构体的枚举参数声明
#include "resources/texture.inl"
