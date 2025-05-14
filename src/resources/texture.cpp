// src/resources/texture.cpp

#include "resources/texture.h"
#include "kits/glfw.h"
#include "threads/taskQueue.h"

namespace CubeDemo {

Texture::Texture()
    : ID(0),
      m_Valid(true),
      State(LoadState::Init),
      m_RetryCount(0) {}

Texture::~Texture() {
}

void Texture::Bind(unsigned int slot = 0) const {
    if(!TaskQueue::IsMainThread()) {
        throw std::runtime_error("纹理绑定必须在主线程!");
    }
    
    if(!m_Valid.load(std::memory_order_acquire)) {
        glBindTexture(GL_TEXTURE_2D, 0);
        return;
    }
    
    GLuint current_id = ID.load(); // 原子加载
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_2D, current_id);
}

void Texture::MarkValid(bool valid) {
    m_Valid.store(valid, std::memory_order_release);
    // 状态同步
    if (!valid) State.store(LoadState::Failed);
}

}   // namespace CubeDemo