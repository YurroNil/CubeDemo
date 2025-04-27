// src/resources/texture.cpp

#include "resources/texture.h"
#include "utils/glfwKits.h"
#include "threads/taskQueue.h"

namespace CubeDemo {

Texture::Texture() 
    : ID(0), 
      m_Valid(true),
      State(LoadState::Init),
      m_RetryCount(0) {}

Texture::~Texture() {
    // s_TextureAliveCount--;
    // if(m_Valid.exchange(false)) {
    //     // 仅当窗口存在时提交删除任务
    //     if(ID != 0 && Window::GetWindow() != nullptr) {
    //         GLuint tempID;
    //         TaskQueue::AddTasks([id = tempID]{
    //             if(glIsTexture(id)) {
    //                 glDeleteTextures(1, &id);
    //                 UntrackGLObj(id);
    //                 }
    //         }, true);
    //     }
    // }
}

void Texture::Bind(unsigned int slot = 0) const {
    if(!TaskQueue::IsMainThread()) {
        throw std::runtime_error("纹理绑定必须在主线程!");
    }
    
    if(!m_Valid.load(std::memory_order_acquire)) {
        glBindTexture(GL_TEXTURE_2D, 0);
        return;
    }
    
    GLuint currentID = ID.load(); // 原子加载
    glActiveTexture(GL_TEXTURE0 + slot);
    glBindTexture(GL_TEXTURE_2D, currentID);
}


}   // namespace CubeDemo