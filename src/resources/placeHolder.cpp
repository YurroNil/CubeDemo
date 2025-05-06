#include "resources/placeHolder.h"

using millisec = std::chrono::milliseconds;

namespace CubeDemo {

namespace {
    constexpr int MAX_RETRY = 3;
    constexpr millisec RETRY_BASE_DELAY{100};
    std::unordered_map<string, int> s_RetryCounters;
    std::mutex s_RetryMutex;
}

// 该类是为了防止没有提供纹理时，以提供一个默认的纹理

TexturePtr PlaceHolder::Create(const string& path, const string& type) {
    auto placeholder = TexturePtr(new Texture());
    placeholder->Path = path;
    placeholder->Type = type;
    placeholder->State.store(LoadState::Placeholder);
    
    std::lock_guard lock(s_TextureMutex);
    s_TexturePool[path] = placeholder;
    return placeholder;
}

void PlaceHolder::ScheAsyncLoad(const string& path, const string& type, TexturePtr placeholder) {
    s_ActiveLoads.fetch_add(1, std::memory_order_relaxed);
    
    RL::EnqueueIOJob([=, self = placeholder]() mutable {
        try {
            auto image_data = IL::Load(path);
            TaskQueue::AddTasks([=, self = std::move(self)] {
                if(!self->m_Valid.load(std::memory_order_acquire)) return;
                
                auto realTex = CreateFromData(image_data, path, type);
                FinalizeTex(self, realTex);
                s_ActiveLoads.fetch_sub(1, std::memory_order_relaxed);
            }, true);
        } catch(...) {
            HandleLoadFailure(path, type, placeholder);
        }
    });
}

void PlaceHolder::FinalizeTex(TexturePtr placeholder, TexturePtr realTex) {
    // 原子交换逻辑

    TaskQueue::PushTaskSync([&]{
        GLuint oldID = placeholder->ID.load();
        placeholder->ID = realTex->ID.load();
        if(oldID != 0) glDeleteTextures(1, &oldID);
    });
    realTex->ID.store(0); // 仅允许在主线程操作后重置

}

// 失败处理
void PlaceHolder::HandleLoadFailure(const string& path, const string& type, TexturePtr placeholder) {
    s_ActiveLoads.fetch_sub(1, std::memory_order_relaxed);
    
    // 重试逻辑
    {
        std::lock_guard retryLock(s_RetryMutex);
        auto& count = s_RetryCounters[path];
        if(++count <= MAX_RETRY) {
            const auto delay = RETRY_BASE_DELAY * (1 << count);
            std::this_thread::sleep_for(delay);
            ScheAsyncLoad(path, type, placeholder);
            return;
        }
        s_RetryCounters.erase(path);
    }
    
    // 最终失败处理
    TaskQueue::AddTasks([=] {
        if(!placeholder->m_Valid.load()) return;
        
        ApplyTex(placeholder);
        placeholder->State.store(LoadState::Failed);
        std::cerr << "[Error] 最终加载失败: " << path << "\n";

    }, true);
}

void PlaceHolder::ApplyTex(TexturePtr tex) {
    static unsigned s_PlaceholderID = CreatePatterns();
    
    std::lock_guard lock(s_TextureMutex);
    tex->ID = s_PlaceholderID;
    tex->State.store(LoadState::Placeholder);
}

unsigned PlaceHolder::CreatePatterns() {
    constexpr int SIZE = 128;
    std::array<unsigned char, SIZE*SIZE*4> pixels;

    // 生成棋盘格图案，作为默认的纹理
    for(int y=0; y<SIZE; ++y) {
        for(int x=0; x<SIZE; ++x) {
            bool isBlack = ((x/16) + (y/16)) % 2 == 0;
            pixels[(y*SIZE + x)*4 + 0] = isBlack ? 0 : 255; // R
            pixels[(y*SIZE + x)*4 + 1] = isBlack ? 0 : 255; // G
            pixels[(y*SIZE + x)*4 + 2] = isBlack ? 0 : 255; // B
            pixels[(y*SIZE + x)*4 + 3] = 255; // A
        }
    }

    unsigned id = 0;
    TaskQueue::PushTaskSync([&]() {

        glGenTextures(1, &id);
        glBindTexture(GL_TEXTURE_2D, id);
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SIZE, SIZE, 0,  GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        
        glGenerateMipmap(GL_TEXTURE_2D);
    });

    return id;
}

}   // namespace CubeDemo