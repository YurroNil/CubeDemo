// src/graphics/lod.cpp

#include "graphics/lod.h"
#include "loaders/model.h"
#include <future>
#include <iostream>

using ML = CubeDemo::Loaders::Model;
namespace CubeDemo::Graphics {

void LODSystem::Init(
    MeshArray&& base_meshes,     // 基础网格（移动语义转移所有权）
    float bound_rad,       // 包围球半径
    const std::vector<float>& ratios, // 简化比例列表
    SimplifyAlg algorithm       // 简化算法
) {
    std::lock_guard<std::mutex> lock(m_LevelsMutex);
    m_Levels.clear(); // 清空现有层级

    /* 添加原始LOD层级 */
    const float base_trans_dist = bound_rad * 0.5f;
    m_Levels.emplace_back(
        std::move(base_meshes),
        base_trans_dist,
        true
    );

    /* 生成简化层级 */
    for (float ratio : ratios) {
        MeshArray simplified_meshes;
        const auto& src_meshes = m_Levels.front().GetMeshes(); // 使用原始层级的网格

        // 生成简化网格
        for (const auto& mesh : src_meshes) {
            Mesh simplified = algorithm(mesh, ratio);
            simplified_meshes.push_back(std::move(simplified));
        }

        // 计算过渡距离：基于包围球半径和简化比例
        float trans_dist = bound_rad * (1.0f + ratio * 2.0f);
        m_Levels.emplace_back(
            std::move(simplified_meshes),
            trans_dist,
            true
        );
    }
}


// LODLevel实现
LODLevel::LODLevel(LODLevel&& other) noexcept 
    : m_Meshes(std::move(other.m_Meshes)),
      m_TransDistance(other.m_TransDistance),
      m_IsReady(other.m_IsReady.load(std::memory_order_acquire)) 
{
    // 确保原子状态正确转移
    other.m_IsReady.store(false, std::memory_order_release);
}

LODLevel& LODLevel::operator=(LODLevel&& other) noexcept {
    if (this != &other) {
        m_Meshes = std::move(other.m_Meshes);
        m_TransDistance = other.m_TransDistance;
        m_IsReady.store(other.m_IsReady.load(std::memory_order_acquire), std::memory_order_release);
        other.m_IsReady.store(false, std::memory_order_release);
    }
    return *this;
}


LODLevel::LODLevel(MeshArray meshes, float transition_dist, bool ready)
    : m_Meshes(std::move(meshes)),
      m_TransDistance(transition_dist),
      m_IsReady(ready) {}

const MeshArray& LODLevel::GetMeshes() const noexcept {
    return m_Meshes;
}

float LODLevel::GetTransDistance() const noexcept {
    return m_TransDistance;
}

bool LODLevel::IsReady() const noexcept {
    return m_IsReady.load(std::memory_order_acquire);
}

void LODLevel::MarkReady() noexcept {
    m_IsReady.store(true, std::memory_order_release);
}

// LODSystem实现
void LODSystem::AddPrebuiltLevel(LODLevel&& level) {
    std::lock_guard<std::mutex> lock(m_LevelsMutex);
    m_Levels.emplace_back(std::move(level));
}

const LODLevel& LODSystem::SelectLevel(
    const vec3& anchor_point,
    const vec3& observer_pos) const 
{
    std::lock_guard<std::mutex> lock(m_LevelsMutex);
    const float distance = glm::distance(anchor_point, observer_pos);
    
    for (auto it = m_Levels.rbegin(); it != m_Levels.rend(); ++it) {
        if (distance >= it->GetTransDistance() && it->IsReady()) {
            return *it;
        }
    }
    return m_Levels.front(); // 返回最低细节层级
}

const LODLevel& LODSystem::GetLevelByIndex(size_t index) const {
    std::lock_guard<std::mutex> lock(m_LevelsMutex);
    return m_Levels.at(index);
}

const LodLevelArray& LODSystem::GetlevelArray() const {
    std::lock_guard<std::mutex> lock(m_LevelsMutex);
    return m_Levels;
}

size_t LODSystem::GetLevelCount() const noexcept {
    std::lock_guard<std::mutex> lock(m_LevelsMutex);
    return m_Levels.size();
}

void LODSystem::GenLodHierarchy_async(
    const LODLevel& base_level,
    const std::vector<float>& simp_factors,
    SimplifyAlg algorithm)
{
    // 创建异步任务
    auto asyncTask = [this, &base_level, simp_factors, algorithm]() {
        try {
            for (const float ratio : simp_factors) {
                MeshArray simplified_meshes;
                
                // 并行处理每个网格
                std::vector<std::future<Mesh>> futures;
                for (const auto& mesh : base_level.GetMeshes()) {
                    futures.push_back(std::async(std::launch::async, 
                        [&mesh, ratio, &algorithm]() {
                            return algorithm(mesh, ratio);
                        }));
                }
                
                // 收集结果
                for (auto& future : futures) {
                    simplified_meshes.push_back(future.get());
                }
                
                // 创建并添加新层级
                LODLevel new_level(std::move(simplified_meshes),  ratio * base_level.GetTransDistance());
                new_level.MarkReady();
                
                std::lock_guard<std::mutex> lock(m_LevelsMutex);
                m_Levels.push_back(std::move(new_level));
            }
        } catch (const std::exception& e) {
            std::cerr << "LOD生成失败: " << e.what() << std::endl;
        }
    };
    
    std::thread(std::move(asyncTask)).detach();
}

void LODSystem::GenLodHierarchy_sync(
    const LODLevel& base_level,
    const std::vector<float>& simp_factors,
    SimplifyAlg algorithm)
{
    for (const float ratio : simp_factors) {
        MeshArray simplified_meshes;

        for (const auto& mesh : base_level.GetMeshes()) {

            Mesh simplified = Graphics::simplify_mesh(std::move(mesh), ratio);
            simplified.m_textures = mesh.m_textures; 
            simplified_meshes.push_back(std::move(simplified));
        }
        
        std::lock_guard<std::mutex> lock(m_LevelsMutex);
        m_Levels.emplace_back(
            std::move(simplified_meshes), 
            base_level.GetTransDistance() * (1.0f + ratio),
            true // 直接标记为就绪
        );
    }
}

}   // namespace CubeDemo::Graphics
