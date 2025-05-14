// src/utils/MsaConv.cpp
#include "utils/msaConv.h"
#include <iostream>
#include <vector>


namespace CubeDemo::Graphics {

using MeshType = OpenMesh::TriMesh_ArrayKernelT<CubeDemo::FinalTraits>;


// 转换自定义Mesh到OpenMesh格式
void conv_to_openMesh(const CubeDemo::Mesh& src, MeshType& trimesh) {

    trimesh.clear();
    // 在简化前释放所有顶点状态
    trimesh.request_vertex_status();
    trimesh.request_edge_status();
    trimesh.request_face_status();

    // 顶点添加代码
    std::vector<MeshType::VertexHandle> vhandles;
    for (const auto& v : src.Vertices) {
        auto vh = trimesh.add_vertex(MeshType::Point(
            v.Position.x, v.Position.y, v.Position.z));
        
        // 设置法线
        if (trimesh.has_vertex_normals()) {
            trimesh.set_normal(vh, MeshType::Normal(
                v.Normal.x, v.Normal.y, v.Normal.z));
        }

        // 设置纹理坐标
        if (trimesh.has_vertex_texcoords2D()) {
            trimesh.set_texcoord2D(vh, MeshType::TexCoord2D(
                v.TexCoords.x, v.TexCoords.y));
        }

        vhandles.push_back(vh);
    }

    // 添加面
    const auto& indices = src.GetIndices();
    for (size_t i = 0; i < indices.size(); i += 3) {
        if (i+2 >= indices.size()) break;
        
        if (
            indices[i] < vhandles.size() &&
            indices[i+1] < vhandles.size() &&
            indices[i+2] < vhandles.size()
        ) {
            trimesh.add_face(
                vhandles[indices[i]],
                vhandles[indices[i+1]],
                vhandles[indices[i+2]]
            );
        }
    }

    trimesh.update_normals();

    if (!trimesh.is_trimesh()) {
        std::cerr << "[WARNING] 网格包含非三角面！" << std::endl;
        trimesh.triangulate();
    }

}

// 转换OpenMesh到自定义格式
CubeDemo::Mesh conv_from_openMesh(const MeshType& trimesh, const CubeDemo::Mesh& original) {

    CubeDemo::VertexArray vertices;

    for (auto vh : trimesh.vertices()) {
        CubeDemo::Vertex vertex;
        auto point = trimesh.point(vh);
        vertex.Position = {point[0], point[1], point[2]};

        // 获取法线
        if (trimesh.has_vertex_normals()) {
            auto normal = trimesh.normal(vh);
            vertex.Normal = {normal[0], normal[1], normal[2]};
        }

        // 获取纹理坐标
        if (trimesh.has_vertex_texcoords2D()) {
            auto texcoord = trimesh.texcoord2D(vh);
            vertex.TexCoords = {texcoord[0], texcoord[1]};
        }

        vertices.push_back(vertex);
    }

    // 索引收集
    CubeDemo::UnsignedArray indices;
    for (auto fh : trimesh.faces()) {
        for (auto fv : trimesh.fv_range(fh)) {
            indices.push_back(fv.idx());
        }
    }

    std::cout << "[OpenMesh] 转换后顶点数: " << vertices.size() << ", 转换后面数: " << indices.size()/3 << std::endl;

    return CubeDemo::Mesh(vertices, indices, original.m_textures);
}

// 简化网格
CubeDemo::Mesh simplify_mesh(const CubeDemo::Mesh& src, float simplify_ratio) {
    
    if (simplify_ratio <= 0.0f || simplify_ratio >= 1.0f) {

        // 比率不在合适范围, 则跳过网格简化
        return CubeDemo::Mesh(src.Vertices, src.GetIndices(), src.m_textures); // 直接返回原始网格（带纹理）
    }

    try {
        MeshType trimesh;
        conv_to_openMesh(src, trimesh);

        // 初始化简化器
        OpenMesh::Decimater::DecimaterT<MeshType> decimater(trimesh);
        OpenMesh::Decimater::ModQuadricT<MeshType>::Handle hModQuadric;
        
        if (!decimater.add(hModQuadric)) {
            std::cerr << "[MSA] 添加quadric模块失败！" << std::endl;
            return CubeDemo::Mesh(src.Vertices, src.GetIndices(), src.m_textures); // 返回带纹理的原始网格
        }

        if (!decimater.initialize()) {
            std::cerr << "[MSA] Decimater未初始化！" << std::endl;
            return CubeDemo::Mesh(src.Vertices, src.GetIndices(), src.m_textures);
        }

        size_t target = std::max(
            static_cast<size_t>(trimesh.n_vertices() * (1.0f - simplify_ratio)),
            static_cast<size_t>(4) // 至少保留4个顶点以构成面
        );

        std::cout << "[OpenMesh] 简化目标：原顶点=" << src.Vertices.size() << ", 目标顶点=" << target;

        // 执行简化并记录移除的顶点数
        size_t removed = decimater.decimate_to(target);
        trimesh.garbage_collection();

        std::cout << ", 实际移除顶点数: " << removed 
              << ", 剩余顶点数: " << trimesh.n_vertices() << std::endl;

        // 传递原始网格的纹理
        return conv_from_openMesh(trimesh, src);

    } 
    catch (const std::exception& e) {
        std::cerr << "[MSA] Error: " << e.what() << std::endl;
        return CubeDemo::Mesh(src.Vertices, src.GetIndices(), src.m_textures);
    }
}

} // namespace CubeDemo::Graphics
