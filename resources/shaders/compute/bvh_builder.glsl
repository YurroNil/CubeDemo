// resources/shaders/compute/bvh_builder.glsl
#version 460 core

#define FLT_MAX 3.402823466e+38

layout(local_size_x = 256) in;

struct TriangleData {
    vec3 v0, v1, v2;
    vec3 n0, n1, n2;
    vec2 uv0, uv1, uv2;
    int materialIndex;
    vec3 emission;
};

layout(std430, binding = 0) buffer BVHBuffer {
    TriangleData triangles[];
};

// BVH节点结构
struct BVHNode {
    vec3 minBound;
    vec3 maxBound;
    int leftChild;
    int rightChild;
    int triangleIndex;
    int isLeaf;
};

layout(std430, binding = 1) buffer BVHNodes {
    BVHNode nodes[];
};

// 临时结构用于构建BVH
struct AABB {
    vec3 minBound;
    vec3 maxBound;
    int triangleIndex;
};

shared AABB localAABBs[256]; // 使用固定大小避免动态索引

// 计算三角形的AABB
AABB ComputeTriangleAABB(int idx) {
    AABB aabb;
    aabb.minBound = min(min(triangles[idx].v0, triangles[idx].v1), triangles[idx].v2);
    aabb.maxBound = max(max(triangles[idx].v0, triangles[idx].v1), triangles[idx].v2);
    aabb.triangleIndex = idx;
    return aabb;
}

// Morton编码
uint morton3D(vec3 v) {
    const uint BITS = 10;
    const uint MASK = (1u << BITS) - 1;
    
    uint x = uint(v.x * float(MASK));
    uint y = uint(v.y * float(MASK));
    uint z = uint(v.z * float(MASK));
    
    uint code = 0;
    for (uint i = 0; i < BITS; i++) {
        code |= ((x & (1u << i)) << (2 * i)) |
                ((y & (1u << i)) << (2 * i + 1)) |
                ((z & (1u << i)) << (2 * i + 2));
    }
    return code;
}

// 计算AABB中心
vec3 ComputeAABBCenter(AABB aabb) {
    return (aabb.minBound + aabb.maxBound) * 0.5;
}

// 并行排序函数
void ParallelSort(inout AABB data[256]) {
    for (uint i = 0; i < 256; i++) {
        for (uint j = i + 1; j < 256; j++) {
            uint codeI = morton3D(ComputeAABBCenter(data[i]));
            uint codeJ = morton3D(ComputeAABBCenter(data[j]));
            
            if (codeI > codeJ) {
                AABB temp = data[i];
                data[i] = data[j];
                data[j] = temp;
            }
        }
    }
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint localIdx = gl_LocalInvocationID.x;
    
    // 步骤1: 计算每个三角形的AABB
    if (idx < triangles.length()) {
        localAABBs[localIdx] = ComputeTriangleAABB(int(idx));
    } else {
        // 填充无效数据
        localAABBs[localIdx].minBound = vec3(FLT_MAX);
        localAABBs[localIdx].maxBound = vec3(-FLT_MAX);
        localAABBs[localIdx].triangleIndex = -1;
    }
    
    barrier();
    
    // 步骤2: 在本地工作组内排序
    ParallelSort(localAABBs);
    
    barrier();
    
    // 步骤3: 构建BVH节点
    if (localIdx == 0) {
        // 为当前工作组创建BVH节点
        BVHNode node;
        node.minBound = vec3(FLT_MAX);
        node.maxBound = vec3(-FLT_MAX);
        node.leftChild = -1;
        node.rightChild = -1;
        node.triangleIndex = -1;
        node.isLeaf = 0;
        
        // 计算工作组AABB
        for (int i = 0; i < 256; i++) {
            if (localAABBs[i].triangleIndex != -1) {
                node.minBound = min(node.minBound, localAABBs[i].minBound);
                node.maxBound = max(node.maxBound, localAABBs[i].maxBound);
            }
        }
        
        // 存储节点
        uint nodeIndex = gl_WorkGroupID.x;
        nodes[nodeIndex] = node;
        
        // 如果是叶子节点（三角形数量少）
        if (gl_WorkGroupSize.x <= 4) {
            nodes[nodeIndex].isLeaf = 1;
            for (int i = 0; i < 256; i++) {
                if (localAABBs[i].triangleIndex != -1) {
                    nodes[nodeIndex].triangleIndex = localAABBs[i].triangleIndex;
                    break;
                }
            }
        }
    }
    
    barrier();
}