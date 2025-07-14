// resources/shaders/core/compute/ray_tracing.glsl
#version 460 core

#define EPSILON 0.0001
#define FLT_MAX 3.402823466e+38

layout(local_size_x = 8, local_size_y = 8) in;

// 输出纹理
layout(rgba32f, binding = 0) uniform image2D uOutput;
layout(rgba32f, binding = 1) uniform image2D uAccumulation;

// 场景数据结构
struct TriangleData {
    vec3 v0, v1, v2;
    vec3 n0, n1, n2;
    vec2 uv0, uv1, uv2;
    int materialIndex;
    vec3 emission;
};

struct MaterialData {
    vec3 diffuse;
    vec3 specular;
    vec3 emission;
    float shininess;
    float opacity;
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

layout(std430, binding = 2) buffer BVHBuffer {
    TriangleData triangles[];
};

layout(std430, binding = 3) buffer MaterialBuffer {
    MaterialData materials[];
};

layout(std430, binding = 4) buffer BVHNodes {
    BVHNode nodes[];
};

uniform struct Camera {
    vec3 Position;
    mat4 View;
    mat4 Proj;
} uCamera;

uniform int uSampleCount;

// 光线结构
struct Ray {
    vec3 origin;
    vec3 direction;
    float tmin;
    float tmax;
};

// 命中记录
struct HitRecord {
    float t;
    vec3 position;
    vec3 normal;
    int materialIndex;
};

// 生成相机光线
Ray GenerateRay(vec2 uv) {
    // 计算NDC坐标（范围[-1,1]）
    vec4 rayClip = vec4(uv * 2.0 - 1.0, -1.0, 1.0);
    
    // 转换到观察空间
    vec4 rayEye = inverse(uCamera.Proj) * rayClip;
    rayEye = vec4(rayEye.xy, -1.0, 0.0);
    
    // 转换到世界空间
    vec3 rayDir = normalize(vec3(inverse(uCamera.View) * rayEye));
    
    return Ray(uCamera.Position, rayDir, 0.001, FLT_MAX);
}

// 光线与AABB相交检测
bool RayAABBIntersect(Ray ray, vec3 minBound, vec3 maxBound) {
    vec3 invDir = 1.0 / ray.direction;
    vec3 t0 = (minBound - ray.origin) * invDir;
    vec3 t1 = (maxBound - ray.origin) * invDir;
    
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    
    float tenter = max(tmin.x, max(tmin.y, tmin.z));
    float texit = min(tmax.x, min(tmax.y, tmax.z));
    
    return tenter <= texit && texit > ray.tmin;
}

// 光线与三角形相交（Möller-Trumbore算法）
bool RayTriangleIntersect(Ray ray, TriangleData tri, inout HitRecord rec) {
    vec3 edge1 = tri.v1 - tri.v0;
    vec3 edge2 = tri.v2 - tri.v0;
    vec3 h = cross(ray.direction, edge2);
    float a = dot(edge1, h);
    
    if (a > -EPSILON && a < EPSILON)
        return false; // 光线与三角形平行
    
    float f = 1.0 / a;
    vec3 s = ray.origin - tri.v0;
    float u = f * dot(s, h);
    if (u < 0.0 || u > 1.0)
        return false;
    
    vec3 q = cross(s, edge1);
    float v = f * dot(ray.direction, q);
    if (v < 0.0 || u + v > 1.0)
        return false;
    
    float t = f * dot(edge2, q);
    if (t > ray.tmin && t < ray.tmax && t < rec.t) {
        rec.t = t;
        rec.position = ray.origin + ray.direction * t;
        
        // 使用重心坐标插值法线
        float w = 1.0 - u - v;
        rec.normal = normalize(w * tri.n0 + u * tri.n1 + v * tri.n2);
        rec.materialIndex = tri.materialIndex;
        return true;
    }
    return false;
}

// BVH遍历
bool TraverseBVH(Ray ray, inout HitRecord rec) {
    int stack[32];
    int stackPtr = 0;
    stack[stackPtr++] = 0; // 根节点
    
    bool hit = false;
    
    while (stackPtr > 0) {
        int nodeIdx = stack[--stackPtr];
        BVHNode node = nodes[nodeIdx];
        
        // 检查光线是否与AABB相交
        if (!RayAABBIntersect(ray, node.minBound, node.maxBound)) {
            continue;
        }
        
        // 如果是叶子节点
        if (node.isLeaf == 1) {
            if (node.triangleIndex >= 0) {
                TriangleData tri = triangles[node.triangleIndex];
                if (RayTriangleIntersect(ray, tri, rec)) {
                    hit = true;
                    ray.tmax = rec.t; // 更新光线最大距离
                }
            }
        } else {
            // 内部节点，将子节点加入栈
            if (node.leftChild >= 0) stack[stackPtr++] = node.leftChild;
            if (node.rightChild >= 0) stack[stackPtr++] = node.rightChild;
        }
    }
    
    return hit;
}

// 简单漫反射材质
vec3 DiffuseMaterial(vec3 normal, vec3 lightDir, MaterialData mat) {
    float diff = max(dot(normal, lightDir), 0.0);
    return mat.diffuse * diff;
}

// 镜面反射材质
vec3 SpecularMaterial(vec3 normal, vec3 lightDir, vec3 viewDir, MaterialData mat) {
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), mat.shininess);
    return mat.specular * spec;
}

// 计算颜色
vec3 CalculateColor(Ray ray) {
    HitRecord rec;
    rec.t = ray.tmax;
    rec.materialIndex = -1;
    
    // 使用BVH加速求交
    if (TraverseBVH(ray, rec)) {
        MaterialData mat = materials[rec.materialIndex];
        
        // 简单光照模型
        vec3 lightPos = vec3(10.0, 10.0, 10.0);
        vec3 lightDir = normalize(lightPos - rec.position);
        vec3 viewDir = normalize(ray.origin - rec.position);
        
        // 漫反射
        vec3 diffuse = DiffuseMaterial(rec.normal, lightDir, mat);
        
        // 镜面反射
        vec3 specular = SpecularMaterial(rec.normal, lightDir, viewDir, mat);
        
        // 自发光
        vec3 emission = mat.emission;
        
        return diffuse + specular + emission;
    }
    
    // 天空盒
    float t = 0.5 * (normalize(ray.direction).y + 1.0);
    return mix(vec3(1.0), vec3(0.5, 0.7, 1.0), t);
}

void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(uOutput);
    if (pixel.x >= size.x || pixel.y >= size.y) return;

    // 计算UV坐标（0到1）
    vec2 uv = vec2(pixel) / vec2(size);

    // 生成光线
    Ray ray = GenerateRay(uv);

    // 计算颜色
    vec3 color = CalculateColor(ray);

    // 累加颜色（用于渐进式渲染）
    if (uSampleCount > 0) {
        vec3 accum = imageLoad(uAccumulation, pixel).rgb;
        color = (accum * float(uSampleCount) + color) / float(uSampleCount + 1);
    }

    // 存储到累加纹理
    imageStore(uAccumulation, pixel, vec4(color, 1.0));

    // 输出到目标纹理
    imageStore(uOutput, pixel, vec4(color, 1.0));
}