// src/graphics/cuda_utils.h
#pragma once

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << " (" #call ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

inline void cuda_sync() {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA sync error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <typename T>
T* cuda_alloc(size_t count) {
    T* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_ptr, count * sizeof(T)));
    return d_ptr;
}

template <typename T>
void cuda_free(T* d_ptr) {
    if (d_ptr) {
        CUDA_CHECK(cudaFree(d_ptr));
    }
}

template <typename T>
void cuda_upload(T* d_dest, const T* h_src, size_t count) {
    CUDA_CHECK(cudaMemcpy(d_dest, h_src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void cuda_download(T* h_dest, const T* d_src, size_t count) {
    CUDA_CHECK(cudaMemcpy(h_dest, d_src, count * sizeof(T), cudaMemcpyDeviceToHost));
}
