/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

/*

  hipcc -O3 --offload-arch=gfx1103 -lhipblas gemm_hipblas_01.cpp -o gemm
  HSA_OVERRIDE_GFX_VERSION=11.0.1 ./gemm
  ./gemm

*/

#define HIPBLAS_V2

#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>

#include <iostream>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "tools.hpp"

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPBLAS_ERROR
#define CHECK_HIPBLAS_ERROR(error)                              \
    if(error != HIPBLAS_STATUS_SUCCESS)                         \
    {                                                           \
        fprintf(stderr, "hipBLAS error: ");                     \
        if(error == HIPBLAS_STATUS_NOT_INITIALIZED)             \
            fprintf(stderr, "HIPBLAS_STATUS_NOT_INITIALIZED");  \
        if(error == HIPBLAS_STATUS_ALLOC_FAILED)                \
            fprintf(stderr, "HIPBLAS_STATUS_ALLOC_FAILED");     \
        if(error == HIPBLAS_STATUS_INVALID_VALUE)               \
            fprintf(stderr, "HIPBLAS_STATUS_INVALID_VALUE");    \
        if(error == HIPBLAS_STATUS_MAPPING_ERROR)               \
            fprintf(stderr, "HIPBLAS_STATUS_MAPPING_ERROR");    \
        if(error == HIPBLAS_STATUS_EXECUTION_FAILED)            \
            fprintf(stderr, "HIPBLAS_STATUS_EXECUTION_FAILED"); \
        if(error == HIPBLAS_STATUS_INTERNAL_ERROR)              \
            fprintf(stderr, "HIPBLAS_STATUS_INTERNAL_ERROR");   \
        if(error == HIPBLAS_STATUS_NOT_SUPPORTED)               \
            fprintf(stderr, "HIPBLAS_STATUS_NOT_SUPPORTED");    \
        if(error == HIPBLAS_STATUS_INVALID_ENUM)                \
            fprintf(stderr, "HIPBLAS_STATUS_INVALID_ENUM");     \
        if(error == HIPBLAS_STATUS_UNKNOWN)                     \
            fprintf(stderr, "HIPBLAS_STATUS_UNKNOWN");          \
        fprintf(stderr, "\n");                                  \
        exit(EXIT_FAILURE);                                     \
    }
#endif

//static constexpr size_t ITERATIONS = 100000;
//#define DIM1 256
//#define DIM2 256
//#define DIM3 256

//static constexpr size_t ITERATIONS = 10000;
//#define DIM1 512
//#define DIM2 512
//#define DIM3 512

static constexpr size_t ITERATIONS = 10000;
#define DIM1 1024
#define DIM2 1024
#define DIM3 1024

//static constexpr size_t ITERATIONS = 1000;
//#define DIM1 2048
//#define DIM2 2048
//#define DIM3 2048

//static constexpr size_t ITERATIONS = 100;
//#define DIM1 4096
//#define DIM2 4096
//#define DIM3 4096

//==================================
template<typename T>
T* allocateHost(const std::size_t size) {
  void * ptr;
  HIP_CHECK_ERROR(hipHostMalloc(&ptr, size*sizeof(T), hipHostMallocNonCoherent));
  return reinterpret_cast<T*>(ptr);
}

template<typename T>
void deallocateHost(T * ptr) {
  HIP_CHECK_ERROR(hipHostFree((void*)ptr));
}

template<typename T>
T* getDeviceMem(T* host_adr) {
  void * ptr=nullptr;
  HIP_CHECK_ERROR(hipHostGetDevicePointer(&ptr, host_adr, 0));
  return reinterpret_cast<T*>(ptr);
}

int main()
{
    hipblasOperation_t transa = HIPBLAS_OP_N, transb = HIPBLAS_OP_T;
    //hipblasOperation_t transa = HIPBLAS_OP_T, transb = HIPBLAS_OP_N;

    int    m = DIM1, n = DIM2, k = DIM3;
    int    lda, ldb, ldc;
    size_t size_a, size_b, size_c;
    int    a_stride_1, a_stride_2, b_stride_1, b_stride_2;

    hipblasGemmAlgo_t algo = HIPBLAS_GEMM_DEFAULT;

    // HPA
    using T   = __half;
    using Tex = float;
#ifdef HIPBLAS_V2
    // Compiling with HIPBLAS_V2 in cmake
    hipDataType          a_type       = HIP_R_16F;
    hipDataType          b_type       = HIP_R_16F;
    hipDataType          c_type       = HIP_R_16F;
    hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F;
#else
    hipblasDatatype_t a_type       = HIPBLAS_R_16F;
    hipblasDatatype_t b_type       = HIPBLAS_R_16F;
    hipblasDatatype_t c_type       = HIPBLAS_R_16F;
    hipblasDatatype_t compute_type = HIPBLAS_R_32F;
#endif

    std::cout << "hipblasGemmEx example" << std::endl;
    if(transa == HIPBLAS_OP_N)
    {
        lda        = m;
        size_a     = k * size_t(lda);
        a_stride_1 = 1;
        a_stride_2 = lda;
        std::cout << "N";
    }
    else
    {
        lda        = k;
        size_a     = m * size_t(lda);
        a_stride_1 = lda;
        a_stride_2 = 1;
        std::cout << "T";
    }
    if(transb == HIPBLAS_OP_N)
    {
        ldb        = k;
        size_b     = n * size_t(ldb);
        b_stride_1 = 1;
        b_stride_2 = ldb;
        std::cout << "N: ";
    }
    else
    {
        ldb        = n;
        size_b     = k * size_t(ldb);
        b_stride_1 = ldb;
        b_stride_2 = 1;
        std::cout << "T: ";
    }
    ldc    = m;
    size_c = n * size_t(ldc);

    Tex alpha = 1, beta = 0;

    std::cout << "m, n, k, lda, ldb, ldc = " << m << ", " << n << ", " << k << ", " << lda << ", "
              << ldb << ", " << ldc << std::endl;

    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    T* ha = allocateHost<T>(ITERATIONS*size_a);
    std::vector<T> hb(size_b);
    std::vector<T> hc(size_c);

    // initial data on host
    srand(1);
    for(int i = 0; i < size_a; ++i)
    {
        // random number in [-2, 2]
        ha[i] = __float2half(rand() % 5 - 2.0f);
    }
    for(int i = 0; i < size_b; ++i)
    {
        hb[i] = __float2half(rand() % 5 - 2.0f);
    }
    for(int i = 0; i < size_c; ++i)
    {
        hc[i] = __float2half(rand() % 5 - 2.0f);
    }

    // allocate memory on device
    T *db, *dc;
    T *da[ITERATIONS]; for(size_t i=0; i<ITERATIONS; i++) da[i]=getDeviceMem(&ha[i*m*k]);
    CHECK_HIP_ERROR(hipMalloc(&db, size_b * sizeof(T)));
    CHECK_HIP_ERROR(hipMalloc(&dc, size_c * sizeof(T)));

    hipblasHandle_t handle;
    CHECK_HIPBLAS_ERROR(hipblasCreate(&handle));
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

    for (int r=0; r<5; r++) {
        mesure time; time.start();
        // copy matrices from host to device
        CHECK_HIP_ERROR(hipMemcpy(db, hb.data(), sizeof(T) * size_b, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, hc.data(), sizeof(T) * size_c, hipMemcpyHostToDevice));
        for (size_t u=0; u<ITERATIONS; u++) {
            CHECK_HIPBLAS_ERROR(hipblasGemmEx(handle,
                                              transa,
                                              transb,
                                              m,
                                              n,
                                              k,
                                              &alpha,
                                              da[u],
                                              a_type,
                                              lda,
                                              db,
                                              b_type,
                                              ldb,
                                              &beta,
                                              dc,
                                              c_type,
                                              ldc,
                                              compute_type,
                                              algo));
        }
        // copy output from device to CPU
        CHECK_HIP_ERROR(hipMemcpy(hc.data(), dc, sizeof(T) * size_c, hipMemcpyDeviceToHost));
        HIP_CHECK_ERROR(hipDeviceSynchronize());
        auto dt = time.end();
        printf("%g us %g gigaflops\n", (dt/ITERATIONS)*1000000, (1e-9*2*m*n*k*ITERATIONS)/dt);
    }

    deallocateHost(ha);
    CHECK_HIP_ERROR(hipFree(db));
    CHECK_HIP_ERROR(hipFree(dc));
    CHECK_HIPBLAS_ERROR(hipblasDestroy(handle));
    return EXIT_SUCCESS;
}

