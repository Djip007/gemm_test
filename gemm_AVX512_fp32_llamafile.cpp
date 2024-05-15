//
//                   _   _          ___ _      _   ___
//                  | |_(_)_ _ _  _| _ ) |    /_\ / __|
//                  |  _| | ' \ || | _ \ |__ / _ \\__ \.
//                   \__|_|_||_\_, |___/____/_/ \_\___/
//                             |__/
//
//                    BASIC LINEAR ALGEBRA SUBPROGRAMS
//
//
// This file implements multithreaded CPU matrix multiplication for the
// common contiguous use case C = Aᵀ * B. These kernels are designed to
// have excellent performance[1] for matrices that fit in the CPU cache
// without imposing any overhead such as cache filling or malloc calls.
//
// This implementation does not guarantee any upper bound with rounding
// errors, which grow along with k. Our goal's to maximally exploit the
// hardware for performance, and then use whatever resources remain for
// improving numerical accuracy.
//
// [1] J. Tunney, ‘LLaMA Now Goes Faster on CPUs’, Mar. 2024. [Online].
//     Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].

/*
modification of original referenced on top.

- keep only AVX512
- 

- use same bench as other test.

========================================================
Env: 
  fedora 40 / gcc 14

Build:
  g++ -Wall -O3 -fopenmp -march=native -o gemm gemm_AVX512_fp32_llamafile.cpp

Run:
# for Ryzen 7940HS :
#  - 8 thread
  OMP_NUM_THREADS=8 GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./gemm
#  - 1 thread == serial
  OMP_NUM_THREADS=1 GOMP_CPU_AFFINITY="12" ./gemm


*/

#include "tools.hpp"
#include <immintrin.h>
#include <cassert>

#pragma GCC diagnostic ignored "-Wignored-attributes"

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED ARITHMETIC OPERATIONS

inline float add(float x, float y) {
    return x + y;
}
inline float sub(float x, float y) {
    return x - y;
}
inline float mul(float x, float y) {
    return x * y;
}

inline __m128 add(__m128 x, __m128 y) {
    return _mm_add_ps(x, y);
}
inline __m128 sub(__m128 x, __m128 y) {
    return _mm_sub_ps(x, y);
}
inline __m128 mul(__m128 x, __m128 y) {
    return _mm_mul_ps(x, y);
}

inline __m256 add(__m256 x, __m256 y) {
    return _mm256_add_ps(x, y);
}
inline __m256 sub(__m256 x, __m256 y) {
    return _mm256_sub_ps(x, y);
}
inline __m256 mul(__m256 x, __m256 y) {
    return _mm256_mul_ps(x, y);
}

inline __m512 add(__m512 x, __m512 y) {
    return _mm512_add_ps(x, y);
}
inline __m512 sub(__m512 x, __m512 y) {
    return _mm512_sub_ps(x, y);
}
inline __m512 mul(__m512 x, __m512 y) {
    return _mm512_mul_ps(x, y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED HORIZONTAL SUM

inline float hsum(float x) {
    return x;
}
inline float hsum(__m128 x) {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}
inline float hsum(__m256 x) {
    return hsum(_mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x)));
}
inline float hsum(__m512 x) {
    return _mm512_reduce_add_ps(x);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED MEMORY LOADING

template <typename T, typename U> T load(const U *);

template <> inline float load(const float *p) {
    return *p;
}

template <> inline __m128 load(const float *p) {
    return _mm_loadu_ps(p);
}

template <> inline __m256 load(const float *p) {
    return _mm256_loadu_ps(p);
}

template <> inline __m512 load(const float *p) {
    return _mm512_loadu_ps(p);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// ABSTRACTIONS

/**
 * Computes a * b + c.
 *
 * This operation will become fused into a single arithmetic instruction
 * if the hardware has support for this feature, e.g. Intel Haswell+ (c.
 * 2013), AMD Bulldozer+ (c. 2011), etc.
 */
template <typename T, typename U> inline U madd(T a, T b, U c) {
    return add(mul(a, b), c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT MATRIX MULTIPLICATION

//     tinyBLAS<16, __m512, __m512, float, float, float> tb{A, lda, B, ldb, C, ldc, ith, nth};

template <int KN, typename DOT, typename VECTOR, typename TA, typename TB, typename TC>
class tinyBLAS {
  public:
    tinyBLAS(const TA *A, int lda, const TB *B, int ldb, TC *C, int ldc, int ith, int nth)
        : A(A), B(B), C(C), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int m, int n, int k) {
        mnpack(0, m, 0, n, k);
    }

  private:
    void mnpack(int m0, int m, int n0, int n, int k) {
        int mc, nc, mp, np;
        switch ((std::min(m - m0, 8) << 4) | std::min(n - n0, 8)) {
        case 0x88:
        case 0x87:
        case 0x86:
        case 0x85:
        case 0x78:
        case 0x77:
        case 0x76:
        case 0x75:
        case 0x68:
        case 0x67:
        case 0x66:
        case 0x65:
        case 0x58:
        case 0x57:
        case 0x56:
        case 0x55:
            mc = 5;
            nc = 5;
            gemm<5, 5>(m0, m, n0, n, k);  // N+M, M*N = 10/25 = 0,4
            break;
        case 0x48:
        case 0x47:
        case 0x46:
            mc = 4;
            nc = 6;
            gemm<4, 6>(m0, m, n0, n, k);   // N+M, M*N = 10/24 = 0,416666667
            break;
        case 0x84:
        case 0x74:
        case 0x64:
            mc = 6;
            nc = 4;
            gemm<6, 4>(m0, m, n0, n, k);   // N+M, M*N = 10/24 = 0,416666667
            break;
        case 0x83:
            mc = 8;
            nc = 3;
            gemm<8, 3>(m0, m, n0, n, k);  // N+M, M*N = 11/24 = 0,458333333
            break;
        case 0x38:
            mc = 3;
            nc = 8;
            gemm<3, 8>(m0, m, n0, n, k);   // N+M, M*N = 11/24 = 0,458333333
            break;
        case 0x45:
            mc = 4;
            nc = 5;
            gemm<4, 5>(m0, m, n0, n, k);   // N+M, M*N = 9/20 = 0,45
            break;
        case 0x54:
            mc = 5;
            nc = 4;
            gemm<5, 4>(m0, m, n0, n, k);   // N+M, M*N = 9/20 = 0,45
            break;
        case 0x73:
            mc = 7;
            nc = 3;
            gemm<7, 3>(m0, m, n0, n, k);   // N+M, M*N = 10/21 = 0,476190476
            break;
        case 0x37:
            mc = 3;
            nc = 7;
            gemm<3, 7>(m0, m, n0, n, k);    // N+M, M*N = 10/21 = 0,476190476
            break;
        case 0x44:
            mc = 4;
            nc = 4;
            gemm<4, 4>(m0, m, n0, n, k);     // N+M, M*N = 8/16 = 0,5
            break;
        case 0x63:
            mc = 6;
            nc = 3;
            gemm<6, 3>(m0, m, n0, n, k);    // N+M, M*N = 9/18 = 0,5
            break;
        case 0x36:
            mc = 3;
            nc = 6;
            gemm<3, 6>(m0, m, n0, n, k);    // N+M, M*N = 9/18 = 0,5
            break;
        case 0x53:
            mc = 5;
            nc = 3;
            gemm<5, 3>(m0, m, n0, n, k);    // N+M, M*N = 8/15 = 0,533333333
            break;
        case 0x35:
            mc = 3;
            nc = 5;
            gemm<3, 5>(m0, m, n0, n, k);    // N+M, M*N = 8/15 = 0,533333333
            break;
        case 0x43:
            mc = 4;
            nc = 3;
            gemm<4, 3>(m0, m, n0, n, k);    // N+M, M*N = 7/12 = 0,583333333
            break;
        case 0x34:
            mc = 3;
            nc = 4;
            gemm<3, 4>(m0, m, n0, n, k);    // N+M, M*N = 7/12 = 0,583333333
            break;
        case 0x82:
            mc = 8;
            nc = 2;
            gemm<8, 2>(m0, m, n0, n, k);    // N+M, M*N = 10/16 = 0,625
            break;
        case 0x28:
            mc = 2;
            nc = 8;
            gemm<2, 8>(m0, m, n0, n, k);    // N+M, M*N = 10/16 = 0,625
            break;
        case 0x72:
            mc = 7;
            nc = 2;
            gemm<7, 2>(m0, m, n0, n, k);    // N+M, M*N = 9/14 = 0,642857143
            break;
        case 0x27:
            mc = 2;
            nc = 7;
            gemm<2, 7>(m0, m, n0, n, k);    // N+M, M*N = 9/14 = 0,642857143
            break;
        case 0x62:
            mc = 6;
            nc = 2;
            gemm<6, 2>(m0, m, n0, n, k);    // N+M, M*N = 8/12 = 0,666666667
            break;
        case 0x26:
            mc = 2;
            nc = 6;
            gemm<2, 6>(m0, m, n0, n, k);    // N+M, M*N = 8/12 = 0,666666667
            break;
        case 0x81:
            mc = 8;
            nc = 1;
            gemm<8, 1>(m0, m, n0, n, k);
            break;
        case 0x52:
            mc = 5;
            nc = 2;
            gemm<5, 2>(m0, m, n0, n, k);
            break;
        case 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n, k);
            break;
        case 0x25:
            mc = 2;
            nc = 5;
            gemm<2, 5>(m0, m, n0, n, k);
            break;
        case 0x42:
            mc = 4;
            nc = 2;
            gemm<4, 2>(m0, m, n0, n, k);
            break;
        case 0x24:
            mc = 2;
            nc = 4;
            gemm<2, 4>(m0, m, n0, n, k);
            break;
        case 0x18:
            mc = 1;
            nc = 8;
            gemm<1, 8>(m0, m, n0, n, k);
            break;
        case 0x71:
            mc = 7;
            nc = 1;
            gemm<7, 1>(m0, m, n0, n, k);
            break;
        case 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n, k);
            break;
        case 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n, k);
            break;
        case 0x17:
            mc = 1;
            nc = 7;
            gemm<1, 7>(m0, m, n0, n, k);
            break;
        case 0x61:
            mc = 6;
            nc = 1;
            gemm<6, 1>(m0, m, n0, n, k);
            break;
        case 0x16:
            mc = 1;
            nc = 6;
            gemm<1, 6>(m0, m, n0, n, k);
            break;
        case 0x51:
            mc = 5;
            nc = 1;
            gemm<5, 1>(m0, m, n0, n, k);
            break;
        case 0x41:
            mc = 4;
            nc = 1;
            gemm<4, 1>(m0, m, n0, n, k);
            break;
        case 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n, k);
            break;
        case 0x15:
            mc = 1;
            nc = 5;
            gemm<1, 5>(m0, m, n0, n, k);
            break;
        case 0x14:
            mc = 1;
            nc = 4;
            gemm<1, 4>(m0, m, n0, n, k);
            break;
        case 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n, k);
            break;
        case 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n, k);
            break;
        case 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n, k);
            break;
        case 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n, k);
            break;
        case 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n, k);
            break;
        default:
            return;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np, k);
        mnpack(m0, m, np, n, k);
    }

    template <int RM, int RN> void gemm(int m0, int m, int n0, int n, int k) {
        int ytiles = (m - m0) / RM;
        int xtiles = (n - n0) / RN;
        int tiles = xtiles * ytiles;
        int duty = (tiles + nth - 1) / nth;
        int start = duty * ith;
        int end = start + duty;
        if (end > tiles)
            end = tiles;
        for (int job = start; job < end; ++job) {
            int ii = m0 + job / xtiles * RM;
            int jj = n0 + job % xtiles * RN;
            DOT Cv[RN][RM] = {0};
            for (int l = 0; l < k; l += KN)
                for (int j = 0; j < RN; ++j)
                    for (int i = 0; i < RM; ++i)
                        Cv[j][i] = madd(load<VECTOR>(A + lda * (ii + i) + l), //
                                        load<VECTOR>(B + ldb * (jj + j) + l), //
                                        Cv[j][i]);
            TC Cd[RN][RM];
            for (int j = 0; j < RN; ++j)
                for (int i = 0; i < RM; ++i)
                    Cd[j][i] = hsum(Cv[j][i]);
            for (int j = 0; j < RN; ++j)
                for (int i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = Cd[j][i];
        }
    }

    const TA *const A;
    const TB *const B;
    TC *const C;
    const int lda;
    const int ldb;
    const int ldc;
    const int ith;
    const int nth;
};


static void llamafile_sgemm_impl(int m, int n, int k, const float *A, int lda, const float *B,
                                 int ldb, float *C, int ldc, int ith, int nth) {
    assert(m >= 0);
    assert(n >= 0);
    assert(k >= 0);
    assert(lda >= k);
    assert(ldb >= k);
    assert(ldc >= m);
    assert(nth > 0);
    assert(ith < nth);
    assert(1ll * lda * m <= 0x7fffffff);
    assert(1ll * ldb * n <= 0x7fffffff);
    assert(1ll * ldc * n <= 0x7fffffff);
    assert(!(lda % (64 / sizeof(float))));
    assert(!(ldb % (64 / sizeof(float))));
    tinyBLAS<16, __m512, __m512, float, float, float> tb{A, lda, B, ldb, C, ldc, ith, nth};
    tb.matmul(m, n, k);
}

void llamafile_sgemm(int m, int n, int k, const float *A, int lda, const float *B, int ldb,
                     float *C, int ldc) {
    if (1ll * n*m*k < (64*64*64)) {
        llamafile_sgemm_impl(m, n, k, A, lda, B, ldb, C, ldc, 0, 1);
    } else {
        int nth = sysconf(_SC_NPROCESSORS_ONLN);
        #pragma omp parallel for
        for (int ith = 0; ith < nth; ++ith)
            llamafile_sgemm_impl(m, n, k, A, lda, B, ldb, C, ldc, ith, nth);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////


//#define MNK 16
//#define MNK 32
//#define ITERATIONS 1000000
//int m = 65;
//int n = 65;
//int k = 64;
//#define ITERATIONS 100000
//int m = 125;
//int n = 125;
//int k = 128;
//#define ITERATIONS 100000
//int m = 255;
//int n = 255;
//int k = 256;
#define ITERATIONS 10000
int m = 510;
int n = 510;
int k = 512;
//#define ITERATIONS 1000
//int m = 1025;
//int n = 1025;
//int k = 1024;
//#define ITERATIONS 100
//int m = 2050;
//int n = 2050;
//int k = 2048;
//#define ITERATIONS 10
//#define MNK 4096
//#define ITERATIONS 2
//#define MNK 8192
//int m = MNK;
//int n = MNK;
//int k = MNK;

float *A[ITERATIONS], *B, *C;

template<int I>
void multiply() {
    for (int i=0; i<I; i++) {
        llamafile_sgemm(m, n, k, A[i], k, B, k, C, n);
    }
    volatile float x = C[0];
    (void)x;
}

int main() {
    for (int i=0; i<ITERATIONS; i++) {
        A[i] = new_test_matrix<float>(m, k);
    }
    B = new_test_matrix<float>(n, k);
    C = new_test_matrix<float>(n, m);
    multiply<1>();
    control(n,m,C,(float)k);
    printf("gemm<%d,%d,%d>\n", m,n,k);
    for (int nb=0; nb<5; nb++) {
        BENCH(multiply<ITERATIONS>());
    }
}


