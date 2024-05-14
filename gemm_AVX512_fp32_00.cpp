/*

Env: 
  fedora 40 / gcc 14

Build:
  g++ -Wall -O3 -fopenmp -march=native -o gemm gemm_AVX512_fp32_00.cpp

Run:
# for Ryzen 7940HS :
#  - 8 thread
  OMP_NUM_THREADS=8 GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./gemm
#  - 1 thread == serial
  OMP_NUM_THREADS=1 GOMP_CPU_AFFINITY="12" ./gemm

Note:
  the Kernel can be use with [M,N] in [1..16, 1..32]
  
compute C(m,n)=A(m,k)*B(k,n) with some "special" storage:
                          ~: A[m/M][k][M]   B[n/N][k][N]   C[m/M][n][M]  
but "last" collone is store: A     [k][m%M] B     [k][n%N] C     [n][n%M]

the kernel compute:   
for (k=0; k<K; k++)
    C[M,N] += A[M,k] * B[k,N]

*/

#include "tools.hpp"
#include <immintrin.h>

// le Kernel:
using float32_t = float;

// m*n*k / MxN => pA[M,k] pB[N,k] pC[M,n]
template<int M, int N>
void gemm_fp32_MxN(const float32_t* pA, const float32_t* pB, float32_t* pC, const std::size_t K) { 
    static_assert(M>0);
    static_assert(N>0);
    static_assert(M<=32); // pour l'instant 16 ...
    constexpr __mmask16 _m = ((1<<M)-1);

    if constexpr(M<=8) {
        if (K==0) {
            for (std::size_t ij=0; ij<(M*N); ++ij) {
                pC[ij] = 0;
            }
        } else if(K==1) {
            auto A = _mm256_maskz_loadu_ps(_m, pA);
            for (int i=0; i<N; i++) {
                auto B = _mm256_set1_ps(*pB++);
                auto C = _mm256_mul_ps(A,B);
                _mm256_mask_storeu_ps(pC+M*i, _m, C);
            }
        } else {
            __m256 C[N];
            for (std::size_t k=1; k<=K; ++k) {
                __m256 A = _mm256_maskz_loadu_ps(_m, pA+(k-1)*M);
                #pragma GCC unroll 128
                //#pragma omp unroll full
                for (int i=0; i<N; i++) {
                    if (k==1) {
                        //std::cout << "    => ("<<i<<","<<_m<<")"<<std::endl;
                        // init de C
                        C[i] = _mm256_mul_ps(A, _mm256_set1_ps(*pB++));
                    } else if (k==K) {
                        // ecriture de C
                        C[i] = _mm256_fmadd_ps(A, _mm256_set1_ps(*pB++), C[i]);
                        _mm256_mask_storeu_ps(pC+M*i, _m, C[i]);
                    } else {
                        // accumulation
                        C[i] = _mm256_fmadd_ps(A, _mm256_set1_ps(*pB++), C[i]);
                    }
                }
            }
        }
    } else if constexpr(M<=16) {
        if (K==0) {
            for (std::size_t ij=0; ij<(M*N); ++ij) {
                pC[ij] = 0;
            }
        } else if(K==1) {
            auto A = _mm512_maskz_loadu_ps(_m, pA);
            for (int i=0; i<N; i++) {
                auto B = _mm512_set1_ps(*pB++);
                auto C = _mm512_mul_ps(A,B);
                _mm512_mask_storeu_ps(pC+M*i, _m, C);
            }
        } else {
            __m512 C[N];
            for (std::size_t k=1; k<=K; ++k) {
                auto A = _mm512_maskz_loadu_ps(_m, pA+(k-1)*M);
                #pragma GCC unroll 128
                //#pragma omp unroll full
                for (int i=0; i<N; i++) {
                    if (k==1) {
                        C[i] = _mm512_mul_ps(A,_mm512_set1_ps(*pB++));
                    } else if (k==K) {
                        C[i] = _mm512_fmadd_ps(A,_mm512_set1_ps(*pB++),C[i]);
                        _mm512_mask_storeu_ps(pC+M*i, _m, C[i]);
                    } else {
                        C[i] = _mm512_fmadd_ps(A,_mm512_set1_ps(*pB++),C[i]);
                    }
                }
            }
        }
    } else if constexpr(M<=24) {
        // M€[17:24]..  => __m512 + __m256
        static_assert(M<=16, "Cas ou M>16 non codé");
    } else if constexpr(M<=32) {
        // M€[25:32].. => 2 x __m512
        static_assert(M<=24, "Cas ou M>24 non codé");
    } else {
        // M > 32 ...
        static_assert(M<=32, "Cas ou M>32 non codé ca fait vraiment beaucoups");
    }
}

template<int I, int J> constexpr int MIN() { return I<J?I:J; }

template<int M, int N, int M_, int N_>
inline void sgemm_512_bloc_IJ(float* A, float* B, float* C, int m, int n, int k, int m0, int n0) {
    const auto _m = m0*M*k;
    const auto _n = n0*N*k;
    const auto _c = m0*M*n+n0*M_*N;
    gemm_fp32_MxN<M_,N_>(A+_m, B+_n, C+_c, k);
}

template<int M, int N, int N_>
inline void sgemm_512_bloc_I(float* A, float* B, float* C, int m, int n, int k, int m0, int n0) {
    static_assert(M<=16);
    const int M_ = (m-m0*M);
    switch(M_) {
        case  1: sgemm_512_bloc_IJ<M,N,MIN<M, 1>(),N_>(A,B,C,m,n,k,m0,n0); break;
        case  2: sgemm_512_bloc_IJ<M,N,MIN<M, 2>(),N_>(A,B,C,m,n,k,m0,n0); break;
        case  3: sgemm_512_bloc_IJ<M,N,MIN<M, 3>(),N_>(A,B,C,m,n,k,m0,n0); break;
        case  4: sgemm_512_bloc_IJ<M,N,MIN<M, 4>(),N_>(A,B,C,m,n,k,m0,n0); break;
        case  5: sgemm_512_bloc_IJ<M,N,MIN<M, 5>(),N_>(A,B,C,m,n,k,m0,n0); break;
        case  6: sgemm_512_bloc_IJ<M,N,MIN<M, 6>(),N_>(A,B,C,m,n,k,m0,n0); break;
        case  7: sgemm_512_bloc_IJ<M,N,MIN<M, 7>(),N_>(A,B,C,m,n,k,m0,n0); break;
        case  8: sgemm_512_bloc_IJ<M,N,MIN<M, 8>(),N_>(A,B,C,m,n,k,m0,n0); break;
        case  9: sgemm_512_bloc_IJ<M,N,MIN<M, 9>(),N_>(A,B,C,m,n,k,m0,n0); break;
        case 10: sgemm_512_bloc_IJ<M,N,MIN<M,10>(),N_>(A,B,C,m,n,k,m0,n0); break;
        case 11: sgemm_512_bloc_IJ<M,N,MIN<M,11>(),N_>(A,B,C,m,n,k,m0,n0); break;
        case 12: sgemm_512_bloc_IJ<M,N,MIN<M,12>(),N_>(A,B,C,m,n,k,m0,n0); break;
        case 13: sgemm_512_bloc_IJ<M,N,MIN<M,13>(),N_>(A,B,C,m,n,k,m0,n0); break;
        case 14: sgemm_512_bloc_IJ<M,N,MIN<M,14>(),N_>(A,B,C,m,n,k,m0,n0); break;
        case 15: sgemm_512_bloc_IJ<M,N,MIN<M,15>(),N_>(A,B,C,m,n,k,m0,n0); break;
        default: // >=16
            sgemm_512_bloc_IJ<M,N,M,N_>(A,B,C,m,n,k,m0,n0);
            break;
    }
}

template<int M, int N>
void sgemm_512_bloc_J(float* A, float* B, float* C, int m, int n, int k, int m0, int n0) {
    static_assert(N<=32);
    const int N_ = (n-n0*N);
    switch(N_) {
        case  1: sgemm_512_bloc_I<M,N,MIN<N, 1>()>(A,B,C,m,n,k,m0,n0); break;
        case  2: sgemm_512_bloc_I<M,N,MIN<N, 2>()>(A,B,C,m,n,k,m0,n0); break;
        case  3: sgemm_512_bloc_I<M,N,MIN<N, 3>()>(A,B,C,m,n,k,m0,n0); break;
        case  4: sgemm_512_bloc_I<M,N,MIN<N, 4>()>(A,B,C,m,n,k,m0,n0); break;
        case  5: sgemm_512_bloc_I<M,N,MIN<N, 5>()>(A,B,C,m,n,k,m0,n0); break;
        case  6: sgemm_512_bloc_I<M,N,MIN<N, 6>()>(A,B,C,m,n,k,m0,n0); break;
        case  7: sgemm_512_bloc_I<M,N,MIN<N, 7>()>(A,B,C,m,n,k,m0,n0); break;
        case  8: sgemm_512_bloc_I<M,N,MIN<N, 8>()>(A,B,C,m,n,k,m0,n0); break;
        case  9: sgemm_512_bloc_I<M,N,MIN<N, 9>()>(A,B,C,m,n,k,m0,n0); break;
        case 10: sgemm_512_bloc_I<M,N,MIN<N,10>()>(A,B,C,m,n,k,m0,n0); break;
        case 11: sgemm_512_bloc_I<M,N,MIN<N,11>()>(A,B,C,m,n,k,m0,n0); break;
        case 12: sgemm_512_bloc_I<M,N,MIN<N,12>()>(A,B,C,m,n,k,m0,n0); break;
        case 13: sgemm_512_bloc_I<M,N,MIN<N,13>()>(A,B,C,m,n,k,m0,n0); break;
        case 14: sgemm_512_bloc_I<M,N,MIN<N,14>()>(A,B,C,m,n,k,m0,n0); break;
        case 15: sgemm_512_bloc_I<M,N,MIN<N,15>()>(A,B,C,m,n,k,m0,n0); break;
        case 16: sgemm_512_bloc_I<M,N,MIN<N,16>()>(A,B,C,m,n,k,m0,n0); break;
        case 17: sgemm_512_bloc_I<M,N,MIN<N,17>()>(A,B,C,m,n,k,m0,n0); break;
        case 18: sgemm_512_bloc_I<M,N,MIN<N,18>()>(A,B,C,m,n,k,m0,n0); break;
        case 19: sgemm_512_bloc_I<M,N,MIN<N,19>()>(A,B,C,m,n,k,m0,n0); break;
        case 20: sgemm_512_bloc_I<M,N,MIN<N,20>()>(A,B,C,m,n,k,m0,n0); break;
        case 21: sgemm_512_bloc_I<M,N,MIN<N,21>()>(A,B,C,m,n,k,m0,n0); break;
        case 22: sgemm_512_bloc_I<M,N,MIN<N,22>()>(A,B,C,m,n,k,m0,n0); break;
        case 23: sgemm_512_bloc_I<M,N,MIN<N,23>()>(A,B,C,m,n,k,m0,n0); break;
        case 24: sgemm_512_bloc_I<M,N,MIN<N,24>()>(A,B,C,m,n,k,m0,n0); break;
        case 25: sgemm_512_bloc_I<M,N,MIN<N,25>()>(A,B,C,m,n,k,m0,n0); break;
        case 26: sgemm_512_bloc_I<M,N,MIN<N,26>()>(A,B,C,m,n,k,m0,n0); break;
        case 27: sgemm_512_bloc_I<M,N,MIN<N,27>()>(A,B,C,m,n,k,m0,n0); break;
        case 28: sgemm_512_bloc_I<M,N,MIN<N,28>()>(A,B,C,m,n,k,m0,n0); break;
        case 29: sgemm_512_bloc_I<M,N,MIN<N,29>()>(A,B,C,m,n,k,m0,n0); break;
        case 30: sgemm_512_bloc_I<M,N,MIN<N,30>()>(A,B,C,m,n,k,m0,n0); break;
        case 31: sgemm_512_bloc_I<M,N,MIN<N,31>()>(A,B,C,m,n,k,m0,n0); break;
        default: // >=32
            sgemm_512_bloc_I<M,N,N>(A,B,C,m,n,k,m0,n0);
            break;
    }
}

// assume storage is:
//                          ~: A[m/M][k][M]   B[n/N][k][N]   C[m/M][n][M]  
//but "last" collone is store: A     [k][m%M] B     [k][n%N] C     [n][n%M]
void sgemm_512(float* A, float* B, float* C, int m, int n, int k) {
    constexpr int M = 16;  // <=16
    constexpr int N = 16;  // <=32
    
    const int m_b = m/M + (m%M>0?1:0);
    const int n_b = n/N + (n%N>0?1:0);
    
#pragma omp parallel for collapse(2) //  private(C)
    for (int i=0; i<m_b; ++i) {
    for (int j=0; j<n_b; ++j) {
        sgemm_512_bloc_J<M,N>(A,B,C,m,n,k,i,j);
    }
    }
}

////////////////////////////////////////////////////////////////
#ifndef MKL_INT
#define MKL_INT int
#endif

//#define ITERATIONS 1000000
//#define MNK 16
//#define MNK 32
//#define MNK 64
//#define MNK 128
//#define ITERATIONS 100000
//#define MNK 256
#define ITERATIONS 10000
#define MNK 512
//#define ITERATIONS 1000
//#define MNK 1024
//#define ITERATIONS 100
//#define MNK 2048
//#define ITERATIONS 10
//#define MNK 4096
//#define ITERATIONS 2
//#define MNK 8192

MKL_INT m = MNK;
MKL_INT n = MNK;
MKL_INT k = MNK;

float *A[ITERATIONS], *B, *C;

template<int I>
void multiply() {
    for (int i=0; i<I; i++) {
        sgemm_512(A[i], B, C, m, n, k);
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

