/*

Env: 
  fedora 40 / gcc 14

Build:
  g++ -Wall -O3 -fopenmp -march=native -o gemm gemm_AVX512_bf16_00.cpp

Run:
# for Ryzen 7940HS :
#  - 8 thread
  OMP_NUM_THREADS=8 GOMP_CPU_AFFINITY="0,2,4,6,8,10,12,14" ./gemm
#  - 1 thread == serial
  OMP_NUM_THREADS=1 GOMP_CPU_AFFINITY="12" ./gemm

Note:
  the Kernel can be use with N in [4..6] // some reduction missing
  bench with [5,5] [4,6] [6,4]...

------------------------------------
compute C(m,n)=A(m,k)*B(k,n)  only if m%M == n%N == k%32 == 0 !!
  with storage: A[m][k] B[n][k] C[m][n]  (B is transposed)  

the kernel compute:
for (k=0; k<K; k+=32)
    C[1..M,1..N] += A[1..M,k..k+32] * B[k..k+32,1..N]

*/


#include "tools.hpp"
#include <immintrin.h>


using fp32_t = float;

// "simple" define of bf16
# pragma pack(push, 1)
union bf16_t {
  struct {
    unsigned short fraction:7;
    unsigned short exponent:8;  // -127
    bool           sign:1;      // +:false -:true
  } p;
  unsigned short u=0x8000; // +0?
  // auto &sign = p.sign;

   //cast to bfloat16
   bf16_t& operator =(float float_val){
      u = (*reinterpret_cast<unsigned int *>(&float_val))>>16;
      return *this;
   }

};
#pragma pack(pop)

// load N fp32_t convert in bf16_t and pad with 0
//  => return: bf16_t[32]
template <int N>
inline auto load(const fp32_t *X) {
    static_assert(N<=32,"no vector with N>32");
    if constexpr (N<16) {
        constexpr __mmask16 _m = ((1<<N)-1);
        __m512 x1 = _mm512_maskz_loadu_ps(_m, X);
        __m512 x2 = {0};
        return _mm512_cvtne2ps_pbh(x2,x1);
    } else 
    if constexpr (N==16) {
        auto x1 = _mm512_loadu_ps(X);
        __m512 x2 = {0};
        return _mm512_cvtne2ps_pbh(x2,x1);
    } else 
    if constexpr (N<32) {
        constexpr __mmask16 _m = ((1<<(N-16))-1);
        auto   x1 = _mm512_loadu_ps(X);
        __m512 x2 = _mm512_maskz_loadu_ps(_m, X+16);
        return _mm512_cvtne2ps_pbh(x2,x1);
    } else { // N==32
        auto x1 = _mm512_loadu_ps(X);
        auto x2 = _mm512_loadu_ps(X+16);
        return _mm512_cvtne2ps_pbh(x2,x1);
    }
}

// load N bf16_t and pad with 0
//  => return: bf16_t[32]
template <int N>
inline auto load(const bf16_t *X) {
    static_assert(N<=32,"no vector with N>32");
    if constexpr (N<32) {
        constexpr __mmask32 _m = ((1<<N)-1);
        return (__m512bh) _mm512_maskz_loadu_epi16(_m, X);
    } else { // N==32
        return (__m512bh) _mm512_loadu_epi16(X);
    }
}

// convert to bf16 to add, loose some precession but much faster!
// somme 4 vector of fp32_t[16] 
// return vector of fp32_t[16] with:
//  res[0] = x1[0]+x1[1]+...+x1[15]
//  res[1] = x2[0]+x2[1]+...+x2[15]
//  res[2] = x3[0]+x3[1]+...+x3[15]
//  res[3] = x4[0]+x4[1]+...+x4[15]
// use dot2 on bf16 for reduce. 
inline auto hadd(__m512 x1, __m512 x2, __m512 x3, __m512 x4) {
    __m512              un = _mm512_set_ps(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
    __m512bh           _un = _mm512_cvtne2ps_pbh(un, un);  // comment generer ca de facon "static"!
    constexpr __m512  zero = {0};
    // OK on va reduire tout ca avec des produits scalaires...
    auto _x12 = _mm512_cvtne2ps_pbh(x2,x1);
    auto _x34 = _mm512_cvtne2ps_pbh(x4,x3);
    auto x12 = _mm512_dpbf16_ps(zero, _x12, _un);      // 16 => 8;
    auto x34 = _mm512_dpbf16_ps(zero, _x34, _un);      // 16 => 8;
    auto _x1234 = _mm512_cvtne2ps_pbh(x34,x12);
    auto x1234 = _mm512_dpbf16_ps(zero, _x1234, _un);  // 8 => 4;
    _x1234 = _mm512_cvtne2ps_pbh(zero, x1234);
    x1234 = _mm512_dpbf16_ps(zero, _x1234, _un);  // 4 => 2;
    _x1234 = _mm512_cvtne2ps_pbh(zero, x1234);
    x1234 = _mm512_dpbf16_ps(zero, _x1234, _un);  // 2 => 1;
    return x1234;
}
inline auto hadd(__m512 x1, __m512 x2, __m512 x3, __m512 x4, __m512 x5) {
    __m512              un = _mm512_set_ps(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
    __m512bh           _un = _mm512_cvtne2ps_pbh(un, un);  // comment generer ca de facon "static"!
    constexpr __m512  zero = {0};
    // OK on va reduire tout ca avec des produits scalaires...
    auto _x12 = _mm512_cvtne2ps_pbh(x2,x1);
    auto _x34 = _mm512_cvtne2ps_pbh(x4,x3);
    auto _x5  = _mm512_cvtne2ps_pbh(zero,x5);
    auto x12 = _mm512_dpbf16_ps(zero, _x12, _un);      // 16 => 8;
    auto x34 = _mm512_dpbf16_ps(zero, _x34, _un);      // 16 => 8;
         x5  = _mm512_dpbf16_ps(zero, _x5, _un);      // 16 => 8;
    auto _x1234 = _mm512_cvtne2ps_pbh(x34,x12);
    auto x1234  = _mm512_dpbf16_ps(zero, _x1234, _un); // 8 => 4;
    _x5         = _mm512_cvtne2ps_pbh(zero,x5);
    x5          = _mm512_dpbf16_ps(zero, _x5, _un);    // 8 => 4;
    auto _x12345 = _mm512_cvtne2ps_pbh(x5,x1234);
    auto  x12345 = _mm512_dpbf16_ps(zero, _x12345, _un);  // 4 => 2;
    _x12345 = _mm512_cvtne2ps_pbh(zero,x12345);
    x12345 = _mm512_dpbf16_ps(zero, _x12345, _un);  // 2 => 1;
    return x12345;
}
inline auto hadd(__m512 x1, __m512 x2, __m512 x3, __m512 x4, __m512 x5, __m512 x6) {
    __m512              un = _mm512_set_ps(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
    __m512bh           _un = _mm512_cvtne2ps_pbh(un, un);  // comment generer ca de facon "static"!
    constexpr __m512  zero = {0};
    // OK on va reduire tout ca avec des produits scalaires...
    auto _x12 = _mm512_cvtne2ps_pbh(x2,x1);
    auto _x34 = _mm512_cvtne2ps_pbh(x4,x3);
    auto _x56 = _mm512_cvtne2ps_pbh(x6,x5);
    auto  x12 = _mm512_dpbf16_ps(zero, _x12, _un); // 16 => 8;
    auto  x34 = _mm512_dpbf16_ps(zero, _x34, _un); // 16 => 8;
    auto  x56 = _mm512_dpbf16_ps(zero, _x56, _un); // 16 => 8;
    auto _x14 = _mm512_cvtne2ps_pbh(x34,x12);
    auto  x14 = _mm512_dpbf16_ps(zero, _x14, _un); // 8 => 4;
    _x56      = _mm512_cvtne2ps_pbh(zero,x56);
    x56       = _mm512_dpbf16_ps(zero, _x56, _un); // 8 => 4;
    auto _x16 = _mm512_cvtne2ps_pbh(x56,x14);
    auto  x16 = _mm512_dpbf16_ps(zero, _x16, _un); // 4 => 2;
    _x16      = _mm512_cvtne2ps_pbh(zero,x16);
    x16       = _mm512_dpbf16_ps(zero, _x16, _un); // 2 => 1;
    return x16;
}

// write C after last reduction
template<typename... T>
inline void store(fp32_t *pX, T&&... x) {
    constexpr __mmask16 _m = ((1<<sizeof...(T))-1);
    auto pack = hadd(std::forward<T>(x)...);
    _mm512_mask_storeu_ps(pX, _m, pack);
}

template<typename... T>
inline void store(bf16_t *pX, T&&... x) {
    constexpr __mmask32 _m = ((1<<sizeof...(T))-1);
    auto pack = hadd(std::forward<T>(x)...);
    auto _pack = _mm512_cvtne2ps_pbh(pack,pack);
    _mm512_mask_storeu_epi16(pX, _m, (__m512i)_pack); // pas de _mm512_mask_storeu_bf
}

// always use it for compute gemm !
// C[i] += A[2i]*B[2i] + A[2i+1]*B[2i+1]
inline auto madd(const __m512bh& A, const __m512bh& B, const __m512& C) {
    return _mm512_dpbf16_ps(C, A, B);
}
inline auto madd(const __m256bh& A, const __m256bh& B, const __m256& C) {
    return _mm256_dpbf16_ps(C, A, B);
}
inline auto madd(const __m128bh& A, const __m128bh& B, const __m128& C) {
    return _mm_dpbf16_ps(C, A, B);
}

template<int M, int N, typename TA, typename TB, typename TC>
void gemm(const TA *pA, const TB *pB, TC *pC, std::size_t ldc, std::size_t K) {
    using v_t = decltype(load<32>(std::declval<const fp32_t *>()));
    //== using v_t = __m512;
    static_assert(N>0);
    static_assert(M>0);
    // A[?,K]
    // B[?,K]
    // C[?,ldc]
    if constexpr(M==1 && N==1) {
        // TODO:
    } else {
        __m512 C[M][N];
        v_t A[M];
        v_t B;
        for(int j=0; j<N; j++) {
            for(int i=0; i<M; i++) {
                C[i][j] = _mm512_setzero_ps();
            }
        }
        for (std::size_t k=0; k<K; k+=32) {
          for(int i=0; i<M; i++) {
              A[i] = load<32>(pA+i*K+k);
          }
          for(int j=0; j<N; j++) {
            B = load<32>(pB+j*K+k);
            for(int i=0; i<M; i++) {
                C[i][j] = madd(A[i], B, C[i][j]);
            }
          }
        }
        
        // reduce and store C res.
        for(int i=0; i<M; i++) {
          if constexpr (N==4) {
              store(pC+ldc*i, C[i][0], C[i][1], C[i][2], C[i][3]);
          }
          if constexpr (N==5) {
              store(pC+ldc*i, C[i][0], C[i][1], C[i][2], C[i][3], C[i][4]);
          }
          if constexpr (N==6) {
              store(pC+ldc*i, C[i][0], C[i][1], C[i][2], C[i][3], C[i][4], C[i][5]);
          }
        }
    }
}


template<int M, int N, int _M=M, int _N=N, typename TA, typename TB, typename TC>
inline void sgemm_512_bloc(TA* A, TB* B, TC* C, int m, int n, int k, int m0, int n0) {
    // choix du kernel:
    if constexpr ((_M==M) && (_N==N)) { // seul cas traité pour l'instant
        const int Mr = (m-m0*M);
        const int Nr = (n-n0*N);
        if ( Mr>=_M && Nr>=_N) {
            // OK c'est la bonne taille.
            const auto _m = m0*M*k;
            const auto _n = n0*N*k;
            const auto _c = m0*M*n+n0*N;  // C[m][n] ...  n se suit...n0
            gemm<_M,_N>(A+_m, B+_n, C+_c, n, k);
        }
    }
}


template<typename TA, typename TB, typename TC>
void sgemm_512(TA* A, TB* B, TC* C, int m, int n, int k) {
    constexpr int M = 5;
    constexpr int N = 5;
    
    const int m_b = m/M + (m%M>0?1:0);
    const int n_b = n/N + (n%N>0?1:0);
    
#   pragma omp parallel for collapse(2) //  private(C)
    for (int i=0; i<m_b; ++i) {
        for (int j=0; j<n_b; ++j) {
            sgemm_512_bloc<M,N>(A,B,C,m,n,k,i,j);
        }
    }
}

//#define ITERATIONS 100000
//int m = 255;
//int n = 255;
//int k = 256;

#define ITERATIONS 10000
int m = 510;
int n = 510;
int k = 512;

//#define ITERATIONS 1000
//int m = 1000;
//int n = 1000;
//int k = 1024;

using type_t = bf16_t;
//using type_t = fp32_t;

type_t *A[ITERATIONS], *B, *C;

template<int I>
void multiply() {
    for (int i=0; i<I; i++) {
        sgemm_512(A[i], B, C, m, n, k);
    }
    volatile type_t x = C[0];
    (void)x;
}

int main() {
    for (int i=0; i<ITERATIONS; i++) {
        A[i] = new_test_matrix<type_t>(m, k);
    }
    B = new_test_matrix<type_t>(n, k);
    C = new_test_matrix<type_t>(m, n);
    multiply<1>();
    //control(m,n,C,(float)k);
    printf("gemm<%d,%d,%d>\n", m,n,k);
    for (int nb=0; nb<5; nb++) {
        BENCH(multiply<ITERATIONS>());
    }
}

