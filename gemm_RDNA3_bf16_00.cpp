/*
Env: 
  fedora 40 / hip 6.0

Build:
  hipcc -O3 --offload-arch=gfx1103 gemm_RDNA3_bf16_00.cpp -o gemm

Run:
# for Ryzen 7940HS :
   ./gemm

Note:
  this kernel use WMMA inst on RDNA3 AMD GPU. it is test and design for APU (eGPU) and use host RAM.
  the alloc nead to be donne by HIP for speed, but can be use as normal memory on host without copy!
  
  only mutiple 16x16 matrice are alowed with this benchmark.

*/

#include "tools.hpp"
#include <hip/hip_runtime.h>

// les types... 
typedef __bf16 bfloat16_t;
typedef __bf16 bfloat16x16_t __attribute__((ext_vector_type(16)));
typedef float  float32_t;
typedef float  float32x8_t   __attribute__((ext_vector_type(8)));

// to load A/B matrice from RAM
template <int NB, int SIZE, int K0>
__device__ inline void load(const bfloat16_t* X, bfloat16_t X_frag[SIZE][K0+2], int I0, int i0, int k0, int k, int K) {
    #pragma unrool
    for (int i=0; i<NB; ++i) {
        X_frag[I0+i][k] = X[K*(i0+i) + k0+k];
    }
}

// the hip kernel
template<int M2, int N2, int M1=1, int N1=1, int M0=16, int N0=16, int K0=16>
__global__ void __launch_bounds__(16*2*M2*N2) wmma_matmul(const bfloat16_t* __restrict__ a, const bfloat16_t* __restrict__ b, float32_t* __restrict__ c, size_t I, size_t J, size_t K)
{
    // onnly possible values!
    static_assert(M0==16);
    static_assert(N0==16);
    static_assert(K0==16);

    //  block:  C[i0:i0+M*16,j0:j0+N*16];
    const int I0 = blockIdx.x*M0*M1*M2;
    const int J0 = blockIdx.y*N0*N1*N2;
    const int lane = threadIdx.x;    // == M0/N0/K0
    const int wave = threadIdx.y;    // 0 ou 1 suivant si on trait les paires ou les impaires.
    // threadIdx.z  € [0..M2*N2]
    const int i2 = (threadIdx.z%M2) * M0*M1;  // [0..M1]*16*M1
    const int j2 = (threadIdx.z/M2) * N0*N1;  // [0..N1]*16*N1
    
    // strategie: lane<=>k ; les autre sont a repartir sur [0..M1*N1]
    constexpr int M_size    = M0*M1*M2;
    constexpr int N_size    = N0*N1*N2;

    constexpr int NB_BLOC   = 2*M2*N2; // == nb_tread/nb_lane(16)/2 wave...
    constexpr int BLOC_A    = M_size/NB_BLOC;  static_assert(M_size%NB_BLOC == 0);  // la taille du bloc (/lane)
    constexpr int BLOC_B    = N_size/NB_BLOC;  static_assert(N_size%NB_BLOC == 0);
    const int IA = (2*threadIdx.z+wave)*BLOC_A;
    const int JB = (2*threadIdx.z+wave)*BLOC_B;
    
    __shared__ bfloat16_t A_frag[2][M_size][K0+2];  // [flip/flop][I][K]
    __shared__ bfloat16_t B_frag[2][N_size][K0+2];  // [flip/flop][J][K]
    
    // initialize c fragment to 0
    float32x8_t   c_frag[M1][N1] = {};  // [i:i+16]
    bfloat16x16_t a_frag[M1]; // [k..k+K0]
    bfloat16x16_t b_frag[N1]; // [k..k+K0]

    // chargement des elements en "local"
    load<BLOC_A,M_size,K0>(a, A_frag[0], IA, I0+IA, 0, lane, K);
    load<BLOC_B,N_size,K0>(b, B_frag[0], JB, J0+JB, 0, lane, K);
    __syncthreads();

    int l1=K0;
    int flip=0;
    int flop=1;
    for (; l1<K; l1+=K0) {
        #pragma unrool
        for (int _i1=0; _i1<M1; ++_i1)
            #pragma unrool
            for (int _k=0; _k<K0; _k++) a_frag[_i1][_k]=A_frag[flip][i2+_i1*M0+lane][_k];
        #pragma unrool
        for (int _j1=0; _j1<N1; ++_j1)
            #pragma unrool
            for (int _k=0; _k<K0; _k++) b_frag[_j1][_k]=B_frag[flip][j2+_j1*N0+lane][_k];
            
        // les 2 wave remplisent 1/2 de c!
        #pragma unrool
        for (int _i1=0; _i1<M1; ++_i1) {
            #pragma unrool
            for (int _j1=0; _j1<N1; ++_j1) {
                c_frag[_i1][_j1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag[_i1], b_frag[_j1], c_frag[_i1][_j1]);
            }
        }
        
        flop = (flip+1)%2;
        load<BLOC_A,M_size,K0>(a, A_frag[flop], IA, I0+wave*BLOC_A, l1, lane, K);
        load<BLOC_B,N_size,K0>(b, B_frag[flop], JB, J0+JB, l1, lane, K);
        flip = flop;
        __syncthreads();
    }
    #pragma unrool
    for (int _i1=0; _i1<M1; ++_i1)
        for (int _k=0; _k<K0; _k++) a_frag[_i1][_k]=A_frag[flip][i2+_i1*M0+lane][_k];
    #pragma unrool
    for (int _j1=0; _j1<N1; ++_j1)
        for (int _k=0; _k<K0; _k++) b_frag[_j1][_k]=B_frag[flip][j2+_j1*N0+lane][_k];

    #pragma unrool
    for (int _i1=0; _i1<M1; ++_i1) {
        #pragma unrool
        for (int _j1=0; _j1<N1; ++_j1) {
            c_frag[_i1][_j1] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag[_i1], b_frag[_j1], c_frag[_i1][_j1]);
        }
    }

    // ecriture de C
    #pragma unrool
    for (int _i1=0; _i1<M1; ++_i1)
    #pragma unrool
    for (int _j1=0; _j1<N1; ++_j1) {
        const int pos = I*(J0+j2+_j1*N0+lane) + I0+i2+_i1*M0;
        #pragma unrool
        for (int ele = 0; ele < M0/2; ++ele) { // == i
            c[ pos + ele*2 + wave] = c_frag[_i1][_j1][ele];
        }
    }
}

// simplify kernel start:
template<int M2, int N2, int M1, int N1>
__host__ inline void sgemm_wmma(bfloat16_t* a, bfloat16_t* b, float32_t* c, size_t I, size_t J, size_t K) {
    constexpr int M0=16;
    constexpr int N0=16;
    constexpr int K0=16;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(wmma_matmul<M2,N2,M1,N1>),
                       dim3(I/(M0*M1*M2),J/(N0*N1*N2),1), dim3(16, 2, M2*N2),
                       sizeof(bfloat16_t)*2*K0*(M2*M1*M0+N2*N1*N0), 0,
                       a,b,c, I,J,K);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
}

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

//#define ITERATIONS 100000
//size_t m = 256;
//size_t n = 256;
//size_t k = 256;

#define ITERATIONS 10000
size_t m = 512;
size_t n = 512;
size_t k = 512;

//#define ITERATIONS 1000
//size_t m = 1024;
//size_t n = 1024;
//size_t k = 1024;

// kernel config ...  nead to be adjust for best perf depend on m/n/k size.
constexpr size_t M2 = 2;
constexpr size_t N2 = 4;
constexpr size_t M1 = 4;
constexpr size_t N1 = 2;

bfloat16_t *A[ITERATIONS], *B;
float32_t  *C;

template<int I>
void multiply() {
    for (int i=0; i<I; i++) {
        sgemm_wmma<M2,N2,M1,N1>(A[i], B, C, m, n, k);
    }
    volatile float32_t x = C[0];
    (void)x;
}

int main() {
    bfloat16_t* a = allocateHost<bfloat16_t>(m*k*ITERATIONS);
    bfloat16_t* b = allocateHost<bfloat16_t>(k*n);
    float32_t*  c = allocateHost<float32_t>(m*n);
    
    for(int i=0; i<ITERATIONS; i++) A[i]=getDeviceMem(&a[i*m*k]);
    B=getDeviceMem(b);
    C=getDeviceMem(c);

    for (int l=0; l<ITERATIONS; ++l) 
    for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j)
        a[l*m*k + i*k + j] = (bfloat16_t)1.f;
    for (int i = 0; i < n; ++i)
    for (int j = 0; j < k; ++j)
        b[i*k + j] = (bfloat16_t)1.f;
    for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
        c[i*m + j] = -0.5f;

    multiply<1>();
    //control(m,n,C,(float)k);
    printf("gemm<%zu:%zu,%zu:%zu>(%zu,%zu,%zu)\n", M2,N2, M1,N1, m,n,k);
    for (int nb=0; nb<5; nb++) {
        BENCH(multiply<ITERATIONS>());
    }
}

