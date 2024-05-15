I get resently a laptop with a nice Ryzen 7940HS CPU and 64 Go of RAM.

So now I can test AVX512 / bf16 / RDNA3 ...

After read this nice post of https://justine.lol/matmul/ I start test new gemm kernel. 

Be carefull most of the gemm is "wrong":
  - only some size supported
  - special storage neaded
  - ...

## gemm_AVX512_fp32_llamafile.cpp
extract from tinyblas https://justine.lol/matmul/ for "reference"
- keep only AVX512 release + OpenMP

compute C=A*B with kernel on block C[M,N] ([5,5]...) with float_32
```c++
for (k=0; k<K; k+=16)
    C[1:M,1:N] += A[1:M,k:k+16]*B[k:k+16,1:N]
```

Ryzen 7940HS => 
- M=N=K= 128  => ~ 320 GFlops
- M=N=K= 256  => ~ 715 GFlops
- M=N=K= 512  => ~ 925 GFlops
- M=N=K=1024  => ~ 930 GFlops
- M=N=K=2048  => ~ 540 GFlops

## gemm_AVX512_fp32_00.cpp
compute C=A*B with kernel on block C[M,N] ([16,16]...) with float_32
```c++
for (k=0; k<K; k++)
    C[1:M,1:N] += A[1:M,k]*B[k,1:N]
```

Ryzen 7940HS => 
- M=N=K= 128  => ~ 780 GFlops
- M=N=K= 256  => ~1000 GFlops
- M=N=K= 512  => ~1090 GFlops
- M=N=K=1024  => ~1100 GFlops
- M=N=K=2048  => ~ 950 GFlops

## gemm_AVX512_bf16_00.cpp
compute C=A*B with kernel on block C[M,N] ([5,5]...) with bfloat_16
```c++
for (k=0; k<K; k+=32)
    C[1:M,1:N] += A[1:M,k:k+32]*B[k:k+32,1:N]
```

A/B/C can be bfloat_16 or float_32.

- Ryzen 7940HS => ~1200 GFlops with fp32
- Ryzen 7940HS => ~1600 GFlops with bf16

## gemm_RDNA3_bf16_00.cpp
use RDNA3 eGPU on AMD APU (not dGPU!!!)
and wmma instuction that release micro kernel of size:
```
C[16,16] += A[16,16]*B[16,16]
```
Ryzen 7940HS => 
- M=N=K= 256  => ~1600 GFlops
- M=N=K= 512  => ~3500 GFlops
- M=N=K=1024  => ~6600 GFlops

