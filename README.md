I get resently a laptop with a nice Ryzen 7940HS CPU and 64 Go of RAM.

So now I can test AVX512 / bf16 / RDNA3 ...

After read this nice post of https://justine.lol/matmul/ I start test new gemm kernel. 

Be carefull most of the gemm is "wrong":
  - only some size supported
  - special storage neaded
  - ...


## gemm_AVX512_fp32_00.cpp
compute C=A*B with kernel on block C[M,N] ([16,16]...) with float_32
```c++
for (k=0; k<K; k++)
    C[1:M,1:N] += A[1:M,k]*B[k,1:N]
```

## gemm_AVX512_bf16_00.cpp
compute C=A*B with kernel on block C[M,N] ([5,5]...) with bfloat_16
```c++
for (k=0; k<K; k+=32)
    C[1:M,1:N] += A[1:M,k:k+32]*B[k:k+32,1:N]
```

A/B/C can be bfloat_16 or float_32.


