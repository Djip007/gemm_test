I get resently a laptop with a nice Ryzen 7940HS CPU and 64 Go of RAM.

So now I can test AVX512 / bf16 / RDNA3 ...

After read this nice post of https://justine.lol/matmul/ I start test new gemm kernel. 

Be carefull most of the gemm is "wrong":
  - only some size supported
  - special storage neaded
  - ...



