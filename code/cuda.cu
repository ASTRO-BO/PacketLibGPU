#include <stdio.h>
#include <cuda.h>

__global__ void kernel(int a, int b)
{
  //statements
}

__global__ void add(int a, int b, int *c)
{
        *c = a + b;
}

void cuda_function(int a, int b)
{
	int c;
	int *dev_c;

    // Devo allocare la memoria
	cudaMalloc( (void**)&dev_c, sizeof(int) );

    // lancio il kernel
    add<<<1,1>>>(a, b, dev_c);

    // Il numerello sta ancora sulla GPU. Me lo devo copiare sulla mamoria del processore
	// prima di poterlo usare
	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost );

	printf("%d + %d = %d\n", a, b, c);

	// E' sempre una buona abitudine liberare la memoria dopo averla usata
    cudaFree( dev_c );    
}
