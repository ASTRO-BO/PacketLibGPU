#include <stdio.h>
#include <cuda.h>

typedef unsigned short word;

__global__ void kernel(int a, int b)
{
  //statements
}

__global__ void add(int a, int b, int *c)
{
        *c = a + b;
}

__global__ void gpuproc(const word* A, word* B, int numElements)
{
	// So simple because I use just one kernel, but using more ones is also simple
	for (size_t i=0; i<numElements; i++)
	{
		B[i] = A[i] - A[i] + 1;
	}
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

void cuda_proc(word* h_data, int numElements)
{

	// Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    
	//printf("Processing data ... ");

	// data size
	size_t size = numElements * sizeof(word); // not appropriate for compressed data
	
	/*
	// Allocate the device vector d_data
    float *d_data = NULL;
    err = cudaMalloc((void **)&d_data, size); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    */
    
	// Allocate the device input vector A
    word *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Allocate the device output vector B
    word *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy the host input vector A in host memory to the device input vectors in
    // device memory
    err = cudaMemcpy(d_A, h_data, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Launch the CUDA Kernel
    gpuproc<<<1, 1>>>(d_A, d_B, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch the kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy the device result vector in device memory to the host result vector
    // in host memory.

    err = cudaMemcpy(h_data, d_B, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Verify that the result vector is correct
    /* Verification moved in the main() file
    for (int i = 0; i < numElements; ++i)
    {
        if (h_data[i] != 1)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    */
    
    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    
    /*
    // Copy the host vector h_data in host memory to the device vector d_data in
    // device memory 
    err = cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Launch the CUDA Kernel
    gpuproc<<<1, 1>>>(h_data, d_data, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch the kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	*/
	
	//printf("Test PASSED.\n");
}
