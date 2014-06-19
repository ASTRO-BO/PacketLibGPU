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


__device__ void sig_ext_gpu(const word *data, int numElements, double *maximum, double *time)
{
	const int maxSample=40, windowSize=9;
	
	*maximum =0.;
	*time=0.;
	int position = 0; 
	double sum = 0.;
		
	// Maximum sum (sliding window search)
	for (int sample = 0; sample < windowSize; ++sample)
	{
		sum += data[sample];
	}
	*maximum = sum;
	for (int sample = 1; sample <= maxSample - windowSize; ++sample)
	{
		sum += data[windowSize + sample - 1] - data[sample - 1];
		if (sum > *maximum)
		{
			*maximum = sum;
			position = sample;
		}
	}
	// Time
	sum = 0.;
	for (int sample=0; sample < windowSize; ++sample)
	{
		sum += data[position + sample] * (position + sample);
	}
	*time = sum / *maximum;
}



__global__ void gpuproc(const word* A, word* B, double *maximum, double *time, int numElements)
{
	// So simple because I'm using just one kernel in this test.
	for (size_t i=0; i<numElements; i++)
	{
		B[i] = A[i] + 1;
	}
	// GPU processing
	sig_ext_gpu(A, numElements, maximum, time);
}



void cuda_proc(word* h_data, double* maximum, double* time, int numElements)
{

	// Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    
	//printf("Processing data ... ");

	// data size
	size_t size = numElements * sizeof(word); // not appropriate for compressed data
	    
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
    
    // Allocate the device output
    double* d_maximum;
	err = cudaMalloc( (void**)&d_maximum, sizeof(double) );
	
	if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Allocate the device output
    double* d_time;
	err = cudaMalloc( (void**)&d_time, sizeof(double) );
    
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
    gpuproc<<<1, 1>>>(d_A, d_B, d_maximum, d_time, numElements);
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
    
    // Copy the device result in device memory to the host result
    // in host memory.
    
    err = cudaMemcpy(maximum, d_maximum, sizeof(double), cudaMemcpyDeviceToHost );
    
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy value from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(time, d_time, sizeof(double), cudaMemcpyDeviceToHost );
    
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy value from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
        
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
    
    err = cudaFree(d_maximum);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaFree(d_time);

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
    // Comment: TIME EXPENSIVE
    
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
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

    // Il numerello sta ancora sulla GPU. Me lo devo copiare sulla memoria del processore
	// prima di poterlo usare
	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost );

	printf("%d + %d = %d\n", a, b, c);

	// E' sempre una buona abitudine liberare la memoria dopo averla usata
    cudaFree( dev_c );    
}