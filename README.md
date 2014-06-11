GPU 4 CTA RTA
=============

GPU computing for the Real Time Analysis

This demo does the followings steps:    

* Read data arrays from a stream using the PacketLib    

	* for each data array:

		* allocate the GPU device memory

		* copy the array in the GPU memory

		* Process tha data array in one GPU kernel (simply: r[i] <- a[i] - a[i] + 1)

		* return the value

		* test the result
    
    The GPU processing can be parallelized using the kernel index in the device function.
    
    All Tests PASSED.
