GPU 4 CTA RTA
=============

GPU computing for the Real Time Analysis

This demo does the followings steps:    

* Read data arrays from a stream using the PacketLib    

	* for each data array:

		* Allocate the GPU device memory

		* Copy the data array in the GPU memory

		* Extract the signal with the sliding window method, implemented in both CPU and GPU

		* Test the results
    
    The GPU processing can be parallelized using the kernel index in the device function.
    
    All Tests PASSED.
