all: packetlib_gpu_demo

packetlib_gpu_demo: cuda.o mac_clock_gettime.o main.o packetlibop.o
	cc  -m64 -fexceptions -Wall -g  -o packetlib_gpu_demo -L/usr/local/cuda/lib -lcuda -lcudart  -lstdc++ -lpacket *.o

cuda.o: code/cuda.cu
	nvcc -c -arch=sm_20 code/cuda.cu 

mac_clock_gettime.o: code/mac_clock_gettime.cpp
	cc  -m64 -fexceptions -Wall -g -I code  -c code/mac_clock_gettime.cpp

main.o: code/main.cpp
	cc  -m64 -fexceptions -Wall -g -I code  -c code/main.cpp

packetlibop.o: code/packetlibop.cpp
	cc  -m64 -fexceptions -Wall -g -I code  -c code/packetlibop.cpp

clean: 
	rm -f *o packetlib_gpu_demo
