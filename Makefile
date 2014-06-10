# Compilers
NVCC = nvcc
CLANG = clang++

all: packetlibgpu

packetlibgpu: cuda.o mac_clock_gettime.o main.o packetlibop.o
	$(CLANG)  -m64 -fexceptions -Wall -g  -o packetlibgpu -L/usr/local/cuda/lib -lcudart -lpacket *.o

cuda.o: code/cuda.cu
	$(NVCC) -c -arch=sm_20 code/cuda.cu 

mac_clock_gettime.o: code/mac_clock_gettime.cpp
	$(CLANG)  -m64 -fexceptions -Wall -g -I code  -c code/mac_clock_gettime.cpp

main.o: code/main.cpp
	$(CLANG)  -m64 -fexceptions -Wall -g -I code  -c code/main.cpp

packetlibop.o: code/packetlibop.cpp
	$(CLANG)  -m64 -fexceptions -Wall -g -I code  -c code/packetlibop.cpp

clean: 
	rm -f *o packetlibgpu
