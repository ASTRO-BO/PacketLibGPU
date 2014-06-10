# Instal dir
CTAGPU = $(HOME)/Works/CTA/PacketLibGPU-install

# Compilers
NVCC = nvcc
CLANG = clang++

all: packetlibgpu

install: all
	cp packetlibgpu $(CTAGPU)/bin

packetlibgpu: cuda.o mac_clock_gettime.o main.o packetlibop.o
	$(CLANG)  -m64 -fexceptions -Wall  -o packetlibgpu -L/usr/local/cuda/lib -lcudart -lpacket *.o

cuda.o: code/cuda.cu
	$(NVCC) -c -arch=sm_50 code/cuda.cu 

mac_clock_gettime.o: code/mac_clock_gettime.cpp
	$(CLANG)  -m64 -fexceptions -Wall -I code  -c code/mac_clock_gettime.cpp

main.o: code/main.cpp
	$(CLANG)  -m64 -fexceptions -Wall -I code  -c code/main.cpp

packetlibop.o: code/packetlibop.cpp
	$(CLANG)  -m64 -fexceptions -Wall -I code  -c code/packetlibop.cpp

clean: 
	rm -f *o packetlibgpu
