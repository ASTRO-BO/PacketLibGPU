/***************************************************************************
	main.cpp
	-------------------
 copyright            : (C) 2014 Andrea Bulgarelli, Giovanni De Cesare, Andrea Zoli, Valentina Fioretti
 email                : bulgarelli@iasfbo.inaf.it
                        decesare@iasfbo.inaf.it		
			zoli@iasfbo.inaf.it
			fioretti@iasfbo.inaf.it
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/




#include "packetlibop.h"

void sig_ext(word *data, int numElements, double *maximum, double *time);

void swap(byte* stream, dword dim) {
	
	for(dword i = 0; i< dim; i+=2)
	{
		/// For odd dimensions
		if((dim - i) != 1)
		{
			byte btemp = stream[i];
			stream[i] = stream[i+1];
			stream[i+1] = btemp;
		}
	}
}

dword getdword(dword value, bool streamisbigendian) {
	if(streamisbigendian)
		return value;
	else {
		
		dword tmp = value << 16;
		//cout << tmp << endl;
		tmp += value >> 16;
		//cout << tmp << endl;
		return tmp;
	}
}

struct CTAPacketHeaders {
	word idAndAPID;
	word ssc;
	dword packetLength;
	word crcTypeAndSubtype;
	word compression;
};

struct CTADataHeaders {
	dword times;
	dword timens;
	word arrayID;
	word runNumber;
	dword eventNumber;
	word telescopID;
	word arrayTriggerData;
	word npixels;
	word nsamples;
	word npixelsID;
};



extern "C++" void cuda_function(int a, int b);
extern "C++" void cuda_proc(word* data, double* maximum, double* time, int numElements);
int main(int argc, char *argv[]) {
	
	/*
	*   CUDA simple test 
	*/
	// cout << ">> GPU test: simple add <<" << endl; 
	// cuda_function(23, 34);
	// cout << "** end test **" << endl;
	
	unsigned long totbytes = 0;
	unsigned long nops = 0;
	
	struct timespec start;

	string ctarta;
	
	if(argc == 2) {
		const char* home = getenv("CTARTA");
		if (!home)
		{
			std::cerr << "ERROR: CTARTA environment variable is not defined." << std::endl;
			return 0;
		}
			ctarta = home;		
	} else {
		cerr << "ERROR: Please provide the raw file as input" << endl;
		cout << "Usage: packetlibgpu file.raw" << endl;
		return 0;
	}
	
	string configFileName = ctarta + "/share/packetlibdemo/ctacamera_all.xml";
	
	char* filename = argv[1];
	
	InputPacketStream* ips = 0;
	
	try
	{
		cout << "Create input packet stream" << endl;
		ips = new InputPacketStream();
		ips->setFileNameConfig(configFileName.c_str());
		ips->createStreamStructure();			
		// Create and open an input device: file
		Input* in;
		in = (Input*) new InputFile(ips->isBigEndian());
		char** param = (char**) new char*[2];
		param[0] = (char*) filename; // file name
		param[1] = 0;
		in->open(param); /// open input
		// connect the input packet stream with the input device
		ips->setInput(in);					
	}
	catch (PacketException* e)
	{
		cout << "Error on stream constructor: ";
		cout << e->geterror() << endl;
	}

	clock_gettime( CLOCK_MONOTONIC, &start);

	try {
		cout << "Decoding and get the array of camera data" << endl;

		Packet* p = ips->getPacketType("CHEC-CAM");
		
		int indexEventNumber = p->getPacketSourceDataField()->getFieldIndex("eventNumber");
		int indexNPixels = p->getPacketSourceDataField()->getFieldIndex("Number of pixels");
		int indexNSamples = p->getPacketSourceDataField()->getFieldIndex("Number of samples");
		
		while((p = ips->readPacket())) {
			nops++;
			if(p->getPacketID() == PACKETNOTRECOGNIZED)
			{
				cout << "Packet not recognized" << endl;
			} else 
			{
				dword packetSize = p->size(); // Total packet size
				word* cameraData = (word*)p->getData()->stream; // Data 
				dword dataSize   = p->getData()->size(); 
				
				totbytes += packetSize;
				if (nops < 1) // Control the printing ...
				{
					cout << "Event number: " << p->getPacketSourceDataField()->getFieldValue(indexEventNumber) << endl;
					cout << "Number of pixels: " << p->getPacketSourceDataField()->getFieldValue(indexNPixels) << endl;
					cout << "Number of Samples: " << p->getPacketSourceDataField()->getFieldValue(indexNSamples) << endl;
					cout << "Number of bytes (total): " << packetSize << endl;
					cout << "Number of bytes (data): "  << dataSize   << endl;				
				}
				
				int numElements = dataSize/sizeof(word); 
				
				// CPU processing
				double maximum=0.;
				double time=0.;
				sig_ext(cameraData, numElements, &maximum, &time);
				cout << "The CPU says: Maximum signal " << maximum << " ";
				cout << "at time  " << time << endl;
				
				// GPU processing
				word* testData = (word *)malloc(dataSize); // used to test the GPU results
				for (int i=0; i < numElements; ++i)
				{
					testData[i] = cameraData[i];
				} 
				double maximum_gpu=0.;
				double time_gpu=0.;				
				cuda_proc(cameraData, &maximum_gpu, &time_gpu, numElements);
				cout << "The GPU says: Maximum signal " << maximum_gpu << " ";
				cout << "at time  " << time_gpu << endl;	
				cout << "GPU verifying data ... ";
				/*
				for (word pixel=0; pixel < 2048; pixel++)
				{
					for (word sample = 0; sample < 40; sample ++)
					{
						cout << cameraData[pixel*40 + sample] << " ";
					}
					cout << endl;
				}
				cout << endl;
				*/
				// Verify that the result vector is correct
				
    			for (int i = 0; i < numElements; ++i)
    			{
        			if (cameraData[i] - testData[i] != 1)
        			{
            			fprintf(stderr, "Result verification failed at element %d!\n", i);
            			exit(EXIT_FAILURE);
        			}
    			}								
				cout << "Test PASSED." <<endl;
				free(testData); 		
			}
		}
		cout << endl << endl;
		endHertz(true, start, totbytes, nops);
	}
	catch (PacketException* e)
	{
		cout << "Error in decoding for routing: ";
		cout << e->geterror() << endl;
	}
	
	return 0;	
}
