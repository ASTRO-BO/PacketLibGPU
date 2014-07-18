#include "packetlibop.h"

void sig_ext(word *data, int numElements, double *maximum, double *time)
{
	const int maxSample=40, windowSize=9;
	
	*maximum =0.;
	*time=0.;
	int position = 0; 
	double sum = 0.;
	
	// The first 40 elements of data array are the first sample
	
	// Test
	cout << endl << "First sample: ";
	for (word pixel=0; pixel < 1; pixel++)
	{
		for (word sample = 0; sample < maxSample; sample ++)
		{
			cout << data[pixel*40 + sample] << " ";
		}
		cout << endl;
	}		
	
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