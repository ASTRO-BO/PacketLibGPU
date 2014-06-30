/*
 *  Created on: March 2014
 *  Author: Juan Jose Rodriguez-Vazquez
 *  CIEMAT, Madrid (Spain)
 */

#ifndef _CTA_CUDA_H_
#define _CTA_CUDA_H_

/*
 * LIBRARIES
 */
#include "CtaCudaException.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <tr1/memory>	//Provides shared_ptr for memory management
#include <string>
#include <vector>

/*
 * CLASSES
 */

class ctaCuda {

public:

	/*
	 * METHODS
	 */
																				ctaCuda( unsigned long long bufferSizeInBytes, unsigned int samplesPerWaveform );
																				~ctaCuda();
	std::vector< std::tr1::shared_ptr< unsigned short > >	getHostPointers( void );
	void									 									ProcessData( unsigned long long waveformsInBuffer );

};

#endif /* _CTA_CUDA_H_ */
