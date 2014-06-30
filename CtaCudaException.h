/*
 *  Created on: March 2014
 *  Author: Juan Jose Rodriguez-Vazquez
 *  CIEMAT, Madrid (Spain)
 */

#ifndef _CTA_CUDA_EXCEPTION_H_
#define _CTA_CUDA_EXCEPTION_H_

#include <exception>
#include <string>

class ctaCudaException : public std::exception {

public:
								ctaCudaException( const std::string errorString ) throw();
								~ctaCudaException( void ) throw();
	const std::string		What( void ) const throw();

};

#endif /* _CTA_CUDA_EXCEPTION_H_ */
