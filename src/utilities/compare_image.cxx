//
// $Id$: $Date$
//

/*!
\file

\brief compare images

\author Matthew Jacobson
\author PARAPET project

\date    $Date$
\version $Revision$

This utility compares two input (interfile) images. Arrays are deemed identical if
their maximum absolute difference is less than a hard-coded tolerance value.
*/

#include "VoxelsOnCartesianGrid.h"
#include "interfile.h"
#include "ArrayFunction.h"

#include <iostream> 
#include <fstream>
#include <numeric>

#ifndef TOMO_NO_NAMESPACES
using std::iostream;
using std::ofstream;
using std::ios;
using std::cerr;
using std::endl;
#endif



//********************** main



USING_NAMESPACE_TOMO


int main(int argc, char *argv[])
{

	const float tolerance=0.0001;
	float max_error;

    if(argc<3)
	error("Usage:compare_image old.hv new.hv \n");

	//TODO should accept all image types
	VoxelsOnCartesianGrid<float> first_operand,second_operand;

	first_operand= 
	* dynamic_cast<VoxelsOnCartesianGrid<float> *>(
	DiscretisedDensity<3,float>::read_from_file(argv[1]).get());

	second_operand= 
	* dynamic_cast<VoxelsOnCartesianGrid<float> *>(
	DiscretisedDensity<3,float>::read_from_file(argv[2]).get());

	float reference_max=first_operand.find_max();
	float reference_min=first_operand.find_min();

	float amplitude=fabs(reference_max)>fabs(reference_min)?
					 fabs(reference_max):fabs(reference_min);

	first_operand -= second_operand;
	in_place_abs(first_operand);
	max_error=first_operand.find_max();

	bool same=(max_error/amplitude<=tolerance)?true:false;

	cerr<<endl<<"Maximum absolute error = "<<max_error<<endl;
	cerr<<"Error relative to sup-norm of first array = "<<(max_error/amplitude)*100<<endl;

	cerr<<"Image arrays deemed ";

	if(same)
	cerr<<"identical";
	else cerr<<"different";
	cerr<<endl;

	return same?EXIT_SUCCESS:EXIT_FAILURE;

} //end main
