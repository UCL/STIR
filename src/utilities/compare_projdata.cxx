//
// $Id$: $Date$
//

/*!
\file

\brief process sinogram data

\author Matthew Jacobson
\author PARAPET project

\date    $Date$
\version $Revision$

This utility compares two input (interfile) projection data sets. 
The input data are deemed identical if their maximum absolute difference 
is less than a hard-coded tolerance value.

*/





#include "ProjDataFromStream.h"
#include "SegmentByView.h"
#include "SegmentBySinogram.h"
#include "Sinogram.h"
#include "Viewgram.h"

//#include "Scanner.h"
#include "ArrayFunction.h" 
#include "recon_array_functions.h"
#include "display.h"
#include "interfile.h"
#include "utilities.h"
#include "shared_ptr.h"

#include <numeric>
#include <fstream> 
#include <iostream> 

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::fstream;
#endif



START_NAMESPACE_TOMO


//*********************** prototypes

void compare(SegmentByView<float>& sino1,SegmentByView<float> &sino2,
float &max_error, float &amplitude);

shared_ptr<ProjData> ask_proj_data(char *input_query);
//*********************** functions

void update_comparison(SegmentByView<float>& input1,SegmentByView<float> &input2,float &max_error, float &amplitude)
{
	float reference_max=input1.find_max();
	float reference_min=input1.find_min();

	float local_amplitude=fabs(reference_max)>fabs(reference_min)?
										fabs(reference_max):fabs(reference_min);

	amplitude=local_amplitude>amplitude?local_amplitude:amplitude;

	input1-=input2;
    in_place_abs(input1);
	float max_local_error=input1.find_max();

    max_error=max_local_error>max_error? max_local_error:max_error;
  
}




END_NAMESPACE_TOMO

//********************** main



USING_NAMESPACE_TOMO


int main(int argc, char *argv[])
{

	const float tolerance=0.0001;

	float max_error=0.F, amplitude=0.F;

    if(argc<3)
	error("Usage:compare_data old.hs new.hs [max_segment]\n");

	

	shared_ptr<ProjData> first_operand=ProjData::read_from_file(argv[1]);
	shared_ptr<ProjData> second_operand=ProjData::read_from_file(argv[2]);

	int max_segment=first_operand->get_max_segment_num();
	if(argc==4 && atoi(argv[3])>=0 && atoi(argv[3])<max_segment) 
		max_segment=atoi(argv[3]);

    for (int segment_num = -max_segment; segment_num <= max_segment ; segment_num++) 
	{

     SegmentByView<float> input1=first_operand->get_segment_by_view(segment_num);
     SegmentByView<float> input2=second_operand->get_segment_by_view(segment_num);

	 update_comparison(input1,input2,max_error,amplitude);

	}

	bool same=(max_error/amplitude<=tolerance)?true:false;


	cerr<<endl<<"Maximum absolute error = "<<max_error<<endl;
	cerr<<"Error relative to sup-norm of first array = "<<(max_error/amplitude)*100<<" %"<<endl;

	cerr<<"Projection arrays deemed ";

	if(same)
	  cerr<<"identical";
	else cerr<<"different";
	cerr<<endl;

	return same?EXIT_SUCCESS:EXIT_FAILURE;

} //end main
