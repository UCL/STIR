//
// $Id$: $Date$
//

/*!
\file
\ingroup utilities
\brief compare 2 files with sinogram data

\author Matthew Jacobson
\author PARAPET project

\date    $Date$
\version $Revision$

This utility compares two input projection data sets. 
The input data are deemed identical if their maximum absolute difference 
is less than a hard-coded tolerance value.
Diagnostic output is written to stdout, and the return value indicates
if the files are identical or not.

*/




#include "ProjData.h"
#include "SegmentByView.h"
#include "ArrayFunction.h" 
#include "shared_ptr.h"

#include <numeric>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::cout;
using std::endl;
#endif



START_NAMESPACE_TOMO


//*********************** prototypes

void compare(SegmentByView<float>& sino1,SegmentByView<float> &sino2,
	     float &max_error, float &amplitude);


//*********************** functions

void 
update_comparison(SegmentByView<float>& input1,SegmentByView<float> &input2,
		  float &max_error, float &amplitude)
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

  const float tolerance=0.0001F;

  float max_error=0.F, amplitude=0.F;

  if(argc<3)
  {
    cerr<< "Usage:" << argv[0] << " old_projdata new_projdata [max_segment_num]\n";
    exit(EXIT_FAILURE);
  }

	

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


  cout<<endl<<"Maximum absolute error = "<<max_error<<endl;
  cout<<"Error relative to sup-norm of first array = "<<(max_error/amplitude)*100<<" %"<<endl;

  cout<<"Projection arrays deemed ";

  if(same)
    cout<<"identical";
  else cout<<"different";
  cout<<endl;

  return same?EXIT_SUCCESS:EXIT_FAILURE;

} //end main
