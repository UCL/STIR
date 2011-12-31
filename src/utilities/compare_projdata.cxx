//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
\file
\ingroup utilities
\brief compare 2 files with sinogram data

\author Matthew Jacobson
\author Kris Thielemans
\author PARAPET project

$Date$
$Revision$

This utility compares two input projection data sets. 
The input data are deemed identical if their maximum absolute difference 
is less than a hard-coded tolerance value.
Diagnostic output is written to stdout, and the return value indicates
if the files are identical or not. Note however that a non-success return
value (which is 1 on most systems) might also indicate an error reading
the data.

*/

#include "stir/ProjData.h"
#include "stir/SegmentByView.h"
#include "stir/shared_ptr.h"

#include <iostream>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::cout;
using std::endl;
using std::max;
#endif



START_NAMESPACE_STIR


//*********************** functions

// WARNING: modifies input1
void 
update_comparison(SegmentByView<float>& input1, const SegmentByView<float> &input2,
		  float &max_pos_error, float& max_neg_error, float &amplitude)
{
  const float reference_max=input1.find_max();
  const float reference_min=input1.find_min();

  const float local_amplitude=
    fabs(reference_max)>fabs(reference_min)?
    fabs(reference_max):fabs(reference_min);

  amplitude=local_amplitude>amplitude?local_amplitude:amplitude;

  input1-=input2;
  const float max_local_pos_error=input1.find_max();
  const float max_local_neg_error=input1.find_min();

  max_pos_error=max_local_pos_error>max_pos_error? max_local_pos_error:max_pos_error;
  max_neg_error=max_local_neg_error<max_neg_error? max_local_neg_error:max_neg_error;
  
}




END_NAMESPACE_STIR

//********************** main





int main(int argc, char *argv[])
{

  USING_NAMESPACE_STIR;

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

  // compare proj_data_info
  {
    shared_ptr<ProjDataInfo> first_sptr(first_operand->get_proj_data_info_ptr()->clone());
    shared_ptr<ProjDataInfo> second_sptr(second_operand->get_proj_data_info_ptr()->clone());
    if (argc==4)
      {
	first_sptr->reduce_segment_range(-max_segment, max_segment);
	second_sptr->reduce_segment_range(-max_segment, max_segment);
      }
    if (*first_sptr != *second_sptr)
      {
	cout << "\nProjection data sizes or other data characteristics are not identical. \n"
	     << "Use list_projdata_info to investigate.\n";
	return EXIT_FAILURE;    }
  }

  const float tolerance=0.0001F;

  float max_pos_error=0.F, max_neg_error=0.F, amplitude=0.F;
  for (int segment_num = -max_segment; segment_num <= max_segment ; segment_num++) 
    {
      SegmentByView<float> input1=first_operand->get_segment_by_view(segment_num);
      const SegmentByView<float> input2=second_operand->get_segment_by_view(segment_num);

      update_comparison(input1,input2,max_pos_error,max_neg_error, amplitude);
    }

  const float max_abs_error=max(max_pos_error, -max_neg_error);
  bool same=(max_abs_error/amplitude<=tolerance)?true:false;

  cout << "\nMaximum absolute error = "<<max_abs_error
       << "\nMaximum in (1st - 2nd) = "<<max_pos_error
       << "\nMinimum in (1st - 2nd) = "<<max_neg_error<<endl;
  cout <<"Error relative to sup-norm of first data-set = "<<(max_abs_error/amplitude)*100<<" %"<<endl;

  cout<<"Projection arrays ";
  if(same)
  {
    cout << (max_abs_error == 0 ? "are " : "deemed ")
         << "identical\n";
  }
  else 
  {
    cout<<"deemed different\n";
  }

  return same?EXIT_SUCCESS:EXIT_FAILURE;

} //end main
