/*!

  \file
  \ingroup projdata
  \brief Implementations for non-inline functions of class stir::ProjDataInMemory

  \author Kris Thielemans
*/
/*
    Copyright (C) 2002 - 2011-02-23, Hammersmith Imanet Ltd
    Copyright (C) 2011, Kris Thielemans
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


#include "stir/ProjDataInMemory.h"
#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"
#include "stir/SegmentByView.h"
#include "stir/ProjDataInterfile.h"
#include "stir/Bin.h"
#include <fstream>


#ifdef STIR_USE_OLD_STRSTREAM
#include <strstream>
#else
#include <sstream>
#endif

#ifndef STIR_NO_NAMESPACES
using std::fstream;
using std::iostream;
using std::ios;
#ifdef STIR_USE_OLD_STRSTREAM
using std::strstream;
#endif
using std::string;
#endif

START_NAMESPACE_STIR

ProjDataInMemory::
~ProjDataInMemory()
{}

ProjDataInMemory::
ProjDataInMemory(shared_ptr<ExamInfo> const& exam_info_sptr,
		 shared_ptr<ProjDataInfo> const& proj_data_info_ptr, const bool initialise_with_0)
  :
  ProjDataFromStream(exam_info_sptr, proj_data_info_ptr, shared_ptr<iostream>()) // trick: first initialise sino_stream_ptr to 0
{
  
#ifdef STIR_USE_OLD_STRSTREAM
  const size_t buffer_size = get_size_of_buffer();
  //buffer = auto_ptr<char>(new char[buffer_size]);
  buffer.reset(new char[buffer_size]);
  sino_stream.reset(new strstream(buffer.get(), buffer_size, ios::in | ios::out | ios::trunc | ios::binary));
#else
  // it would be advantageous to preallocate memory as well
  // the only way to do this is by passing a string of the appropriate size
  // However, if basic_string doesn't do reference counting, we would have
  // temporarily 2 strings of a (potentially large) size in memory.
  // todo?
  sino_stream.reset(new std::stringstream(ios::in | ios::out | ios::binary));
#endif

  if (!*sino_stream)
    error("ProjDataInMemory error initialising stream\n");

  if (initialise_with_0)
  {
    for (int segment_num = proj_data_info_ptr->get_min_segment_num();
         segment_num <= proj_data_info_ptr->get_max_segment_num();
         ++segment_num)
      set_segment(proj_data_info_ptr->get_empty_segment_by_view(segment_num));
  }
}

ProjDataInMemory::
ProjDataInMemory(const ProjData& proj_data)
  : ProjDataFromStream(proj_data.get_exam_info_sptr(),
		       proj_data.get_proj_data_info_ptr()->create_shared_clone(), shared_ptr<iostream>())
{
#ifdef STIR_USE_OLD_STRSTREAM
  const size_t buffer_size = get_size_of_buffer();
  //buffer = auto_ptr<char>(new char[buffer_size]);
  buffer.reset(new char[buffer_size]);
  sino_stream.reset(new strstream(buffer.get(), buffer_size, ios::in | ios::out | ios::binary));
#else
  // it would be advantageous to preallocate memory as well
  // the only way to do this is by passing a string of the appropriate size
  // However, if basic_string doesn't do reference counting, we would have
  // temporarily 2 strings of a (potentially large) size in memory.
  // todo?
  sino_stream.reset(new std::stringstream(ios::in | ios::out | ios::binary));
#endif

  if (!*sino_stream)
    error("ProjDataInMemory error initialising stream");

  // copy data
  // (note: cannot use fill(projdata) as that uses virtual functions, which won't work in a constructor
  for (int segment_num = proj_data_info_ptr->get_min_segment_num();
       segment_num <= proj_data_info_ptr->get_max_segment_num();
       ++segment_num)
    set_segment(proj_data.get_segment_by_view(segment_num));
}

size_t
ProjDataInMemory::
get_size_of_buffer() const
{
  size_t num_sinograms = 0;
  for (int segment_num = proj_data_info_ptr->get_min_segment_num();
       segment_num <= proj_data_info_ptr->get_max_segment_num();
       ++segment_num)
    num_sinograms += proj_data_info_ptr->get_num_axial_poss(segment_num);
  return 
    num_sinograms * 
    proj_data_info_ptr->get_num_views() *
    proj_data_info_ptr->get_num_tangential_poss() *
    sizeof(float);
}

Succeeded
ProjDataInMemory::
write_to_file(const string& output_filename) const
{

  ProjDataInterfile out_projdata(this->get_exam_info_sptr(),
				 this->proj_data_info_ptr, output_filename, ios::out); 
  
  Succeeded success=Succeeded::yes;
  for (int segment_num = proj_data_info_ptr->get_min_segment_num();
       segment_num <= proj_data_info_ptr->get_max_segment_num();
       ++segment_num)
  {
    Succeeded success_this_segment =
      out_projdata.set_segment(get_segment_by_view(segment_num));
    if (success==Succeeded::yes)
      success = success_this_segment;
  }
  return success;
    
}

float 
ProjDataInMemory::get_bin_value(Bin& bin)
{
   Viewgram<float> viewgram = get_viewgram(bin.view_num(),bin.segment_num()); 
    
   return viewgram[bin.axial_pos_num()][bin.tangential_pos_num()]; 

}
END_NAMESPACE_STIR

