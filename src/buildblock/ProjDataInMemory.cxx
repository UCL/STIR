//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Implementations for non-inline functions of class ProjDataInMemory

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Claire Labbe
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInMemory.h"
#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"
#include "stir/SegmentByView.h"
#include "stir/utilities.h"
#include "stir/interfile.h"
#include <fstream>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

#ifndef STIR_NO_NAMESPACES
using std::fstream;
using std::iostream;
using std::ios_base;
#endif

START_NAMESPACE_STIR

ProjDataInMemory::
~ProjDataInMemory()
{}

ProjDataInMemory::
ProjDataInMemory(shared_ptr<ProjDataInfo> const& proj_data_info_ptr, const bool initialise_with_0)
  :
  ProjDataFromStream(proj_data_info_ptr, 0) // trick: first initialise sino_stream_ptr to 0
{
  
#ifdef BOOST_NO_STRINGSTREAM
  const size_t buffer_size = get_size_of_buffer();
  buffer = auto_ptr<char>(new char[buffer_size]);
  sino_stream = new strstream(buffer, buffer_size, ios_base::in | ios_base::out | ios_base::binary);
#else
  // it would be advantageous to preallocate memory as well
  // the only way to do this is by passing a string of the appropriate size
  // However, if basic_string doesn't do reference counting, we would have
  // temporarily 2 strings of a (potentially large) size in memory.
  // todo?
  sino_stream = new std::stringstream(ios_base::in | ios_base::out | ios_base::binary);
#endif

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
: ProjDataFromStream(proj_data.get_proj_data_info_ptr()->clone(), 0)
{
#ifdef BOOST_NO_STRINGSTREAM
  const size_t buffer_size = get_size_of_buffer();
  buffer = auto_ptr<char>(new char[buffer_size]);
  sino_stream = new strstream(buffer, buffer_size, ios_base::in | ios_base::out | ios_base::binary);
#else
  // it would be advantageous to preallocate memory as well
  // the only way to do this is by passing a string of the appropriate size
  // However, if basic_string doesn't do reference counting, we would have
  // temporarily 2 strings of a (potentially large) size in memory.
  // todo?
  sino_stream = new std::stringstream(ios_base::in | ios_base::out | ios_base::binary);
#endif

  // copy data
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
  shared_ptr<iostream> file_stream = 
    new fstream (output_filename.c_str(), ios::out|ios::binary);
  if (!file_stream->good())
  {
    warning("ProjDataInMemory::write_to_file: error opening output file %s\n",
	    output_filename.c_str());
    return Succeeded::no;
  }

  //ProjDataInterfile out_projdata(output_filename, proj_data_info_ptr, ios::out); 
  ProjDataFromStream out_projdata(proj_data_info_ptr, file_stream); 

  Succeeded success =
   write_basic_interfile_PDFS_header(output_filename, out_projdata);

  if (success== Succeeded::no)
    return success;

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


END_NAMESPACE_STIR

  
  
  




