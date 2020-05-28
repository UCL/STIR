/*!

  \file
  \ingroup projdata
  \brief Implementations for non-inline functions of class stir::ProjDataInMemory

  \author Kris Thielemans
*/
/*
    Copyright (C) 2002 - 2011-02-23, Hammersmith Imanet Ltd
    Copyright (C) 2011, Kris Thielemans
    Copyright (C) 2019, UCL
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
#include "stir/Bin.h"
#include "stir/is_null_ptr.h"
#include <fstream>
#include <cstring>

#ifdef STIR_USE_OLD_STRSTREAM
#include <strstream>
#else
#include <boost/interprocess/streams/bufferstream.hpp>
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
ProjDataInMemory(shared_ptr<const ExamInfo> const& exam_info_sptr,
		 shared_ptr<const ProjDataInfo> const& proj_data_info_ptr, const bool initialise_with_0)
  :
  ProjDataFromStream(exam_info_sptr, proj_data_info_ptr, shared_ptr<iostream>()) // trick: first initialise sino_stream_ptr to 0
{
  this->_num_elements = size_all();
  this->buffer.reset(create_buffer(initialise_with_0));
  this->sino_stream = this->create_stream();
}

float *
ProjDataInMemory::
create_buffer(const bool initialise_with_0) const
{
  char *b = new char[this->get_size_of_buffer()];
  if (initialise_with_0)
      memset(b, 0, this->get_size_of_buffer());
  return reinterpret_cast<float*>(b);
}

shared_ptr<std::iostream>
ProjDataInMemory::
create_stream()
const
{
  shared_ptr<std::iostream> output_stream_sptr;
#ifdef STIR_USE_OLD_STRSTREAM
  output_stream_sptr.reset(new strstream(buffer.get(), this->get_size_of_buffer(), ios::in | ios::out | ios::binary));
#else
  // Use boost::bufferstream top reallocate.
  // For std::stringstream, the only way to do this is by passing a string of the appropriate size
  // However, if basic_string doesn't do reference counting, we would have
  // temporarily 2 strings of a (potentially large) size in memory.
  output_stream_sptr.reset
          (new boost::interprocess::bufferstream(reinterpret_cast<char*>(this->buffer.get()), this->get_size_of_buffer(), ios::in | ios::out | ios::binary));
#endif

  if (!*output_stream_sptr)
    error("ProjDataInMemory error initialising stream");

  return output_stream_sptr;
}

ProjDataInMemory::
ProjDataInMemory(const ProjData& proj_data)
  : ProjDataFromStream(proj_data.get_exam_info_sptr(),
		       proj_data.get_proj_data_info_sptr()->create_shared_clone(), shared_ptr<iostream>())
{
  this->buffer.reset(this->create_buffer());
  this->sino_stream = this->create_stream();

  // copy data
  this->fill(proj_data);
}

ProjDataInMemory::
ProjDataInMemory (const ProjDataInMemory& proj_data)
    : ProjDataFromStream(proj_data.get_exam_info_sptr(),
                 proj_data.get_proj_data_info_sptr()->create_shared_clone(), shared_ptr<iostream>())
{
  this->buffer.reset(this->create_buffer());
  this->sino_stream = this->create_stream();

  // copy data
  this->fill(proj_data);
}

size_t
ProjDataInMemory::
get_size_of_buffer() const
{
  return _num_elements * sizeof(float);
}

float 
ProjDataInMemory::get_bin_value(Bin& bin)
{
   Viewgram<float> viewgram = get_viewgram(bin.view_num(),bin.segment_num()); 
    
   return viewgram[bin.axial_pos_num()][bin.tangential_pos_num()]; 

}

void
ProjDataInMemory::
axpby(const float a, const ProjData& x,
      const float b, const ProjData& y)
{
    // To use this method, we require that all three proj data be ProjDataInMemory
    // So cast them. If any null pointers, fall back to default functionality
    ProjDataInMemory *x_pdm = dynamic_cast<ProjDataInMemory*>(this);
    ProjDataInMemory *y_pdm = dynamic_cast<ProjDataInMemory*>(this);
    // At least one is not ProjDataInMemory, fall back to default
    if (is_null_ptr(x_pdm) || is_null_ptr(y_pdm)) {
        ProjData::axpby(a,x,b,y);
        return;
    }
    // Get number of elements
    std::size_t numel = this->_num_elements;
    if (numel != x_pdm->_num_elements || numel != y_pdm->_num_elements)
        error("Expected number of elements to match");

    // Else, all are ProjDataInMemory
    float *buffer   = this->buffer.get();
    float *x_buffer = x_pdm->buffer.get();
    float *y_buffer = y_pdm->buffer.get();

    for (unsigned i=0; i<numel; ++i)
        buffer[i] = a*x_buffer[i] + b*y_buffer[i];
}

END_NAMESPACE_STIR

