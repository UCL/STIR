//
// $Id$
//
/*!

  \file
  \ingroup projdata
  \brief Inline implementations for class stir::ProjDataFromStream

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2009-06-22, Hammersmith Imanet Ltd
    Copyright (C) 2012-06-06 - $Date$, Kris Thielemans
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

START_NAMESPACE_STIR


//ProjDataFromStream::ProjDataFromStream()
//{}

ProjDataFromStream::StorageOrder 
ProjDataFromStream::get_storage_order() const
{ return storage_order; }

int 
ProjDataFromStream::find_segment_index_in_sequence(const int segment_num) const
{
#ifndef STIR_NO_NAMESPACES
  std::vector<int>::const_iterator iter =
    std::find(segment_sequence.begin(), segment_sequence.end(), segment_num);
#else
  vector<int>::const_iterator iter =
    find(segment_sequence.begin(), segment_sequence.end(), segment_num);
#endif
  // TODO do some proper error handling here
  assert(iter !=  segment_sequence.end());
  return static_cast<int>(iter - segment_sequence.begin());
}


std::streamoff 
ProjDataFromStream::get_offset_in_stream() const
{ return offset; }

NumericType 
ProjDataFromStream::get_data_type_in_stream() const
{ return on_disk_data_type; }

ByteOrder 
ProjDataFromStream::get_byte_order_in_stream() const
{ return on_disk_byte_order; }

std::vector<int> 
ProjDataFromStream::get_segment_sequence_in_stream() const
{ return segment_sequence; }

#if 0
// this does not make a lot of sense. How to compare files etc. ?
bool 
ProjDataFromStream::operator ==(const ProjDataFromStream& proj)
{
  
  return 
 (*get_proj_data_info_ptr())== *(proj.get_proj_data_info_ptr())&&
  (scale_factor ==proj.scale_factor)&&
  (get_segment_sequence_in_stream()==proj.get_segment_sequence_in_stream())&&
  (get_offset_in_stream()==proj.get_offset_in_stream()) &&
  (on_disk_data_type == proj.get_data_type_in_stream())&&
  (get_byte_order_in_stream() == proj.get_byte_order_in_stream()) ;
}
 

#endif



END_NAMESPACE_STIR
