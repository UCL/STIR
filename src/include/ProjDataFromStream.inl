//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock
  \brief Inline implementations for class ProjDataFromStream

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

START_NAMESPACE_TOMO


//ProjDataFromStream::ProjDataFromStream()
//{}

ProjDataFromStream::StorageOrder 
ProjDataFromStream::get_storage_order() const
{ return storage_order; }

int 
ProjDataFromStream::find_segment_index_in_sequence(const int segment_num) const
{
#ifndef TOMO_NO_NAMESPACES
  std::vector<int>::const_iterator iter =
    std::find(segment_sequence.begin(), segment_sequence.end(), segment_num);
#else
  vector<int>::const_iterator iter =
    find(segment_sequence.begin(), segment_sequence.end(), segment_num);
#endif
  // TODO do some proper error handling here
  assert(iter !=  segment_sequence.end());
  return iter - segment_sequence.begin();
}


streamoff 
ProjDataFromStream::get_offset_in_stream() const
{ return offset; }

NumericType 
ProjDataFromStream::get_data_type_in_stream() const
{ return on_disk_data_type; }

ByteOrder 
ProjDataFromStream::get_byte_order_in_stream() const
{ return on_disk_byte_order; }

vector<int> 
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



END_NAMESPACE_TOMO
