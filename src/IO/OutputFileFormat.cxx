//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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
  \ingroup IO
  
  \brief  implementation of the stir::OutputFileFormat class 
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/

#include "stir/IO/OutputFileFormat.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

template <typename DataT>
ASCIIlist_type
OutputFileFormat<DataT>::number_format_values;

template <typename DataT>
ASCIIlist_type
OutputFileFormat<DataT>::byte_order_values;

template <typename DataT>
shared_ptr<OutputFileFormat<DataT> >
OutputFileFormat<DataT>::
default_sptr()
{
  return OutputFileFormat<DataT>::_default_sptr;
}

template <typename DataT>
OutputFileFormat<DataT>::
OutputFileFormat(const NumericType& type,
		 const ByteOrder& byte_order)
  : type_of_numbers(type), file_byte_order(byte_order)
{ 
}

template <typename DataT>
void 
OutputFileFormat<DataT>::
set_defaults()
{
  file_byte_order = ByteOrder::native;
  type_of_numbers = NumericType::FLOAT;
  scale_to_write_data = 0.F;

  set_key_values();
}

template <typename DataT>
void 
OutputFileFormat<DataT>::
initialise_keymap()
{

  if (number_format_values.size()==0)
  {
    number_format_values.push_back("bit");
    number_format_values.push_back("ascii");
    number_format_values.push_back("signed integer");
    number_format_values.push_back("unsigned integer");
    number_format_values.push_back("float");

    assert(byte_order_values.size()==0);
    byte_order_values.push_back("LITTLEENDIAN");
    byte_order_values.push_back("BIGENDIAN");
  }
  parser.add_key("byte order",
		 &byte_order_index,
		 &byte_order_values);
  parser.add_key("number format",
		 &number_format_index,
		 &number_format_values);
  parser.add_key("number of bytes per pixel",
		 &bytes_per_pixel);
  parser.add_key("scale_to_write_data",
		 &scale_to_write_data);
}

template <typename DataT>
void 
OutputFileFormat<DataT>::
set_key_values()
{
  byte_order_index =
    file_byte_order == ByteOrder::little_endian ?
      0 : 1;

  string number_format; 
  size_t bytes_per_pixel_in_size_t;
  type_of_numbers.get_Interfile_info(number_format, 
                                     bytes_per_pixel_in_size_t);
  bytes_per_pixel = static_cast<int>(bytes_per_pixel_in_size_t);
  number_format_index =
     parser.find_in_ASCIIlist(number_format, number_format_values); 

}

template <typename DataT>
bool
OutputFileFormat<DataT>::
post_processing()
{
  if (number_format_index<0 ||
      static_cast<ASCIIlist_type::size_type>(number_format_index)>=
        number_format_values.size())
  {
    warning("OutputFileFormat parser internal error: "
            "'number_format_index' out of range\n");
    return true;
  }
  // check if bytes_per_pixel is set if the data type is not 'bit'
  if (number_format_index!=0 && bytes_per_pixel<=0)
  {
    warning("OutputFileFormat parser: "
            "'number_of_bytes_per_pixel' should be > 0\n");
    return true;
  }

  type_of_numbers = NumericType(number_format_values[number_format_index], 
                                bytes_per_pixel);

  if (type_of_numbers == NumericType::UNKNOWN_TYPE)
  {
    warning("OutputFileFormat parser: "
        "unrecognised data type. Check either "
        "'number_of_bytes_per_pixel' or 'number format'\n");
    return true;
  }

  switch(byte_order_index)
  {
    case -1:
	file_byte_order = ByteOrder::native; break;
    case 0:
	file_byte_order = ByteOrder::little_endian; break;
    case 1:
	file_byte_order = ByteOrder::big_endian; break;
    default:
        warning("OutputFileFormat parser internal error: "
                "'byte_order_index' out of range\n");
        return true;
  }

  set_byte_order(file_byte_order, true);
  set_type_of_numbers(type_of_numbers, true);

  return false;
}

template <typename DataT>
NumericType 
OutputFileFormat<DataT>::
get_type_of_numbers() const
{ return type_of_numbers; }

template <typename DataT>
ByteOrder 
OutputFileFormat<DataT>::
get_byte_order()
{ return file_byte_order; }

template <typename DataT>
float
OutputFileFormat<DataT>::
get_scale_to_write_data() const
{ return scale_to_write_data; }

template <typename DataT>
NumericType 
OutputFileFormat<DataT>::
set_type_of_numbers(const NumericType& new_type, const bool warn)
{ return type_of_numbers =  new_type; }

template <typename DataT>
ByteOrder 
OutputFileFormat<DataT>::
set_byte_order(const ByteOrder& new_byte_order, const bool warn)
{ return file_byte_order = new_byte_order; }

template <typename DataT>
float
OutputFileFormat<DataT>::
set_scale_to_write_data(const float new_scale_to_write_data, const bool warn)
{ return scale_to_write_data = new_scale_to_write_data; }

template <typename DataT>
void
OutputFileFormat<DataT>::
set_byte_order_and_type_of_numbers(ByteOrder& new_byte_order, NumericType& new_type, const bool warn)
{
  new_byte_order = set_byte_order(new_byte_order, warn);
  new_type = set_type_of_numbers(new_type, warn);
}

template <typename DataT>
Succeeded  
OutputFileFormat<DataT>::
write_to_file(string& filename, 
	      const DataT& density) const
{
  return actual_write_to_file(filename, density);
}

template <typename DataT>
Succeeded  
OutputFileFormat<DataT>::
write_to_file(const string& filename, 
	      const DataT& density) const
{
  string filename_to_use = filename;
  return 
    write_to_file(filename_to_use, density);
};
 
template class OutputFileFormat<DiscretisedDensity<3,float> >; 

END_NAMESPACE_STIR

#ifdef STIR_DEVEL 
#include "local/stir/modelling/ParametricDiscretisedDensity.h" 
#include "local/stir/modelling/KineticParameters.h" 
namespace stir { 
  template class OutputFileFormat<ParametricVoxelsOnCartesianGrid >;  
} 
#endif 

