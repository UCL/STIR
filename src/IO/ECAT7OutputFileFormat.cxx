//
//
/*
    Copyright (C) 2002-2011, Hammersmith Imanet Ltd
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
  \ingroup ECAT
  \brief Implementation of class stir::ecat::ecat7::ECAT7OutputFileFormat

  \author Kris Thielemans

*/

#ifdef HAVE_LLN_MATRIX

#include "stir/IO/ECAT7OutputFileFormat.h"
#include "stir/NumericType.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

const char * const 
ECAT7OutputFileFormat::registered_name = "ECAT7";

ECAT7OutputFileFormat::
ECAT7OutputFileFormat(const NumericType& type, 
                   const ByteOrder& byte_order) 
{
  base_type::set_defaults();
  set_type_of_numbers(type);
  set_byte_order(byte_order);
}

void 
ECAT7OutputFileFormat::
initialise_keymap()
{
  parser.add_start_key("ECAT7 Output File Format Parameters");
  parser.add_stop_key("End ECAT7 Output File Format Parameters");
  parser.add_key("default scanner name", &default_scanner_name);
  base_type::initialise_keymap();
}

void 
ECAT7OutputFileFormat::
set_defaults()
{
  default_scanner_name = "ECAT 962";
  base_type::set_defaults();
  file_byte_order = ByteOrder::big_endian;
  type_of_numbers = NumericType::SHORT;

  set_key_values();

}

bool
ECAT7OutputFileFormat::
post_processing()
{
  if (base_type::post_processing())
    return true;

  shared_ptr<Scanner> scanner_ptr(Scanner::get_scanner_from_name(default_scanner_name));

  if (find_ECAT_system_type(*scanner_ptr)==0)
    {
      warning("ECAT7OutputFileFormat: default_scanner_name %s is not supported\n",
	      default_scanner_name.c_str());
      return true;
    }

  return false;
}

NumericType 
ECAT7OutputFileFormat::
set_type_of_numbers(const NumericType& new_type, const bool warn)
{
 const NumericType supported_type_of_numbers = 
     NumericType("signed integer", 2);
  if (new_type != supported_type_of_numbers)
  {
    if (warn)
      warning("ECAT7OutputFileFormat: output type of numbers is currently fixed to short (2 byte signed integers)\n");
    type_of_numbers = supported_type_of_numbers;
  }
  else
    type_of_numbers = new_type;
  return type_of_numbers;

}

ByteOrder 
ECAT7OutputFileFormat::
set_byte_order(const ByteOrder& new_byte_order, const bool warn)
{
  if (new_byte_order != ByteOrder::big_endian)
  {
    if (warn)
      warning("ECAT7OutputFileFormat: byte_order is currently fixed to big-endian\n");
    file_byte_order = ByteOrder::big_endian;
  }
  else
    file_byte_order = new_byte_order;
  return file_byte_order;
}

Succeeded  
ECAT7OutputFileFormat::
    actual_write_to_file(string& filename, 
                  const DiscretisedDensity<3,float>& density) const
{
  shared_ptr<Scanner> scanner_ptr(Scanner::get_scanner_from_name(default_scanner_name));
  
  add_extension(filename, ".img");

  Main_header mhead;
  make_ECAT7_main_header(mhead, *scanner_ptr, "", density);
  mhead.num_frames = 1;
  mhead.acquisition_type =
    mhead.num_frames>1 ? DynamicEmission : StaticEmission;
  
  MatrixFile* mptr= matrix_create (filename.c_str(), MAT_CREATE, &mhead);
  if (mptr == 0)
  {
    warning("ECAT7OutputFileFormat::write_to_file: error opening output file %s\n",
      filename.c_str());
    return Succeeded::no;
  }
  
  Succeeded success =
    DiscretisedDensity_to_ECAT7(mptr, density, 1 /*frame_num*/);
  matrix_close(mptr);
  
  return success;
}

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR


#endif // #ifdef HAVE_LLN_MATRIX
