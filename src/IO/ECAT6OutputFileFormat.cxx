//
// $Id$
//
/*
    Copyright (C) 2002-$Date$, Hammersmith Imanet Ltd
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
  \brief Implementation of class stir::ECAT6OutputFileFormat

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/IO/ECAT6OutputFileFormat.h"
#include "stir/NumericType.h"
#include "stir/IO/ecat6_utils.h"
#include "stir/IO/stir_ecat6.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT6


const char * const 
ECAT6OutputFileFormat::registered_name = "ECAT6";

ECAT6OutputFileFormat::
ECAT6OutputFileFormat(const NumericType& type, 
                   const ByteOrder& byte_order) 
{
  base_type::set_defaults();
  set_type_of_numbers(type);
  set_byte_order(byte_order);
}

void 
ECAT6OutputFileFormat::
initialise_keymap()
{
  parser.add_start_key("ECAT6 Output File Format Parameters");
  parser.add_stop_key("End ECAT6 Output File Format Parameters");
  parser.add_key("default scanner name", &default_scanner_name);
  base_type::initialise_keymap();
}

void 
ECAT6OutputFileFormat::
set_defaults()
{
  default_scanner_name = "ECAT 953";
  base_type::set_defaults();
  file_byte_order = ByteOrder::little_endian;
  type_of_numbers = NumericType::SHORT;

}

bool
ECAT6OutputFileFormat::
post_processing()
{
  if (base_type::post_processing())
    return true;

  shared_ptr<Scanner> scanner_ptr = 
    Scanner::get_scanner_from_name(default_scanner_name);

  if (find_ECAT_system_type(*scanner_ptr)==0)
    {
      warning("ECAT6OutputFileFormat: default_scanner_name %s is not supported\n",
	      default_scanner_name.c_str());
      return true;
    }

  return false;
}

NumericType 
ECAT6OutputFileFormat::
set_type_of_numbers(const NumericType& new_type, const bool warn)
{
 const NumericType supported_type_of_numbers = 
     NumericType("signed integer", 2);
  if (new_type != supported_type_of_numbers)
  {
    if (warn)
      warning("ECAT6OutputFileFormat: output type of numbers is currently fixed to short (2 byte signed integers)\n");
    type_of_numbers = supported_type_of_numbers;
  }
  else
    type_of_numbers = new_type;
  return type_of_numbers;

}

ByteOrder 
ECAT6OutputFileFormat::
set_byte_order(const ByteOrder& new_byte_order, const bool warn)
{
  if (new_byte_order != ByteOrder::little_endian)
  {
    if (warn)
      warning("ECAT6OutputFileFormat: byte_order is currently fixed to little-endian\n");
    file_byte_order = ByteOrder::little_endian;
  }
  else
    file_byte_order = new_byte_order;
  return file_byte_order;
}

Succeeded  
ECAT6OutputFileFormat::
    actual_write_to_file(string& filename, 
                  const DiscretisedDensity<3,float>& density) const
{
  shared_ptr<Scanner> scanner_ptr = 
    Scanner::get_scanner_from_name(default_scanner_name);
  
  add_extension(filename, ".img");

  ECAT6_Main_header mhead;
  make_ECAT6_Main_header(mhead, *scanner_ptr, "", density);
  mhead.num_frames = 1;
  
  FILE *fptr= cti_create (filename.c_str(), &mhead);
  if (fptr == NULL)
  {
    warning("ECAT6OutputFileFormat::write_to_file: error opening output file %s\n", 
	    filename.c_str());
    return Succeeded::no;
  }

  Succeeded success =
    DiscretisedDensity_to_ECAT6(fptr,
    density, 
    mhead,
    1 /*frame_num*/);
  fclose(fptr);
  
  return success;
}
END_NAMESPACE_ECAT6
END_NAMESPACE_ECAT
END_NAMESPACE_STIR


