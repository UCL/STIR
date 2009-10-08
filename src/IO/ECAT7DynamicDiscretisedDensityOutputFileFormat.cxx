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
  \brief Implementation of class stir::ecat::ecat7::ECAT7DynamicDiscretisedDensityOutputFileFormat

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#ifdef HAVE_LLN_MATRIX

#include "stir/IO/ECAT7DynamicDiscretisedDensityOutputFileFormat.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/NumericType.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

const char * const 
ECAT7DynamicDiscretisedDensityOutputFileFormat::registered_name = "ECAT7";

ECAT7DynamicDiscretisedDensityOutputFileFormat::
ECAT7DynamicDiscretisedDensityOutputFileFormat(const NumericType& type, 
                   const ByteOrder& byte_order) 
{
  this->set_defaults();
  this->set_type_of_numbers(type);
  this->set_byte_order(byte_order);
}

void 
ECAT7DynamicDiscretisedDensityOutputFileFormat::
initialise_keymap()
{
  this->parser.add_start_key("ECAT7 Output File Format Parameters");
  this->parser.add_stop_key("End ECAT7 Output File Format Parameters");
  this->parser.add_key("default scanner name", &default_scanner_name);
  base_type::initialise_keymap();
}

void 
ECAT7DynamicDiscretisedDensityOutputFileFormat::
set_defaults()
{
  this->default_scanner_name = "ECAT 962";
  base_type::set_defaults();
  this->file_byte_order = ByteOrder::big_endian;
  this->type_of_numbers = NumericType::SHORT;

  this->set_key_values();
}

bool
ECAT7DynamicDiscretisedDensityOutputFileFormat::
post_processing()
{
  if (base_type::post_processing())
    return true;

  shared_ptr<Scanner> scanner_ptr = 
    Scanner::get_scanner_from_name(this->default_scanner_name);

  if (find_ECAT_system_type(*scanner_ptr)==0)
    {
      warning("ECAT7DynamicDiscretisedDensityOutputFileFormat: default_scanner_name %s is not supported\n",
	      this->default_scanner_name.c_str());
      return true;
    }

  return false;
}

NumericType 
ECAT7DynamicDiscretisedDensityOutputFileFormat::
set_type_of_numbers(const NumericType& new_type, const bool warn)
{
 const NumericType supported_type_of_numbers = 
     NumericType("signed integer", 2);
  if (new_type != supported_type_of_numbers)
  {
    if (warn)
      warning("ECAT7DynamicDiscretisedDensityOutputFileFormat: output type of numbers is currently fixed to short (2 byte signed integers)\n");
    this->type_of_numbers = supported_type_of_numbers;
  }
  else
    this->type_of_numbers = new_type;
  return this->type_of_numbers;
}

ByteOrder 
ECAT7DynamicDiscretisedDensityOutputFileFormat::
set_byte_order(const ByteOrder& new_byte_order, const bool warn)
{
  if (new_byte_order != ByteOrder::big_endian)
  {
    if (warn)
      warning("ECAT7DynamicDiscretisedDensityOutputFileFormat: byte_order is currently fixed to big-endian\n");
    this->file_byte_order = ByteOrder::big_endian;
  }
  else
    this->file_byte_order = new_byte_order;
  return this->file_byte_order;
}

Succeeded  
ECAT7DynamicDiscretisedDensityOutputFileFormat::
    actual_write_to_file(string& filename, 
			 const DynamicDiscretisedDensity& dynamic_density) const
{
  add_extension(filename, ".img");
  return
   dynamic_density.write_to_ecat7(filename);
}

  
//template class ECAT7DynamicDiscretisedDensityOutputFileFormat;  

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

#endif // #ifdef HAVE_LLN_MATRIX
