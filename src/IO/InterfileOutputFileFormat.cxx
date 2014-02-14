//
//
/*
    Copyright (C) 2002-2007, Hammersmith Imanet Ltd
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
  \ingroup InterfileIO
  \brief Implementation of class stir::InterfileOutputFileFormat

  \author Kris Thielemans

*/

#include "stir/IO/InterfileOutputFileFormat.h"
#include "stir/IO/interfile.h"

START_NAMESPACE_STIR


const char * const 
InterfileOutputFileFormat::registered_name = "Interfile";

InterfileOutputFileFormat::
InterfileOutputFileFormat(const NumericType& type, 
                   const ByteOrder& byte_order) 
{
  base_type::set_defaults();
  set_type_of_numbers(type);
  set_byte_order(byte_order);
}

void 
InterfileOutputFileFormat::
set_defaults()
{
  base_type::set_defaults();
}

void 
InterfileOutputFileFormat::
initialise_keymap()
{
  parser.add_start_key("Interfile Output File Format Parameters");
  parser.add_stop_key("End Interfile Output File Format Parameters");
  base_type::initialise_keymap();
}

bool 
InterfileOutputFileFormat::
post_processing()
{
  if (base_type::post_processing())
    return true;
  return false;
}

// note 'warn' commented below to avoid compiler warning message about unused variables
ByteOrder 
InterfileOutputFileFormat::
set_byte_order(const ByteOrder& new_byte_order, const bool /* warn */) 
{
  this->file_byte_order = new_byte_order;
  return this->file_byte_order;
}



Succeeded  
InterfileOutputFileFormat::
    actual_write_to_file(string& filename, 
                  const DiscretisedDensity<3,float>& density) const
{
  // TODO modify write_basic_interfile to return filename
  
  Succeeded success =
    write_basic_interfile(filename, density, 
			  this->type_of_numbers, this->scale_to_write_data,
			  this->file_byte_order);
  if (success == Succeeded::yes)
    replace_extension(filename, ".hv");
  return success;
};

END_NAMESPACE_STIR


