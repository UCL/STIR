//
//
/*
  Copyright (C) 2006-2009, Hammersmith Imanet Ltd
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

/*!\file
  \ingroup InterfileIO
  \brief Implementation of class stir::InterfileDynamicDiscretisedDensityOutputFileFormat

\author Kris Thielemans
\author Charalampos Tsoumpas

*/

#include "stir/IO/InterfileDynamicDiscretisedDensityOutputFileFormat.h"
#include "stir/DynamicDiscretisedDensity.h" 
#include "stir/NumericType.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

const char * const 
InterfileDynamicDiscretisedDensityOutputFileFormat::registered_name = "Interfile";

InterfileDynamicDiscretisedDensityOutputFileFormat::
InterfileDynamicDiscretisedDensityOutputFileFormat(const NumericType& type, 
					const ByteOrder& byte_order) 
{
  base_type::set_defaults();
  this->set_type_of_numbers(type);
  this->set_byte_order(byte_order);
}

void 
InterfileDynamicDiscretisedDensityOutputFileFormat::
set_defaults()
{
  base_type::set_defaults();
}

void 
InterfileDynamicDiscretisedDensityOutputFileFormat::
initialise_keymap()
{
  this->parser.add_start_key("Interfile Output File Format");
  this->parser.add_stop_key("End Interfile Output File Format");
  base_type::initialise_keymap();
}

bool 
InterfileDynamicDiscretisedDensityOutputFileFormat::
post_processing()
{
  if (base_type::post_processing())
    return true;
  return false;
}


ByteOrder 
InterfileDynamicDiscretisedDensityOutputFileFormat::
set_byte_order(const ByteOrder& new_byte_order, const bool warn)
{
  if (!new_byte_order.is_native_order())
    {
      if (warn)
	warning("InterfileDynamicDiscretisedDensityOutputFileFormat: byte_order is currently fixed to the native format\n");
      this->file_byte_order = ByteOrder::native;
    }
  else
    this->file_byte_order = new_byte_order;
  return  this->file_byte_order;
}

Succeeded  
InterfileDynamicDiscretisedDensityOutputFileFormat::
actual_write_to_file(string& filename, 
		     const DynamicDiscretisedDensity & density) const
{
  // TODO modify write_basic_interfile to return filename
  
  error("InterfileDynamicDiscretisedDensityOutputFileFormat TODO");
  return Succeeded::no;
};



// class InterfileDynamicDiscretisedDensityOutputFileFormat;  


END_NAMESPACE_STIR
