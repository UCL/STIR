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
  \ingroup InterfileIO
  \brief Implementation of class stir::InterfileParametricDensityOutputFileFormat

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "local/stir/IO/InterfileParametricDensityOutputFileFormat.h"
#include "local/stir/modelling/KineticParameters.h" 
#include "local/stir/modelling/ParametricDiscretisedDensity.h" 
#include "stir/NumericType.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

//#define TEMPLATE template <int num_dimensions, typename elemT>
#define TEMPLATE template <typename DiscDensityT>
//#define InterfileParamDiscDensity InterfileParametricDensityOutputFileFormat<num_dimensions,elemT>
#define InterfileParamDiscDensity InterfileParametricDensityOutputFileFormat<DiscDensityT>


TEMPLATE
const char * const 
InterfileParamDiscDensity::registered_name = "Interfile";

TEMPLATE
InterfileParamDiscDensity::
InterfileParametricDensityOutputFileFormat(const NumericType& type, 
					   const ByteOrder& byte_order) 
{
  base_type::set_defaults();
  this->set_type_of_numbers(type);
  this->set_byte_order(byte_order);
}

TEMPLATE
void 
InterfileParamDiscDensity::
set_defaults()
{
  base_type::set_defaults();
}

TEMPLATE
void 
InterfileParamDiscDensity::
initialise_keymap()
{
  this->parser.add_start_key("Interfile Output File Format Parameters");
   this->parser.add_stop_key("End Interfile Output File Format Parameters");
  base_type::initialise_keymap();
}

TEMPLATE
bool 
InterfileParamDiscDensity::
post_processing()
{
  if (base_type::post_processing())
    return true;
  return false;
}


TEMPLATE
ByteOrder 
InterfileParamDiscDensity::
set_byte_order(const ByteOrder& new_byte_order, const bool warn)
{
  if (!new_byte_order.is_native_order())
  {
    if (warn)
      warning("InterfileParametricDensityOutputFileFormat: byte_order is currently fixed to the native format\n");
     this->file_byte_order = ByteOrder::native;
  }
  else
     this->file_byte_order = new_byte_order;
  return  this->file_byte_order;
}



TEMPLATE
Succeeded  
InterfileParamDiscDensity::
    actual_write_to_file(string& filename, 
			 const ParametricDiscretisedDensity<DiscDensityT>& density) const
{
  // TODO modify write_basic_interfile to return filename
  
  error("InterfileParametricDensityOutputFileFormat TODO");
  return Succeeded::no;
};

#undef ParamDiscDensity
#undef TEMPLATE

template class InterfileParametricDensityOutputFileFormat<ParametricVoxelsOnCartesianGridBaseType>;  


END_NAMESPACE_STIR
