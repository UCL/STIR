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
  \brief Implementation of class stir::ecat::ecat7::ECAT7ParametricDensityOutputFileFormat

  \author Kris Thielemans

*/

#ifdef HAVE_LLN_MATRIX

#include "stir/IO/ECAT7ParametricDensityOutputFileFormat.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/NumericType.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

template <class DiscretisedDensityT>
const char * const 
ECAT7ParametricDensityOutputFileFormat<DiscretisedDensityT>::registered_name = "ECAT7";

template <class DiscretisedDensityT>
ECAT7ParametricDensityOutputFileFormat<DiscretisedDensityT>::
ECAT7ParametricDensityOutputFileFormat(const NumericType& type, 
                   const ByteOrder& byte_order) 
{
  this->set_defaults();
  this->set_type_of_numbers(type);
  this->set_byte_order(byte_order);
}

template <class DiscretisedDensityT>
void 
ECAT7ParametricDensityOutputFileFormat<DiscretisedDensityT>::
initialise_keymap()
{
  this->parser.add_start_key("ECAT7 Output File Format Parameters");
  this->parser.add_stop_key("End ECAT7 Output File Format Parameters");
  this->parser.add_key("default scanner name", &default_scanner_name);
  base_type::initialise_keymap();
}

template <class DiscretisedDensityT>
void 
ECAT7ParametricDensityOutputFileFormat<DiscretisedDensityT>::
set_defaults()
{
  this->default_scanner_name = "ECAT 962";
  base_type::set_defaults();
  this->file_byte_order = ByteOrder::big_endian;
  this->type_of_numbers = NumericType::SHORT;

  this->set_key_values();

}

template <class DiscretisedDensityT>
bool
ECAT7ParametricDensityOutputFileFormat<DiscretisedDensityT>::
post_processing()
{
  if (base_type::post_processing())
    return true;

  shared_ptr<Scanner> scanner_ptr(
    Scanner::get_scanner_from_name(this->default_scanner_name));

  if (find_ECAT_system_type(*scanner_ptr)==0)
    {
      warning("ECAT7ParametricDensityOutputFileFormat: default_scanner_name %s is not supported\n",
	      this->default_scanner_name.c_str());
      return true;
    }

  return false;
}

template <class DiscretisedDensityT>
NumericType 
ECAT7ParametricDensityOutputFileFormat<DiscretisedDensityT>::
set_type_of_numbers(const NumericType& new_type, const bool warn)
{
 const NumericType supported_type_of_numbers = 
     NumericType("signed integer", 2);
  if (new_type != supported_type_of_numbers)
  {
    if (warn)
      warning("ECAT7ParametricDensityOutputFileFormat: output type of numbers is currently fixed to short (2 byte signed integers)\n");
    this->type_of_numbers = supported_type_of_numbers;
  }
  else
    this->type_of_numbers = new_type;
  return this->type_of_numbers;

}

template <class DiscretisedDensityT>
ByteOrder 
ECAT7ParametricDensityOutputFileFormat<DiscretisedDensityT>::
set_byte_order(const ByteOrder& new_byte_order, const bool warn)
{
  if (new_byte_order != ByteOrder::big_endian)
  {
    if (warn)
      warning("ECAT7ParametricDensityOutputFileFormat: byte_order is currently fixed to big-endian\n");
    this->file_byte_order = ByteOrder::big_endian;
  }
  else
    this->file_byte_order = new_byte_order;
  return this->file_byte_order;
}

template <class DiscretisedDensityT>
Succeeded  
ECAT7ParametricDensityOutputFileFormat<DiscretisedDensityT>::
    actual_write_to_file(string& filename, 
			 const ParametricDiscretisedDensity<DiscretisedDensityT>& parametric_density) const
{
  shared_ptr<Scanner> scanner_ptr(
				  Scanner::get_scanner_from_name(this->default_scanner_name));
  
  add_extension(filename, ".img");

  typedef
    typename ParametricDiscretisedDensity<DiscretisedDensityT>::SingleDiscretisedDensityType
    SingleDensityType;
  // somewhat naughty trick to get elemT of DiscretisedDensityT
  typedef typename DiscretisedDensityT::full_value_type KinParsT;

  SingleDensityType single_density = parametric_density.construct_single_density(1);

  Main_header mhead;
  make_ECAT7_main_header(mhead, *scanner_ptr, "", single_density);
  mhead.num_frames = KinParsT::size();

  mhead.acquisition_type = DynamicEmission;
  
  MatrixFile* mptr= matrix_create (filename.c_str(), MAT_CREATE, &mhead);
  if (mptr == 0)
  {
    warning("ECAT7ParametricDensityOutputFileFormat::write_to_file: error opening output file %s\n",
      filename.c_str());
    return Succeeded::no;
  }
  
  for (unsigned int f=1; f<=KinParsT::size(); ++f)
    {      
      parametric_density.construct_single_density(single_density,f);
      Succeeded success =
	DiscretisedDensity_to_ECAT7(mptr, single_density, f);
      if (success == Succeeded::no)
	{
	    matrix_close(mptr);
	    return success;
	}
    }
  matrix_close(mptr);
  
  return Succeeded::yes;
}

template class ECAT7ParametricDensityOutputFileFormat<ParametricVoxelsOnCartesianGridBaseType>;  

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

#endif // #ifdef HAVE_LLN_MATRIX
