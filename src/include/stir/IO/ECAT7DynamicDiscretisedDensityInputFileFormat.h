//
//
#ifndef __stir_IO_ECAT7DynamicDiscretisedDensityInputFileFormat_h__
#define __stir_IO_ECAT7DynamicDiscretisedDensityInputFileFormat_h__
/*
    Copyright (C) 2006 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
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
  \brief Declaration of class stir::ecat::ecat7::ECAT7DynamicDiscretisedDensityInputFileFormat

  \author Kris Thielemans
  \author Charalampos Tsoumpas

*/
#include "stir/IO/InputFileFormat.h"
#include "stir/DynamicDiscretisedDensity.h"
#include <fstream>
#include <string>


#ifndef HAVE_LLN_MATRIX
#error HAVE_LLN_MATRIX not defined: you need the lln ecat library.
#endif

#include "stir/IO/stir_ecat7.h"
START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

//! Class for reading images in ECAT7 file-format.
/*! \ingroup ECAT
    \preliminary

*/
class ECAT7DynamicDiscretisedDensityInputFileFormat :
public InputFileFormat<DynamicDiscretisedDensity >
{
 public:
  virtual const std::string
    get_name() const
  {  return "ECAT7"; }

 protected:
  virtual 
    bool 
    actual_can_read(const FileSignature& signature,
		    std::istream& input) const;

  //! This always fails
  virtual unique_ptr<data_type>
    read_from_file(std::istream& input) const;

  //! read data given a filename
  virtual unique_ptr<data_type>
    read_from_file(const std::string& filename) const;
};

END_NAMESPACE_ECAT
END_NAMESPACE_ECAT7
END_NAMESPACE_STIR

#endif
