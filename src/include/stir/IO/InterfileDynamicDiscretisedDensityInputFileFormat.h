//
// $Id: InterfileDynamicDiscretisedDensityInputFileFormat.h,v 1.0 2014-05-03 18:00:00 nkarakatsanis Exp $
//
#ifndef __stir_IO_InterfileDynamicDiscretisedDensityInputFileFormat_h__
#define __stir_IO_InterfileDynamicDiscretisedDensityInputFileFormat_h__
/*
    Copyright (C) 2006 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - $Date: 2014-05-03 18:00:00 $, Nicolas A Karakatsanis
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
  \ingroup Interfile
  \brief Declaration of class stir::InterfileDynamicDiscretisedDensityInputFileFormat

  \author Nicolas A Karakatsanis

  $Date: 2014-03-05 18:00:00 $
  $Revision: 1.1 $
*/
#include "stir/IO/InputFileFormat.h"
#include "stir/IO/interfile.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/error.h"
#include "stir/warning.h"
#include "stir/is_null_ptr.h"
#include "stir/utilities.h"
#include <iostream>
#include <fstream>
#include <string>
using std::ifstream;


START_NAMESPACE_STIR


//! Class for reading dynamic images in Interfile file-format.
/*! \ingroup Interfile IO
    \preliminary

*/

class InterfileDynamicDiscretisedDensityInputFileFormat :
public InputFileFormat<DynamicDiscretisedDensity >
{
public:
  virtual const std::string
    get_name() const
  {  return "Interfile"; }

protected:
  virtual 
    bool 
    actual_can_read(const FileSignature& signature,
		    std::istream& input) const;

  //! This always fails
  virtual std::auto_ptr<data_type>
    read_from_file(std::istream& input) const;

  //! read data given a filename
  virtual std::auto_ptr<data_type>
    read_from_file(const std::string& filename) const;

 private:
    std::auto_ptr<data_type>
    read_interfile_dyn_image(std::istream& input,
	         const string& directory_for_data = "",
		     const std::ios::openmode open_mode = std::ios::in) const;

    std::auto_ptr<data_type>
    read_interfile_dyn_image(const string& filename,
		     const std::ios::openmode open_mode = std::ios::in) const;			 

 //helper function to read dynamic Interfile image data from filename	
 /*
 static std::auto_ptr<data_type>
    read_interfile_dyn_image(std::istream& input,
	         const string& directory_for_data,
		     const std::ios::openmode open_mode = std::ios::in);
			 
  //helper function to read dynamic Interfile image data from filename	
  static std::auto_ptr<data_type>
    read_interfile_dyn_image(const string& filename,
		     const std::ios::openmode open_mode = std::ios::in);
	
*/	
};

END_NAMESPACE_STIR

#endif
