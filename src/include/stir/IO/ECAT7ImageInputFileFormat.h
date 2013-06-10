//
// $Id$
//
#ifndef __stir_IO_ECAT7ImageInputFileFormat_h__
#define __stir_IO_ECAT7ImageInputFileFormat_h__
/*
    Copyright (C) 2006- $Date$, Hammersmith Imanet Ltd
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
  \brief Declaration of class stir::ecat::ecat7::ECAT7ImageInputFileFormat

  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "stir/IO/InputFileFormat.h"
#include "stir/utilities.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include <fstream>
#include <string>


#ifndef HAVE_LLN_MATRIX
#error HAVE_LLN_MATRIX not define: you need the lln ecat library.
#endif

#include "stir/IO/stir_ecat7.h"
START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

//! Class for reading images in ECAT7 file-format.
/*! \ingroup ECAT
    \preliminary

*/
class ECAT7ImageInputFileFormat :
public InputFileFormat<DiscretisedDensity<3,float> >
{
 public:
  virtual const std::string
    get_name() const
  {  return "ECAT7"; }

 protected:
  virtual 
    bool 
    actual_can_read(const FileSignature& signature,
		    std::istream& input) const
  {
    if (strncmp(signature.get_signature(), "MATRIX", 6) != 0)
      return false;

    // TODO
    // return (is_ECAT7_image_file(filename))
    return true;
  }

  virtual std::auto_ptr<data_type>
    read_from_file(std::istream& input) const
  {
    //TODO
    error("read_from_file for ECAT7 with istream not implemented %s:%d. Sorry",
	  __FILE__, __LINE__);
    return
      std::auto_ptr<data_type>
      (0);
  }
  virtual std::auto_ptr<data_type>
    read_from_file(const std::string& filename) const
  {

    if (is_ECAT7_image_file(filename))
      {
	warning("\nReading frame 1, gate 1, data 0, bed 0 from file %s\n",
		filename.c_str());
	return std::auto_ptr<data_type>
	  (
	   ECAT7_to_VoxelsOnCartesianGrid(filename,
					  /*frame_num, gate_num, data_num, bed_num*/1,1,0,0));
      }
    else
      {
	warning("ECAT7 file %s is not an image file", filename.c_str());
	return std::auto_ptr<data_type>(0);
      }
  }
};

END_NAMESPACE_ECAT
END_NAMESPACE_ECAT7
END_NAMESPACE_STIR

#endif
