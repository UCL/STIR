#ifndef __stir_IO_ECAT6ImageInputFileFormat_h__
#define __stir_IO_ECAT6ImageInputFileFormat_h__
/*
    Copyright (C) 2006, Hammersmith Imanet Ltd
    Copyright (C) 2013, University College London
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
  \brief Declaration of class stir::ecat::ecat6::ECAT6ImageInputFileFormat

  \author Kris Thielemans
*/
#include "stir/IO/InputFileFormat.h"
#include "stir/utilities.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include <fstream>
#include <string>
#include "stir/IO/stir_ecat6.h"
#include "stir/IO/ecat6_utils.h"
START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT6
//! Class for reading images in ECAT6 file-format.
/*! \ingroup ECAT
    \preliminary

*/
class ECAT6ImageInputFileFormat :
public InputFileFormat<DiscretisedDensity<3,float> >
{
 public:
  virtual const std::string
    get_name() const
  {  return "ECAT6"; }

 protected:
  virtual 
    bool 
    actual_can_read(const FileSignature& signature,
		    std::istream& input) const
  {
    return false;
  }

  bool
    can_read(const FileSignature& signature,
                                  std::istream& input) const
  {
    return false; // cannot read from istream
  }

  bool 
    can_read(const FileSignature&,
	     const std::string& filename) const
  {
    return is_ECAT6_image_file(filename);
  }

  virtual std::auto_ptr<data_type>
    read_from_file(std::istream& input) const
  {
    //TODO
    error("read_from_file for ECAT6 with istream not implemented %s:%d. Sorry",
	  __FILE__, __LINE__);
    return
      std::auto_ptr<data_type>
      (0);
  }
  virtual std::auto_ptr<data_type>
    read_from_file(const std::string& filename) const
  {
    if (is_ECAT6_image_file(filename))
      {
        ECAT6_Main_header mhead;
        FILE * cti_fptr=fopen(filename.c_str(), "rb"); 
        if(cti_read_ECAT6_Main_header(cti_fptr, &mhead)!=EXIT_SUCCESS) 
	  {
	    if (cti_fptr!=NULL)
	      fclose(cti_fptr);
	    error ("error reading main header in ECAT 6 file %s\n", filename.c_str());
	  }
        
	warning("\nReading frame 1, gate 1, data 0, bed 0 from file %s\n",
		filename.c_str());
        VoxelsOnCartesianGrid<float> * tmp =
          ECAT6_to_VoxelsOnCartesianGrid(/*frame_num, gate_num, data_num, bed_num*/1,1,0,0,
					 cti_fptr, mhead);
	fclose(cti_fptr);
	return std::auto_ptr<data_type>(tmp);
      }
    else
      {
	error("ECAT6ImageInputFileFormat: file '%s' does not look like an ECAT6 image.",
	      filename.c_str());
	// add return to avoid compiler warnings
	return std::auto_ptr<data_type>(0);
      }

  }
 };

END_NAMESPACE_ECAT
END_NAMESPACE_ECAT6
END_NAMESPACE_STIR


#endif
