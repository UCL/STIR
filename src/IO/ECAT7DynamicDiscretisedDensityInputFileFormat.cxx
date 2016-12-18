//
//
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
  \brief Implementation of class stir::ecat::ecat7::ECAT7DynamicDiscretisedDensityInputFileFormat

  \author Kris Thielemans
  \author Charalampos Tsoumpas

*/
#include "stir/IO/ECAT7DynamicDiscretisedDensityInputFileFormat.h"
#include "stir/Succeeded.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h" // necessary as stir_ecat7 reading routine returns a VoxelsOnCartesianGrid
#include "stir/is_null_ptr.h"
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
bool 
ECAT7DynamicDiscretisedDensityInputFileFormat::
actual_can_read(const FileSignature& signature,
		std::istream& input) const
{
  if (strncmp(signature.get_signature(), "MATRIX", 6) != 0)
    return false;

  // TODO
  // return (is_ECAT7_image_file(filename))
  return true;
}

std::unique_ptr<ECAT7DynamicDiscretisedDensityInputFileFormat::data_type>
ECAT7DynamicDiscretisedDensityInputFileFormat::
read_from_file(std::istream& input) const
{
  //TODO
  error("read_from_file for ECAT7 with istream not implemented %s:%s. Sorry",
	__FILE__, __LINE__);
  return
    std::unique_ptr<data_type>
    (0);
}

std::unique_ptr<ECAT7DynamicDiscretisedDensityInputFileFormat::data_type>
ECAT7DynamicDiscretisedDensityInputFileFormat::
read_from_file(const std::string& filename) const
{
  if (is_ECAT7_image_file(filename))
    {
      Main_header mhead;
      if (read_ECAT7_main_header(mhead, filename) == Succeeded::no)
	{
	  warning("ECAT7DynamicDiscretisedDensityInputFileFormat::read_from_file cannot read %s as ECAT7 (failed to read main header)", filename.c_str());
	  return std::unique_ptr<data_type>(0);
	}

      TimeFrameDefinitions time_frame_definitions(filename);
      shared_ptr<Scanner> scanner_sptr(find_scanner_from_ECAT_system_type(mhead.system_type));

      std::unique_ptr<data_type> 
	dynamic_image_ptr
	(new DynamicDiscretisedDensity(time_frame_definitions,
				       static_cast<double>(mhead.scan_start_time),
				       scanner_sptr)
	 );

      dynamic_image_ptr->set_calibration_factor(mhead.calibration_factor);

      dynamic_image_ptr->set_isotope_halflife(mhead.isotope_halflife);

      // TODO get this from the subheader fields or so
      // dynamic_image_ptr->_is_decay_corrected =
      //  shead.processing_code & DecayPrc
      dynamic_image_ptr->set_if_decay_corrected(false);

      for (unsigned int frame_num=1; frame_num <= dynamic_image_ptr->get_num_time_frames(); ++ frame_num)
	{
	  shared_ptr<DynamicDiscretisedDensity::singleDiscDensT> dens_sptr
	    (ECAT7_to_VoxelsOnCartesianGrid(filename,
					    frame_num, 
					    /* gate_num, data_num, bed_num */ 1,0,0)
	     );
	  if (is_null_ptr(dens_sptr))
	    error("read_from_file for DynamicDiscretisedDensity: No frame %d available", frame_num);
	  dynamic_image_ptr->set_density_sptr(dens_sptr, frame_num );
	}
      return dynamic_image_ptr;
    }
  else
    {
      error("read_from_file for DynamicDiscretisedDensity: ECAT7 file %s is not an image file", filename.c_str());
      // return something to satisfy compilers
      return std::unique_ptr<data_type>(0);
    }
}

END_NAMESPACE_ECAT
END_NAMESPACE_ECAT7
END_NAMESPACE_STIR

