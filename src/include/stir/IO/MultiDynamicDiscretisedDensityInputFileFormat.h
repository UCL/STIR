//
//
#ifndef __stir_IO_MultiDynamicDiscretisedDensityInputFileFormat_h__
#define __stir_IO_MultiDynamicDiscretisedDensityInputFileFormat_h__
/*
    Copyright (C) 2006 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2013-01-01 - 2013, Kris Thielemans
    Copyight (C) 2018, University College London
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
  \ingroup IO
  \brief Declaration of class stir::MultiDynamicDiscretisedDensityInputFileFormat

  \author Kris Thielemans
  \author Richard Brown

*/
#include "stir/IO/InputFileFormat.h"
#include "stir/IO/interfile.h"
#include "stir/utilities.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/error.h"
#include "stir/is_null_ptr.h"
#include "stir/MultipleDataSetHeader.h"
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR

//! Class for reading images in Multi file-format.
/*! \ingroup IO

*/
class MultiDynamicDiscretisedDensityInputFileFormat :
public InputFileFormat<DynamicDiscretisedDensity>
{
 public:
  virtual const std::string
    get_name() const
  {  return "Multi"; }

 protected:
  virtual 
    bool 
    actual_can_read(const FileSignature& signature,
		    std::istream& input) const
  {
    //. todo should check if it's an image
    // checking for "multi :"
    const char * pos_of_colon = strchr(signature.get_signature(), ':');
    if (pos_of_colon == NULL)
      return false;
    std::string keyword(signature.get_signature(), pos_of_colon-signature.get_signature());
    return (
            standardise_interfile_keyword(keyword) == 
            standardise_interfile_keyword("multi"));
  }

  virtual unique_ptr<data_type>
    read_from_file(std::istream& input) const
  {
    // needs more arguments, so we just give up (TODO?)
    unique_ptr<data_type> ret;
    if (is_null_ptr(ret))
      {
	error("failed to read an Multi image from stream");
      }
    return ret;
  }
  virtual unique_ptr<data_type>
    read_from_file(const std::string& filename) const
  {
    MultipleDataSetHeader header;
    if (header.parse(filename.c_str()) == false)
         error("MultiDynamicDiscretisedDensity:::read_from_file: Error parsing %s", filename.c_str());
    
    DynamicDiscretisedDensity* dyn_disc_den_ptr = new DynamicDiscretisedDensity;
    dyn_disc_den_ptr->set_num_densities(header.get_num_data_sets());
    TimeFrameDefinitions time_frames;
    time_frames.set_num_time_frames(header.get_num_data_sets());
    
    for (int i=0; i<header.get_num_data_sets(); ++i) {   
        shared_ptr<DiscretisedDensity<3,float> > t(DiscretisedDensity<3,float>::read_from_file(header.get_filename(i)));
        dyn_disc_den_ptr->set_density_sptr(t,i+1);
        double start = t->get_exam_info().get_time_frame_definitions().get_start_time(1);
        double end = t->get_exam_info().get_time_frame_definitions().get_end_time(1);
        time_frames.set_time_frame(i+1, start, end);
        
        // Set some info on the first frame
        if (i==0) {
            dyn_disc_den_ptr->get_exam_info_sptr()->start_time_in_secs_since_1970 = 
                    t->get_exam_info().start_time_in_secs_since_1970;
            shared_ptr<Scanner> scanner_sptr(Scanner::get_scanner_from_name(t->get_exam_info_sptr()->originating_system));
            dyn_disc_den_ptr->set_scanner(scanner_sptr);
        }
    }    
    
    dyn_disc_den_ptr->set_time_frame_definitions(time_frames);
    // Hard wire some stuff for now (TODO?)
    dyn_disc_den_ptr->set_calibration_factor(1.);
    dyn_disc_den_ptr->set_if_decay_corrected(1.);
    dyn_disc_den_ptr->set_isotope_halflife(6586.2);
    
    return unique_ptr<data_type>(dyn_disc_den_ptr);
  }
};
END_NAMESPACE_STIR

#endif
