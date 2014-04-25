//
// $Id: InterfileDynamicDiscretisedDensityInputFileFormat.cxx,v 1.1 2014-03-05 18:00:00 nkarakatsanis Exp $
//
/*
    Copyright (C) 2006 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - $Date: 2014-03-05 18:00:00 $, Kris Thielemans
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
  \brief Implementation of class stir::InterfileDynamicDiscretisedDensityInputFileFormat

  \author Kris Thielemans
  \author Charalampos Tsoumpas

  $Date: 2014-03-05 18:00:00 $
  $Revision: 1.1 $
*/
#include "stir/IO/InterfileDynamicDiscretisedDensityInputFileFormat.h"
#include "stir/Succeeded.h"
#include "stir/IO/InterfileHeader.h"
#include "stir/IO/interfile.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h" // necessary as interfile reading routine returns a VoxelsOnCartesianGrid
#include "stir/is_null_ptr.h"
#include <boost/format.hpp>
#include "stir/error.h"
#include "stir/warning.h"
#include "stir/info.h"
#include "stir/utilities.h"
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
using std::ifstream;


START_NAMESPACE_STIR

//! Class for reading images in Interfile file-format.
/*! \ingroup ECAT
    \preliminary

*/
bool 
InterfileDynamicDiscretisedDensityInputFileFormat::
actual_can_read(const FileSignature& signature,
		std::istream& input) const
{
      //. todo should check if it's an image
    return is_interfile_signature(signature.get_signature());
}

std::auto_ptr<InterfileDynamicDiscretisedDensityInputFileFormat::data_type>
InterfileDynamicDiscretisedDensityInputFileFormat::
read_from_file(std::istream& input) const
{
    std::auto_ptr<data_type> ret(this->read_interfile_dyn_image(input));
    if (is_null_ptr(ret))
      {
	error("failed to read an Interfile image from stream");
      }
    return ret;
}

std::auto_ptr<InterfileDynamicDiscretisedDensityInputFileFormat::data_type>
InterfileDynamicDiscretisedDensityInputFileFormat::
read_from_file(const std::string& filename) const
{
  
  const int max_length=300;
  char signature[max_length];

  // read signature
  {
    std::ifstream input(filename.c_str(), std::ios::binary);
    if (!input)
      {
       error("InterfileDynamicDiscretisedDensityInputFileFormat::read_from_file: error opening file '%s'. Does it exist?", filename.c_str());
	   return std::auto_ptr<data_type>(0);
      }
    input.read(signature, max_length);
    signature[max_length-1]='\0';
  }

  // Interfile
  if (is_interfile_signature(signature))
  {
#ifndef NDEBUG
    info(boost::format("InterfileDynamicDiscretisedDensityInputFileFormat::read_from_file trying to read %s as Interfile") % filename);
#endif

    std::auto_ptr<data_type> dyn_ptr(this->read_interfile_dyn_image(filename, std::ios::in));
	
    if (!is_null_ptr(dyn_ptr))
      return dyn_ptr;
    else
	  {
       error(boost::format("InterfileDynamicDiscretisedDensityInputFileFormat::read_from_file failed to read %s as Interfile") % filename);
	   return std::auto_ptr<data_type>(0);
	  }
  }
  else
    {
	 error(boost::format("InterfileDynamicDiscretisedDensityInputFileFormat::read_from_file could not identify an Interfile signature for the file: %s \n") % filename);
	 return std::auto_ptr<data_type>(0);
	}
}

// local helper function to read concatenated dynamic image data from Interfile
// It assumes that all images have the same dimensions and byte sizes (which is normally the case)
// It is based on read_interfile_DPDFS (for concatenated Interfile projection data)
std::auto_ptr<InterfileDynamicDiscretisedDensityInputFileFormat::data_type>
InterfileDynamicDiscretisedDensityInputFileFormat::
read_interfile_dyn_image(std::istream& input,
		    const string& directory_for_data,
		     const std::ios::openmode open_mode) const
{
  
  info(boost::format("InterfileDynamicDiscretisedDensityInputFileFormat:: Reading the Interfile image set located in directory\n%1%  ...\n") % directory_for_data.c_str());
  
  InterfileImageHeader hdr;
   
   if (!hdr.parse(input))
    {
      error("InterfileDynamicDiscretisedDensityInputFileFormat::Failed to properly parse the Interfile header file of the provided data stream");
	  return std::auto_ptr<data_type>(0);
    }

  char full_data_file_name[max_filename_length];
  strcpy(full_data_file_name, hdr.data_file_name.c_str());
  prepend_directory_name(full_data_file_name, directory_for_data.c_str());  
 
  for (unsigned int i=1; i<hdr.image_scaling_factors[0].size(); i++)
    if (hdr.image_scaling_factors[0][0] != hdr.image_scaling_factors[0][i])
      { 
	   warning("Interfile warning: all image scaling factors should be equal \n"
		"at the moment. Using the first scale factor only.\n");
	   break;
      }
   
   //shared_ptr<iostream> data_in(new std::fstream (full_data_file_name, open_mode | std::ios::binary));
 
   const int z_size =  hdr.matrix_size[2][0];
   const int y_size =  hdr.matrix_size[1][0];
   const int x_size =  hdr.matrix_size[0][0];
   const int image_size_in_pixels = x_size*y_size*z_size;
   
   unsigned long data_offset = hdr.data_offset_each_dataset[0];
   // offset in file between time frames or parametric images
   unsigned long data_offset_increment = image_size_in_pixels*hdr.type_of_numbers.size_in_bytes();
	 
   std::auto_ptr<data_type> dyn_image_ptr(new DynamicDiscretisedDensity);

   if (is_null_ptr(dyn_image_ptr))
     {
       error(boost::format("InterfileDynamicDiscretisedDensityInputFileFormat: error allocating memory for new object (for file \"%1%\")")
		    % full_data_file_name);
	   return std::auto_ptr<data_type>(0);
     }

   //dyn_image_ptr->set_time_frame_definitions(hdr.time_frame_definitions);
   dyn_image_ptr->resize_densities(hdr.time_frame_definitions);
   unsigned int nframes = dyn_image_ptr->get_num_time_frames();
   info(boost::format("InterfileDynamicDiscretisedDensityInputFileFormat:: %1% frames were identified in the input image set\n") % nframes);

   // TODO set start_time*UTC

   for (unsigned int frame_num=1; frame_num <= nframes; ++frame_num)
        {
		
		 info(boost::format("InterfileDynamicDiscretisedDensityInputFileFormat::read_interfile_dyn_image:\nUsing data offset %1% for frame %2% in file %3%\n\n") % data_offset % frame_num % full_data_file_name);
      
	     shared_ptr<DynamicDiscretisedDensity::singleDiscDensT> 
	      dens_sptr(read_interfile_frame_image(input,
										       data_offset,
											   hdr,
	                                           directory_for_data)
		  );
		
	     if (is_null_ptr(dens_sptr))
		  {
	       error("InterfileDynamicDiscretisedDensityInputFileFormat::read_interfile_dyn_image: No frame %d available", frame_num);
		   return std::auto_ptr<data_type>(0);
		  }

		 
         dyn_image_ptr->set_density_sptr(dens_sptr, frame_num );
         
		 std::cout << "The setting of the buffer of image " << frame_num << " to its proper position in the dynamic buffer was successful\n";
		 
	     // find offset of next frame
	     if (frame_num < nframes)
	      {
	       // note that hdr.data_offset_each_dataset uses zero-based indexing, so next line finds the offset for frame frame_num+1
	       if (hdr.data_offset_each_dataset[frame_num]>0)
		    {
		     if (fabs(static_cast<double>(data_offset) - hdr.data_offset_each_dataset[frame_num]) < data_offset_increment)
		      {
		       error(boost::format("data offset for frame %1% is too small. Difference in offsets needs to be at least %2%")
			    % (frame_num+1) % data_offset_increment);
			   return std::auto_ptr<data_type>(0);
		      }
		     data_offset = hdr.data_offset_each_dataset[frame_num]; //in that case the current offset is provided by the header file and may change between different frames/images
		    }
	       else
		    data_offset += data_offset_increment; //in that case (when current offset is <=0) we assume increment of offset is the same for all frames/images
	      }
		  std::cout << "The data offset was successfully updated to: " << data_offset << "\n\n";
	    } // end loop over frames/images
   return dyn_image_ptr;
}

// local helper function to read concatenated dynamic image data from Interfile
std::auto_ptr<InterfileDynamicDiscretisedDensityInputFileFormat::data_type>
InterfileDynamicDiscretisedDensityInputFileFormat::
read_interfile_dyn_image(const string& filename,
		     const std::ios::openmode open_mode) const
{
  std::ifstream image_stream(filename.c_str(), open_mode);
  if (!image_stream)
    { 
      error(boost::format("DynamicProjData: couldn't open file '%s'") % filename);
    }
  
  return read_interfile_dyn_image(image_stream, get_directory_name(filename.c_str()), open_mode);
}

END_NAMESPACE_STIR