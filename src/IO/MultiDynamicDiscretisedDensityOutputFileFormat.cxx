//
//
/*
  Copyright (C) 2006-2009, Hammersmith Imanet Ltd
  Copyright (C) 2018,2020 University College London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/

/*!\file
  \ingroup MultiIO
  \brief Implementation of class stir::MultiDynamicDiscretisedDensityOutputFileFormat

\author Kris Thielemans
\author Charalampos Tsoumpas
\author Richard Brown

*/

#include "stir/IO/MultiDynamicDiscretisedDensityOutputFileFormat.h"
#include "stir/DynamicDiscretisedDensity.h" 
#include "stir/NumericType.h"
#include "stir/Succeeded.h"
#include "stir/FilePath.h"
#include "stir/MultipleDataSetHeader.h"

START_NAMESPACE_STIR

const char * const 
MultiDynamicDiscretisedDensityOutputFileFormat::registered_name = "Multi";

MultiDynamicDiscretisedDensityOutputFileFormat::
MultiDynamicDiscretisedDensityOutputFileFormat(const NumericType& type, 
					const ByteOrder& byte_order) 
{
  this->set_defaults();
  this->set_type_of_numbers(type);
  this->set_byte_order(byte_order);
}

void 
MultiDynamicDiscretisedDensityOutputFileFormat::
set_defaults()
{
  base_type::set_defaults();
  this->set_type_of_numbers(NumericType::FLOAT);
  this->set_byte_order(ByteOrder::native);
  this->individual_output_type_sptr = OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr();
}

void 
MultiDynamicDiscretisedDensityOutputFileFormat::
initialise_keymap()
{
  this->parser.add_start_key("Multi Output File Format Parameters");
  this->parser.add_stop_key("End Multi Output File Format Parameters");
  this->parser.add_parsing_key("individual output file format type",&individual_output_type_sptr);
  base_type::initialise_keymap();
}

bool 
MultiDynamicDiscretisedDensityOutputFileFormat::
post_processing()
{
  if (base_type::post_processing())
    return true;
  return false;
}


ByteOrder 
MultiDynamicDiscretisedDensityOutputFileFormat::
set_byte_order(const ByteOrder& new_byte_order, const bool warn)
{
  if (!new_byte_order.is_native_order())
    {
      if (warn)
	warning("MultiDynamicDiscretisedDensityOutputFileFormat: byte_order is currently fixed to the native format\n");
      this->file_byte_order = ByteOrder::native;
    }
  else
    this->file_byte_order = new_byte_order;
  return  this->file_byte_order;
}

Succeeded  
MultiDynamicDiscretisedDensityOutputFileFormat::
actual_write_to_file(std::string& filename, 
		     const DynamicDiscretisedDensity & density) const
{
    {
      FilePath file_path(filename, false); // create object without checking if a fo;e exists already
      if (!file_path.get_extension().empty())
        error("MultiDynamicDiscretisedDensityOutputFileFormat: currently needs an output filename without extension. sorry");
    }
    // Create all the filenames
    VectorWithOffset<std::string> individual_filenames(1,int(density.get_num_time_frames()));
    for (int i=1; i<=int(density.get_num_time_frames()); i++)
        individual_filenames[i] = filename + "_" + boost::lexical_cast<std::string>(i);
    
    // Write each individual image
    for (int i=1; i<=int(density.get_num_time_frames()); i++) {
        Succeeded success = individual_output_type_sptr->write_to_file(individual_filenames[i],density.get_density(unsigned(i)));
        if (success != Succeeded::yes)
            warning("MultiDynamicDiscretisedDensity error: Failed to write \"" + individual_filenames[i] + "\".\n");
    }
    
    // Write some multi header info
    filename = filename + ".txt";
    MultipleDataSetHeader::write_header(filename, individual_filenames);
    return Succeeded::yes;
}

// class MultiDynamicDiscretisedDensityOutputFileFormat;  


END_NAMESPACE_STIR
