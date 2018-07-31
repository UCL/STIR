//
//
/*
  Copyright (C) 2006-2009, Hammersmith Imanet Ltd
  Copyright (C) 2018, University College London
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
#include <fstream>

START_NAMESPACE_STIR

const char * const 
MultiDynamicDiscretisedDensityOutputFileFormat::registered_name = "Multi";

MultiDynamicDiscretisedDensityOutputFileFormat::
MultiDynamicDiscretisedDensityOutputFileFormat(const NumericType& type, 
					const ByteOrder& byte_order) 
{
  base_type::set_defaults();
  this->set_type_of_numbers(type);
  this->set_byte_order(byte_order);
}

void 
MultiDynamicDiscretisedDensityOutputFileFormat::
set_defaults()
{
  base_type::set_defaults();
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
    // Create all the filenames
    VectorWithOffset<std::string> individual_filenames(1,int(density.get_num_time_frames()));
    for (int i=1; i<=int(density.get_num_time_frames()); i++)
        individual_filenames[i] = filename + "_" + std::to_string(i);
    
    // Write each individual image
    for (int i=1; i<=int(density.get_num_time_frames()); i++) {
        Succeeded success = individual_output_type_sptr->write_to_file(individual_filenames[i],density.get_density(unsigned(i)));
        if (success != Succeeded::yes)
            warning("MultiDynamicDiscretisedDensity error: Failed to write \"" + individual_filenames[i] + "\".\n");
    }
    
    // Write some multi header info
    filename = filename + ".txt";
    std::ofstream multi_file(filename.c_str());
    if (!multi_file.is_open()) {
        warning("MultiDynamicDiscretisedDensity error: Failed to write \"" + filename + "\".\n");
        return Succeeded::no;
    }
    multi_file << "Multi :=\n";
    multi_file << "\ttotal number of data sets := " << density.get_num_time_frames() << "\n";
    for (int i=1; i<=int(density.get_num_time_frames()); i++)
        multi_file << "\tdata set["<<i<<"] := "<<individual_filenames[i]<<"\n";
    multi_file << "end :=\n";
    multi_file.close();

    return Succeeded::yes;
}

// class MultiDynamicDiscretisedDensityOutputFileFormat;  


END_NAMESPACE_STIR
