/*! 
\file
\ingroup ECAT_utilities
\brief Conversion from interfile (or any format that we can read) 
  to ECAT 7 cti (image and sinogram data)
\author Kris Thielemans
\author PARAPET project


This programme is used to convert image or projection data into CTI ECAT 7 data (input 
can be any format currently supported by the library). It normally should be run as 
follows
<pre>conv_to_ecat7 output_ECAT7_name input_filename1 [input_filename2 ...] scanner_name
</pre>
(for images)
<pre>conv_to_ecat7 -s output_ECAT7_name  input_filename1 [input_filename2 ...]
</pre>
(for emission projection data)
<pre>conv_to_ecat7 -a output_ECAT7_name  input_filename1 [input_filename2 ...]
</pre>
(for sinogram-attenuation data)<br>
If there are no command line parameters, the user is asked for the filenames and options 
instead. The data will be assigned a frame number in the 
order that they occur on the command line.<br>
See buildblock/Scanner.cxx for supported scanner names, but examples are ECAT 953, 
ART, Advance. ECAT HR+, etc. If the scanner_name contains a space, the scanner name has to 
be surrounded by double quotes (&quot;) when used as a command line argument.
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2009, Hammersmith Imanet Ltd
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

#include "stir/DiscretisedDensity.h"
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/IO/read_from_file.h"
#include <iostream>
#include <vector>
#include <string>
#include <boost/format.hpp>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::vector;
using std::string;
#endif

USING_NAMESPACE_STIR
USING_NAMESPACE_ECAT
USING_NAMESPACE_ECAT7




void usage() {

    cerr << "\nConversion from data to ECAT7 CTI.\n"
         << "Multiples files can be written to a single ECAT 7 file.\n"
         << "The data will be assigned a frame number in the "
         << "order that they occur on the command line.\n\n"
         << "Usage: 3 possible forms depending on data type\n"
         << "For sinogram data:\n"
         << "\tconv_to_ecat7 -s [-n] output_ECAT7_name orig_filename1 [orig_filename2 ...]\n"
         << "For sinogram-attenuation data:\n"
         << "\tconv_to_ecat7 -a [-n] output_ECAT7_name orig_filename1 [orig_filename2 ...]\n"
         << "For image data:\n"
         << "\tconv_to_ecat7 output_ECAT7_name orig_filename1 [orig_filename2 ...] scanner_name\n"
         << "scanner_name has to be recognised by the Scanner class\n"
         << "Examples are : \"ECAT 953\", \"RPT\" etc.\n"
         << "(the quotes are required when used as a command line argument)\n\n";
}






int main(int argc, char *argv[])
{
  char cti_name[1000], scanner_name[1000] = "";
  vector<string> filenames;
  bool its_an_image = true;
  bool write_as_attenuation = false;
  float scale_factor = 0.0;
  
   
  int arg_index = 1;
    
  /* Check options - single letters only */
  while ( arg_index < argc && argv[arg_index][0] == '-' ) {
    
    int i = 1;
    char c;
    
    while ( (c = argv[arg_index][i]) != '\0' ) {
      
      switch ( c ) {
        
      case 's':
        its_an_image = false;
        break;
        
      case 'a':
        its_an_image = false;
        write_as_attenuation = true;
        break;
        
      case 'n':
        scale_factor = 1.0F;
        break;
        
      default:
        cerr << "Error: Unknown option " << c << " \n\n";
        usage();
        exit(0);
        break;
        
      }
        
      i++;
    }
    
    arg_index++;
  }
    

  

  /* Check number of remaining arguments */
  if ( (its_an_image == false && argc - arg_index >= 1) || argc - arg_index >= 2) {
    
    // Warn about scaling option on it's own.
    if ( its_an_image == true && scale_factor != 0.0F ) {
      cerr << "Warning: option -n has no effect when converting images.\n\n";
    }
    
    
    /* Parse remaining arguments */
    strcpy(cti_name, argv[arg_index]);
    arg_index++;
    

    int num_files;

    if ( its_an_image ) {
      
      for (num_files = argc - arg_index - 1; num_files > 0; --num_files, arg_index++) {
        filenames.push_back(argv[arg_index]);
      }
      
      strcpy(scanner_name, argv[arg_index]);
      
    } else {
      
      for (num_files = argc - arg_index; num_files>0; --num_files, arg_index++) {
        filenames.push_back(argv[arg_index]);
      }
    }

  } else {
    
    usage();

    cerr << "I will now ask you the same info interactively...\n\n";
    
    its_an_image = ask("Converting images?",true);
    
    if (!its_an_image) {
      write_as_attenuation = ask("Write as attenuation data?",false);
    } else {
      if ( ask("Fix scale factor to 1.0?", false) ) {
        scale_factor = 1.0F;
      }
    }

    int num_files = ask_num("Number of files",1,10000,1);
    filenames.reserve(num_files);
    char cur_name[max_filename_length];
    for (; num_files>0; --num_files)
    {
      ask_filename_with_extension(cur_name,"Name of the input file? ",its_an_image?".hv":".hs");
      filenames.push_back(cur_name);
    }
    
    ask_filename_with_extension(cti_name,"Name of the ECAT7 file? ",
                                its_an_image ? ".img" : ".scn");
    
  }
    


  if (its_an_image)
  {

    shared_ptr<Scanner> scanner_ptr(
      strlen(scanner_name)==0 ?
      Scanner::ask_parameters() :
      Scanner::get_scanner_from_name(scanner_name));

    // read first image
    cerr << "Reading " << filenames[0] << endl;
    shared_ptr<DiscretisedDensity<3,float> > density_ptr(
							 read_from_file<DiscretisedDensity<3,float> >(filenames[0]));
  
    Main_header mhead;
    make_ECAT7_main_header(mhead, *scanner_ptr, filenames[0], *density_ptr);
    mhead.num_frames = filenames.size();
    mhead.acquisition_type =
      mhead.num_frames>1 ? DynamicEmission : StaticEmission;

    MatrixFile* mptr= matrix_create (cti_name, MAT_CREATE, &mhead);
    if (mptr == 0)
    {
      warning(boost::format("conv_to_ecat7: error opening output file %s. Remove first if it exists already") % cti_name);
      return EXIT_FAILURE;
    }
    unsigned int frame_num = 1;

    while (1)
    {
      if (DiscretisedDensity_to_ECAT7(mptr,
                                      *density_ptr, 
                                      frame_num)
                                      == Succeeded::no)
      {
        matrix_close(mptr);
        return EXIT_FAILURE;
      }
      if (++frame_num > filenames.size())
      {
        matrix_close(mptr);
        return EXIT_SUCCESS;
      }
      cerr << "Reading " << filenames[frame_num-1] << endl;
      density_ptr =
        read_from_file<DiscretisedDensity<3,float> >(filenames[frame_num-1]);
    }
  }
  else 
  {
 
    // read first data set
    cerr << "Reading " << filenames[0] << endl;
    shared_ptr<ProjData > proj_data_ptr =
      ProjData::read_from_file(filenames[0]);
  
    Main_header mhead;
    // TODO exam_info currently used from the first frame, which means that time frame info is incorrect
    // better to use DynamicProjData etc.
    make_ECAT7_main_header(mhead, filenames[0], 
			   *proj_data_ptr->get_exam_info_ptr(),
			   *proj_data_ptr->get_proj_data_info_ptr(),
			   write_as_attenuation,
			   NumericType::SHORT);
    // fix time frame info
    mhead.num_frames = filenames.size();
    if (!write_as_attenuation)
      {
	mhead.acquisition_type =
	  mhead.num_frames>1 ? DynamicEmission : StaticEmission;
      }
    MatrixFile* mptr= matrix_create (cti_name, MAT_CREATE, &mhead);
    if (mptr == 0)
    {
      warning(boost::format("conv_to_ecat7: error opening output file %s. Remove first if it exists already") % cti_name);
      return EXIT_FAILURE;
    }

    unsigned int frame_num = 1;

    while (1)
    {
      if (ProjData_to_ECAT7(mptr, *proj_data_ptr, 
                            frame_num, 1, 0, 0, scale_factor) == Succeeded::no)
      {
        matrix_close(mptr);
        return EXIT_FAILURE;
      }
      if (++frame_num > filenames.size())
      {
        matrix_close(mptr);
        return EXIT_SUCCESS;
      }
      cerr << "Reading " << filenames[frame_num-1] << endl;
      proj_data_ptr =
        ProjData::read_from_file(filenames[frame_num-1]);
    }
  }  
}

