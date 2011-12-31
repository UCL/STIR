//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2009, Hammersmith Imanet Ltd
    Copyright 2011 - $Date$, Kris Thielemans
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
  \ingroup ECAT_utilities
  \brief Conversion from ECAT 6 cti to interfile (image and sinogram data)
  \author Kris Thielemans
  \author Damien Sauge
  \author Sanida Mustafovic
  \author PARAPET project
  $Revision$
  $Date$

  \par Usage: 
  <pre>convecat6_if [output_file_name_without_extension cti_data_file_name [scanner_name]]
  </pre>
  The optional \a scanner_name can be used to force to a particular scanner
  (ignoring the system_type in the main header).
  \a scanner_name has to be recognised by the Scanner class.
  Examples are : ECAT 953, RPT, ECAT HR+, Advance etc. If the \a scanner_name 
  contains a space, the scanner name has to be surrounded by double quotes 
  (&quot;) when used as a command line argument.
  <br>
  The program asks if all frames should be written or not. If so, all 
  sinograms/images are converted for a fixed 'data' number. For each data set,
  a suffix is added to the output_filename of the form "_f#g#b#d#" where the # 
  are replaced by the corresponding number of the frame, gate, bed, data.

  \warning CTI ECAT files seem to have a peculiarity that frames and gates are 
  numbered from 1, while bed positions are numbered from 0. Similarly, the number
  of bed positions in the main header seems to be 1 less than the actual number
  present. This is at least the case for single bed studies. If this is not true
  for multi-bed studies, the code would have to be adapted.
  \warning Most of the data in the ECAT 6 headers is ignored (except dimensions)
  \warning Data are multiplied with the subheader.scale_factor, In addition, for
  emission sinograms, the data are multiplied with subheader.loss_correction_fctr
  (unless the loss correction factor is < 0, in which case it is assumed to be 1).
  \warning Currently, the decay correction factor is ignored.

  \todo This could easily be used to convert to other file formats. For images,
  this simply involves changing the OutputFileFormat. For projection data,
  we would have to extend OutputFileFormat to handle projection data.
*/


#include "stir/utilities.h"
#include "stir/IO/InterfileOutputFileFormat.h"
#include "stir/shared_ptr.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/stir_ecat6.h"
#include "stir/IO/ecat6_utils.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::cout;
using std::endl;
using std::min;
using std::max;
#endif



USING_NAMESPACE_STIR
USING_NAMESPACE_ECAT
USING_NAMESPACE_ECAT6

int
main(int argc, char *argv[])
{
  std::string cti_name;
  std::string out_name;
  char * scanner_name_ptr = 0;
  FILE *cti_fptr;
 
  if(argc==3 || argc==4)
    { 
      cti_name = argv[2];
      out_name = argv[1];
      if (argc>3)
        scanner_name_ptr = argv[3];
    }
  else 
    {
      cerr<<"\nConversion from ECAT6 CTI data to interfile.\n";
      cerr<<"Usage: convecat6_if [output_file_name_without_extension cti_data_file_name [scanner_name]]\n"
          <<"The optional scanner_name can be used to force to a particular scanner"
          <<" (ignoring the system_type in the main header).\n"
          << "scanner_name has to be recognised by the Scanner class\n"
	  << "Examples are : \"ECAT 953\", \"RPT\" etc.\n"
	  << "(the quotes are required when used as a command line argument)\n"
          << endl;
	
      if (argc!=1)
	exit(EXIT_FAILURE);

      out_name = ask_filename_with_extension("Name of the output file? (.hv/.hs and .v/.s will be added)","");    
      cti_name = ask_filename_with_extension("Name of the input data file? ",".scn");
        
    }


  // open input file, read main header
  cti_fptr=fopen(cti_name.c_str(), "rb"); 
  if(!cti_fptr) {
    error("Error opening input file: %s",cti_name.c_str());
  }
  ECAT6_Main_header mhead;
  if(cti_read_ECAT6_Main_header(cti_fptr, &mhead)!=EXIT_SUCCESS) {
    error("Unable to read main header in file: %s",cti_name.c_str());
  }

  if (scanner_name_ptr != 0)
  {
    // force scanner
    shared_ptr<Scanner> scanner_ptr(
				    Scanner::get_scanner_from_name(scanner_name_ptr));
    mhead.system_type = find_ECAT_system_type(*scanner_ptr);
  }

  // funnily enough, num_bed_pos seems to be offset with 1
  // (That's to say, in a singled bed study, num_bed_pos==0) 
  // TODO maybe not true for multi-bed studies
  const int num_frames = max(static_cast<int>( mhead.num_frames),1);
  const int num_bed_poss = max(static_cast<int>( mhead.num_bed_pos) + 1,1);
  const int num_gates = max(static_cast<int>( mhead.num_gates),1);


  int min_frame_num = 1;
  int max_frame_num = num_frames;
  int min_bed_num = 0;
  int max_bed_num = num_bed_poss-1; 
  int min_gate_num = 1;
  int max_gate_num = num_gates;
  int data_num = 0;
  bool do_all = true;
  
  if (ask("Attempt all data-sets (Y) or single data-set (N)", true))
    {
      data_num=ask_num("Data number ? ",0,8, 0);

      cout << "Processing frames " << min_frame_num << '-' << max_frame_num
	   << ", gates " <<  min_gate_num << '-' << max_gate_num
	   << ", bed positions " << min_bed_num << '-' << max_bed_num
	   << endl;
    }
  else
    {
      do_all = false;
      min_frame_num= max_frame_num=
	ask_num("Frame number ? ", min_frame_num, max_frame_num, min_frame_num);
      min_bed_num= max_bed_num=
	ask_num("Bed number ? ", min_bed_num, max_bed_num, min_bed_num);
      min_gate_num= max_gate_num=
	ask_num("Gate number ? ", min_gate_num, max_gate_num, min_gate_num);
      data_num=
	ask_num("Data number ? ",0,7, 0);
    }
        
  switch(mhead.file_type)
    { 
    case matImageFile:
      {
	char *new_out_filename = new char[out_name.size()+100];
	for (int frame_num=min_frame_num; frame_num<=max_frame_num;++frame_num)
	  for (int bed_num=min_bed_num; bed_num<=max_bed_num;++bed_num)
	    for (int gate_num=min_gate_num; gate_num<=max_gate_num;++gate_num)
	      {
		strcpy(new_out_filename, out_name.c_str());
		if (do_all)
		  sprintf(new_out_filename+strlen(new_out_filename), "_f%dg%db%dd%d", 
			  frame_num, gate_num, bed_num, data_num);
		cout << "Writing " << new_out_filename << endl;
		shared_ptr<VoxelsOnCartesianGrid<float> > image_ptr(
		  ECAT6_to_VoxelsOnCartesianGrid(frame_num, gate_num, data_num, bed_num,
						 cti_fptr, mhead));
		InterfileOutputFileFormat output_file_format;
		output_file_format.write_to_file(new_out_filename,*image_ptr);
	      }
	delete[] new_out_filename;
        break;
      }
    case matScanFile:
    case matAttenFile:
    case matNormFile:
      {            
        const int max_ring_diff= 
	  ask_num("Max ring diff to store (-1 == num_rings-1)",-1,100,-1);

        const bool arccorrected = 
	  ask("Consider the data to be arc-corrected?",false);

	char *new_out_filename = new char[out_name.size()+100];
	for (int frame_num=min_frame_num; frame_num<=max_frame_num;++frame_num)
	  for (int bed_num=min_bed_num; bed_num<=max_bed_num;++bed_num)
	    for (int gate_num=min_gate_num; gate_num<=max_gate_num;++gate_num)
	      {
		strcpy(new_out_filename, out_name.c_str());
		if (do_all)
		  sprintf(new_out_filename+strlen(new_out_filename), "_f%dg%db%dd%d", 
			  frame_num, gate_num, bed_num, data_num);
		cout << "Writing " << new_out_filename << endl;
		ECAT6_to_PDFS(frame_num, gate_num, data_num, bed_num,
			      max_ring_diff, arccorrected,
			      new_out_filename, cti_fptr, mhead);
	      }
	delete[] new_out_filename;
        break;
      }
    default:
      {
        error("\nSupporting only image, scan, atten or norm file type at the moment. Sorry.\n");            
      }
    }    
  fclose(cti_fptr);
    
  return EXIT_SUCCESS;
}
