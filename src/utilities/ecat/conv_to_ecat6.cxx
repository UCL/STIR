//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
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
\brief Conversion from interfile (or any format that we can read) 
  to ECAT 6 cti (image and sinogram data)
\author Kris Thielemans
\author PARAPET project

This programme is used to convert image or projection data into CTI ECAT 6 data (input 
can be any format currently supported by the library). It normally should be run as 
follows
<pre>conv_to_ecat6 [-k] [-i]  outputfilename.img input_filename1 [input_filename2 ...] scanner_name
</pre>
(for images)
<pre>conv_to_ecat6 -s[2] [-k] [-i] outputfilename.scn input_filename1 [input_filename2 ...]
</pre>
(for projection data)<br>
If there are no command line parameters, the user is asked for the filenames and options 
instead. Unless the -i option is used, the data will be assigned a frame number in the 
order that they occur on the command line.<br>
See buildblock/Scanner.cxx for supported scanner names, but examples are ECAT 953, 
ART, Advance. ECAT HR+, etc. If the scanner_name contains a space, the scanner name has to 
be surrounded by double quotes (&quot;) when used as a command line argument.

\par Command line options:
<ul>
<li>	-s2: This option forces output to 2D sinograms (ignoring higher segments).</li>
<li>	-k: the existing ECAT6 file will NOT be overwritten, but added to. Any existing 
data in the ECAT6 file with the same &lt;frame,gate,data,bed&gt; specification will be 
overwritten.</li>
<li>	-i: ask for &lt;frame,gate,data,bed&gt; for each dataset</li>
</ul>
Note that to store projection data in ECAT6, a 3D sinogram cannot be axially compressed 
(CTI span=1).

*/


#include "stir/DiscretisedDensity.h"
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"
#include "stir/utilities.h"
#include "stir/IO/stir_ecat6.h"
#include "stir/IO/ecat6_utils.h"
#include "stir/IO/read_from_file.h"
#include "stir/Succeeded.h"
#include <iostream>
#include <vector>
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::size_t;
#endif

USING_NAMESPACE_STIR
USING_NAMESPACE_ECAT
USING_NAMESPACE_ECAT6


int main(int argc, char *argv[])
{
  char cti_name[1000], scanner_name[1000] = "";
  vector<string> filenames;
  bool its_an_image = true;
  bool its_a_2D_sinogram = false;
  bool add_to_existing = false;
  bool interactive = false;
  
  if(argc>=4)
  {
    if (strncmp(argv[1],"-s",2)==0)
      {
	its_an_image = false;
	its_a_2D_sinogram = strlen(argv[1])==3 && argv[1][2]=='2';
	if (its_a_2D_sinogram) 
	  cout << "I will write 2D sinograms\n";

	while (argv[2][0] == '-')
	  {
	    if (strcmp(argv[2],"-k")==0)
	      add_to_existing = true;	    
	    else if (strcmp(argv[2],"-i")==0)
	      interactive = true;
	    else
	      warning("Ignored unrecognised option: %s.", argv[2]);
	  
	    argv++; argc--;
	  }
	strcpy(cti_name,argv[2]);
        int num_files = argc-3;
        argv+=3;
        filenames.reserve(num_files);
        for (; num_files>0; --num_files, ++argv)
          filenames.push_back(*argv);	
      }
    else 
      {
	its_an_image = true;
	strcpy(cti_name,argv[1]);	
        int num_files = argc-3;
        argv+=2;
        filenames.reserve(num_files);
        for (; num_files>0; --num_files, ++argv)
          filenames.push_back(*argv);	
	strcpy(scanner_name,*argv);
      }
  }  
  else 
  {
    cerr<< "\nConversion from data to ECAT6 CTI.\n"
	<< "Multiples files can be written to a single ECAT 6 file.\n"
        << "Unless the -i option is used, the data will be assigned a frame number in \n"
        << "the order that they occur on the command line.\n\n"
        << "Usage: 2 possible forms depending on data type\n"
	<< "For sinogram data:\n"
	<< "\tconv_to_ecat6 -s[2] [-k] [-i] output_ECAT6_name orig_filename1 [orig_filename2 ...]\n"
	<< "\tThe -s2 option forces output to 2D sinograms (ignoring higher segments).\n"
	<< "For image data:\n"
	<< "\tconv_to_ecat6 [-k] [-i] output_ECAT6_name orig_filename1 [orig_filename2 ...] scanner_name\n"
	<< "scanner_name has to be recognised by the Scanner class\n"
	<< "Examples are : \"ECAT 953\", \"RPT\" etc.\n"
	<< "(the quotes are required when used as a command line argument)\n\n"
	<< "Options:\n"
	<< "  -k: the existing ECAT6 file will NOT be overwritten,\n"
	<< "\tbut added to. Any existing data in the ECAT6 file with the same <frame,gate,data,bed>\n"
	<< "\tspecification will be overwritten.\n"
	<< "  -i: ask for <frame,gate,data,bed> for each dataset\n\n"
	<< "I will now ask you the same info interactively...\n\n";
    
    its_an_image = ask("Converting images?",true);
    if (!its_an_image)
      its_a_2D_sinogram = ask("Write as 2D sinogram?",false);

    int num_files = ask_num("Number of files",1,10000,1);
    filenames.reserve(num_files);
    char cur_name[max_filename_length];
    for (; num_files>0; --num_files)
    {
      ask_filename_with_extension(cur_name,"Name of the input file? ",its_an_image?".hv":".hs");
      filenames.push_back(cur_name);
    }
    
    ask_filename_with_extension(cti_name,"Name of the ECAT6 file? ",
      its_an_image ? ".img" : ".scn");
  }

  size_t num_frames, num_gates, num_bed_poss, num_data;
  if (interactive)
    {
      num_frames = ask_num("Num frames?",static_cast<size_t>(1),filenames.size(), filenames.size());
      num_gates = ask_num("Num gates?",static_cast<size_t>(1),filenames.size()/num_frames, filenames.size()/num_frames);
      num_bed_poss = ask_num("Num bed positions?",static_cast<size_t>(1),filenames.size(), filenames.size());
      num_data = ask_num("Num data?",static_cast<size_t>(1),filenames.size()/num_frames, filenames.size()/num_frames);
    }
  else
    {
      num_frames = filenames.size();
      num_gates=1;
      num_bed_poss=1;
      num_data=1;
    }
  size_t min_frame_num = 1;
  size_t max_frame_num = num_frames;
  size_t min_bed_num = 0;
  size_t max_bed_num = num_bed_poss-1; 
  size_t min_gate_num = 1;
  size_t max_gate_num = num_gates;
  size_t min_data_num = 0;

  if (its_an_image)
  {

    shared_ptr<Scanner> scanner_ptr(
      strlen(scanner_name)==0 ?
      Scanner::ask_parameters() :
      Scanner::get_scanner_from_name(scanner_name));

    // read first image
    cerr << "Reading " << filenames[0] << endl;
    shared_ptr<DiscretisedDensity<3,float> > 
      density_ptr(read_from_file<DiscretisedDensity<3,float> >(filenames[0]));
  
    ECAT6_Main_header mhead;
    make_ECAT6_Main_header(mhead, *scanner_ptr, filenames[0], *density_ptr);
    mhead.num_frames = filenames.size();

    FILE *fptr= cti_create (cti_name, &mhead);
    if (fptr == NULL)
    {
      warning("conv_to_ecat6: error opening output file %s\n", cti_name);
      return EXIT_FAILURE;
    }
    size_t frame_num = 1;

    while (1)
    {
      if (DiscretisedDensity_to_ECAT6(fptr,
                                      *density_ptr, 
                                      mhead,
                                      frame_num)
                                      == Succeeded::no)
      {
        fclose(fptr);
        return EXIT_FAILURE;
      }
      if (++frame_num > filenames.size())
      {
        fclose(fptr);
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
  
    ECAT6_Main_header mhead;
    FILE *fptr;

    if (add_to_existing)
      {
	fptr = fopen(cti_name,"wb+");
	if (!fptr)
	  error("Error opening cti file %s\n", cti_name);
	if (cti_read_ECAT6_Main_header (fptr, &mhead) == EXIT_FAILURE)
	  error("Error reading main header from cti file %s\n", cti_name);
      }
    else
      {
	make_ECAT6_Main_header(mhead, filenames[0], *proj_data_ptr->get_proj_data_info_ptr());
	mhead.num_frames = num_frames;
	mhead.num_gates = num_gates;
	mhead.num_bed_pos = num_bed_poss-1;
	//mhead.num_data = num_data;
	
	fptr = cti_create (cti_name, &mhead);
	if (fptr == NULL)
	  {
	    warning("conv_to_ecat6: error opening output file %s\n", cti_name);
	    return EXIT_FAILURE;
	  }
      }
    size_t frame_num = 1;

    while (1)
    {
      size_t current_frame_num, current_bed_num, current_gate_num, current_data_num;
      if (interactive)
	{
	  current_frame_num= 
	    ask_num("Frame number ? ", min_frame_num, max_frame_num, min_frame_num);
	  current_bed_num= 
	    ask_num("Bed number ? ", min_bed_num, max_bed_num, min_bed_num);
	  current_gate_num=
	    ask_num("Gate number ? ", min_gate_num, max_gate_num, min_gate_num);
	  current_data_num=
	    ask_num("Data number ? ",0,7, 0);
	}
      else
	{
	  current_frame_num = frame_num;
	  current_bed_num = min_bed_num;
	  current_gate_num = min_gate_num;
	  current_data_num = min_data_num;
	}
      if (ProjData_to_ECAT6(fptr,
                            *proj_data_ptr, 
                            mhead,
                            current_frame_num, current_gate_num, current_data_num, current_bed_num,
			    its_a_2D_sinogram)
                            == Succeeded::no)
      {
        fclose(fptr);
        return EXIT_FAILURE;
      }
      if (++frame_num > filenames.size())
      {
        fclose(fptr);
        return EXIT_SUCCESS;
      }
      cerr << "Reading " << filenames[frame_num-1] << endl;
      proj_data_ptr =
        ProjData::read_from_file(filenames[frame_num-1]);
    }
  }  
}


