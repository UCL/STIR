//
// $Id$
//

/*! 
  \file
  \ingroup utilities
  \brief Conversion from ECAT 6 cti to interfile (image and sinogram data)
  \author Kris Thielemans
  \author Damien Sauge
  \author Sanida Mustafovic
  \author PARAPET project
  $Revision$
  $Date$

  \warning Most of the data in the ECAT 6 headers is ignored (except dimensions)
  \warning Data are scaled using the subheader.scale_factor * subheader.loss_correction_fctr
  (unless the loss correction factor is < 0, in which case it is assumed to be 1).
  \warning Currently, the decay correction factor is ignored.
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/utilities.h"
#include "stir/interfile.h"
#include "stir/shared_ptr.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/CTI/stir_cti.h"
#include "stir/CTI/cti_utils.h"
#include <stdio.h>
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif



USING_NAMESPACE_STIR

int
main(int argc, char *argv[])
{
    char cti_name[max_filename_length], out_name[max_filename_length];
    FILE *cti_fptr;
 
    if(argc==3)
      { 
	strcpy(cti_name, argv[2]);
	strcpy(out_name, argv[1]);
      }
    else 
    {
        cerr<<"\nConversion from ECAT6 CTI data to interfile.\n";
        cerr<<"Usage: convecat6_if [output_file_name cti_data_file_name]\n"<<endl;
	if (argc!=1)
	  exit(EXIT_FAILURE);

	ask_filename_with_extension(out_name,"Name of the output file? (.hv/.hs and .v/.s will be added)","");    
        ask_filename_with_extension(cti_name,"Name of the input data file? ",".scn");
        
    }


    // open input file, read main header
    cti_fptr=fopen(cti_name, "rb"); 
    if(!cti_fptr) {
        error("\nError opening input file: %s\n",cti_name);
    }
    Main_header mhead;
    if(cti_read_main_header(cti_fptr, &mhead)!=EXIT_SUCCESS) {
        error("\nUnable to read main header in file: %s\n",cti_name);
    }
#ifndef STIR_NO_NAMESPACES
             // VC needs this std::
  const int num_frames = std::max(static_cast<int>( mhead.num_frames),1);
  const int num_bed_poss = static_cast<int>( mhead.num_bed_pos);
  const int num_gates = std::max(static_cast<int>( mhead.num_gates),1);
#else
  const int num_frames = max(static_cast<int>( mhead.num_frames),1);
  const int num_bed_poss = static_cast<int>( mhead.num_bed_pos);
  const int num_gates = max(static_cast<int>( mhead.num_gates),1);
#endif


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
    }
  else
    {
      do_all = false;
      min_frame_num= max_frame_num=
	ask_num("Frame number ? ", 1, num_frames,1);
      min_bed_num= max_bed_num=
	ask_num("Bed number ? ", 0,num_bed_poss, 0);
      min_gate_num= max_gate_num=
	ask_num("Gate number ? ", 1, num_gates, 1);
      data_num=
	ask_num("Data number ? ",0,8, 0);
    }
        
  switch(mhead.file_type)
    { 
    case matImageFile:
      {
	char *new_out_filename = new char[strlen(out_name)+100];
	for (int frame_num=min_frame_num; frame_num<=max_frame_num;++frame_num)
	  for (int bed_num=min_bed_num; bed_num<=max_bed_num;++bed_num)
	    for (int gate_num=min_gate_num; gate_num<=max_gate_num;++gate_num)
	      {
		strcpy(new_out_filename, out_name);
		if (do_all)
		  sprintf(new_out_filename+strlen(new_out_filename), "_f%dg%db%dd%d", 
			  frame_num, gate_num, bed_num, data_num);
		cerr << "Writing " << new_out_filename << endl;
		shared_ptr<VoxelsOnCartesianGrid<float> > image_ptr =
		  ECAT6_to_VoxelsOnCartesianGrid(frame_num, gate_num, data_num, bed_num,
						 cti_fptr, mhead);
		write_basic_interfile(new_out_filename,*image_ptr);
	      }
	delete new_out_filename;
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

	char *new_out_filename = new char[strlen(out_name)+100];
	for (int frame_num=min_frame_num; frame_num<=max_frame_num;++frame_num)
	  for (int bed_num=min_bed_num; bed_num<=max_bed_num;++bed_num)
	    for (int gate_num=min_gate_num; gate_num<=max_gate_num;++gate_num)
	      {
		strcpy(new_out_filename, out_name);
		if (do_all)
		  sprintf(new_out_filename+strlen(new_out_filename), "_f%dg%db%dd%d", 
			  frame_num, gate_num, bed_num, data_num);
		cerr << "Writing " << new_out_filename << endl;
		ECAT6_to_PDFS(frame_num, gate_num, data_num, bed_num,
			      max_ring_diff, arccorrected,
			      new_out_filename, cti_fptr, mhead);
	      }
	delete new_out_filename;
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
