// $Id$
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
  \ingroup SimSET 
  \brief  This program converts SimSET 3D sinograms to STIR format

  This program should normally be called from the conv_SimSET_projdata_to_STIR.sh script.
  \warning This program does not read the SimSET header, and is thus terribly unsafe.

  \author Pablo Aguiar
  \author Charalampos Tsoumpas
  \author Kris Thielemans 

  $Date$
  $Revision$
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/Sinogram.h"
#include "stir/Scanner.h"
#include "stir/shared_ptr.h"
#include "stir/IO/read_data.h"
#include "stir/Succeeded.h"
#include <iostream>
#define NUMARG 11


int main(int argc,char **argv)
{
  using namespace stir;

  static char *options[]={
    "argv[1]  SimSET file\n",
    "argv[2]  SimSET file format\n",
    "argv[3]  Angles in SimSET file\n",
    "argv[4]  Bins in SimSET filele\n",
    "argv[5]  Axial slices in SimSET file\n",
    "argv[6]  maximum ring difference to use for writing\n",
    "argv[7]  FOV_radius in cm as given to simset binning module (max_td)\n",
    "argv[8]  range on Z value in cm as given to simset binning module\n",
    "argv[9]  index of 3d-sinogram in file (0-based)\n"
    "argv[10] STIR file name\n"
  };
  if (argc!=NUMARG){
    std::cerr << "\n\nConvert SimSET to STIR\n\n";
    std::cerr << "Not enough arguments !!! ..\n";
    for (int i=1;i<NUMARG;i++) std::cerr << options[i-1];
    exit(EXIT_FAILURE);
  }

  const char * const simset_filename = argv[1];
  const char * const input_data_type = argv[2];
  const int num_views=atoi(argv[3]);
  const int num_tangential_poss=atoi(argv[4]);
  const int num_rings=atoi(argv[5]);
  const int max_ring_difference=atoi(argv[6]);
  const float FOV_radius = atof(argv[7])*10; // times 10 for mm
  const float scanner_length = atof(argv[8])*10; // times 10 for mm
  const int dataset_num = atoi(argv[9]);
  const char * const stir_filename = argv[10];
  const int nitems=num_views*num_tangential_poss;

  if (num_tangential_poss%2 != 1)
    warning("STIR can at present not handle simset data with an even "
	    "number of tangential positions.\n"
	    "Proceed at your own risk (but you will get artifacts in the images");
  FILE *file;
  if( (file=fopen(simset_filename,"rb")) ==NULL){
    error("Cannot open the simset file %s", simset_filename);
  }
  shared_ptr<Scanner> scanner_sptr;
  {
    scanner_sptr = new Scanner(Scanner::E966);
    const float this_scanner_length = 
      scanner_sptr->get_num_rings() * scanner_sptr->get_ring_spacing();
    if (fabs(this_scanner_length - scanner_length)>1.0)
      {
	scanner_sptr = new Scanner(Scanner::E962);
	const float this_scanner_length = 
	  scanner_sptr->get_num_rings() * scanner_sptr->get_ring_spacing();
	if (fabs(this_scanner_length - scanner_length)>1.0)
	  {
	    warning("scanner length %g does not match 966 nor 962. Using 962 anyway", 
		    scanner_length);
	  }
      }
  }
  warning("Selected scanner %s", scanner_sptr->get_name().c_str());

  scanner_sptr->set_num_rings(num_rings);
  scanner_sptr->set_ring_spacing(scanner_length/num_rings);
  scanner_sptr->set_num_detectors_per_ring(num_views*2);
  shared_ptr<ProjDataInfo> proj_data_info_sptr =
    ProjDataInfo::ProjDataInfoCTI( scanner_sptr,
				   /*span=*/1, 
				   /*max_delta=*/max_ring_difference,
				   num_views,
				   num_tangential_poss,
				   /*arc_corrected =*/ true);
  dynamic_cast<ProjDataInfoCylindricalArcCorr&>(*proj_data_info_sptr).
    set_tangential_sampling(2*FOV_radius/num_tangential_poss);

  ProjDataInterfile proj_data(proj_data_info_sptr,
			      stir_filename, std::ios::out);
 


  if(strncmp(input_data_type,"fl",2)==0)
    {
    }
  else 
    {
      error("file format %s not valid. Only fl at present", input_data_type);
    }


  // skip simset header
  const long offset = 32768 + dataset_num*num_rings*(long(num_rings*nitems*4));
  if (fseek(file, offset, SEEK_SET) != 0)
    error("Error while skipping simset header and data sets (%ld). Maybe file too short?", offset); 
  Array<1,float> seq(0,(num_rings*num_rings*nitems)-1);
  read_data(file, seq /*, byteorder */);

  int i_ring_difference=0;
  int n=0;
  while(i_ring_difference<=max_ring_difference)
    {
      int lim_down=0;
      int lim_up=lim_down+((num_rings-i_ring_difference-1)*(num_rings+1));
      for(n=lim_down;n<=lim_up;) //Extraccion de los sinogramas de una serie !!!
	{

	  Sinogram<float> pos_sino = proj_data.get_empty_sinogram((n-lim_down)/(num_rings+1), i_ring_difference);
	  Sinogram<float> neg_sino = proj_data.get_empty_sinogram((n-lim_down)/(num_rings+1), -i_ring_difference);
	  Sinogram<float>::full_iterator pos_sino_iter = pos_sino.begin_all();
	  Sinogram<float>::full_iterator neg_sino_iter = neg_sino.begin_all();

	  int ii=nitems-1;
	  const int i_r1r2=(n + i_ring_difference)*nitems;
	  const int i_r2r1=(n + i_ring_difference*num_rings)*nitems;
     
	  // get 2 sinograms from simset data
	  for(int i=0;i<nitems/2;++i)  
	    {
	      *pos_sino_iter++ = seq[i_r1r2+ii];
	      *neg_sino_iter++ = seq[i_r2r1+ii];
	      ii=ii-1;  
	    }
	  for(int i=nitems/2;i<nitems;++i)   
	    {
	      *neg_sino_iter++ = seq[i_r1r2+ii];
	      *pos_sino_iter++ = seq[i_r2r1+ii];
	      ii=ii-1;  
	    }
	  n=n+num_rings+1;
	  proj_data.set_sinogram(pos_sino);
	  proj_data.set_sinogram(neg_sino);
	}
      i_ring_difference=i_ring_difference+1;
    }

  fclose(file);

  return EXIT_SUCCESS;
}












