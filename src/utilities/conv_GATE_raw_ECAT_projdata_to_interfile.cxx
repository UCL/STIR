/*
 Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
 Copyright (C) 2010- 2013, King's College London
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
/*!
 \file 
 \ingroup utilities 
 \brief  This program converts GATE ECAT output (.ima) into STIR interfile format
 
 \author Charalampos Tsoumpas
 \author Pablo Aguiar
 \author Kris Thielemans 
 */

#include "stir/ProjDataInterfile.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Sinogram.h"
#include "stir/Scanner.h"
#include "stir/IO/read_data.h"
#include "stir/Succeeded.h"
#include "stir/NumericType.h"

#define NUMARG 8

int main(int argc,char **argv)
{
  using namespace stir;
	
  static const char * const options[]={
    "argv[1]  GATE file\n",
    "argv[2]  Angles in GATE file\n",
    "argv[3]  Bins in GATE file\n",
    "argv[4]  Rings in GATE file\n",
    "argv[5]  STIR scanner name\n",
    "argv[6]  maximum ring difference to use for writing\n",
    "argv[7]  STIR file name\n"
  };
  if (argc!=NUMARG){
    std::cerr << "\n\nConvert GATE ECAT format to STIR\n\n";
    std::cerr << "Not enough arguments !!! ..\n";
    for (int i=1;i<NUMARG;i++) std::cerr << options[i-1];
    exit(EXIT_FAILURE);
  }
	
  const char * const GATE_filename = argv[1];
  const int num_views = atoi(argv[2]);
  const int num_tangential_poss = atoi(argv[3]);
  const int num_rings = atoi(argv[4]);
  const char * const scanner_name = argv[5];
  const int max_ring_difference = atoi(argv[6]);
  const char * const STIR_output_filename = argv[7];
  const long num_bins_per_sino = num_views*num_tangential_poss;
		
  if(max_ring_difference>=num_rings)
    error("Cannot have max ring difference larger than the number of rings");	
  if( sizeof(short int)!=2 ) 
    error("Expected Input Data should be in UINT16 format\nand size of short int is should be 2\n") ;
  shared_ptr<Scanner> scanner_sptr( Scanner::get_scanner_from_name(scanner_name));
  if (is_null_ptr(scanner_sptr))
    error("Scanner '%s' is not a valid name", scanner_name);
	
  FILE *GATE_file ;
  if( (GATE_file=fopen(GATE_filename,"rb"))==NULL)
    error("Cannot open GATE file %s", GATE_filename);
  else {
    long GATE_file_size=fseek(GATE_file, 0, SEEK_END);
    GATE_file_size=ftell(GATE_file);
    rewind(GATE_file);
    std::cerr << GATE_file_size << " size of file" << std::endl; 
    std::cerr << num_bins_per_sino << " bins per sino" << std::endl; 
		
    if( GATE_file_size%num_bins_per_sino!=0 ) 
      error("Expected Input Data should be multiple of the number of bins per sinogram. Check input for bins and angles or GATE file.\n") ;
  }
  const float STIR_scanner_length = 
    scanner_sptr->get_num_rings() * scanner_sptr->get_ring_spacing();
  scanner_sptr->set_num_rings(num_rings);
  scanner_sptr->set_ring_spacing(STIR_scanner_length/num_rings);
  scanner_sptr->set_num_detectors_per_ring(num_views*2);
  shared_ptr<ProjDataInfo> proj_data_info_sptr(
                                               ProjDataInfo::ProjDataInfoCTI( scanner_sptr,
                                                                              /*span=*/1, 
                                                                              /*max_delta=*/max_ring_difference,
                                                                              num_views,
                                                                              num_tangential_poss,
                                                                              /*arc_corrected =*/ false)); 
  shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
  ProjDataInterfile proj_data(exam_info_sptr, proj_data_info_sptr,
                              STIR_output_filename, std::ios::out);
	
  // loop over segments in the GATE ECAT output in a fancy way
  for (int seg_num=0; seg_num<=max_ring_difference; seg_num = (seg_num<=0) ? 1+seg_num*-1 : -1*seg_num)
    {
      // correct sign
      const int segment_num = -seg_num;
      for (int axial_pos_num = proj_data.get_min_axial_pos_num(segment_num);
           axial_pos_num <= proj_data.get_max_axial_pos_num(segment_num); 
           axial_pos_num++)
        {
          Sinogram<float> sino = proj_data.get_empty_sinogram(axial_pos_num,segment_num);		
          float scale=1;
          if (read_data(GATE_file, sino, NumericType::SHORT, scale) !=
	      Succeeded::yes)
	    {
              warning("error reading from GATE sino");
              fclose(GATE_file);
	      return EXIT_FAILURE;
	    }
          proj_data.set_sinogram(sino);
        }
    }
  fclose(GATE_file);
	
  return EXIT_SUCCESS;
}
