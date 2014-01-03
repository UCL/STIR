/*
  Copyright (C) 2005 - 2011-12-31, Hammersmith Imanet Ltd
  Copyright (C) 2011-07-01 - 2012, Kris Thielemans
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
  \ingroup scatter
  \brief Compute a mask for the "tails" in the sinogram

  This computes the outer part of the projection data that 
  does not receive any (unscattered) contributions from inside the body.
  Its normal input is a projection data with attenuation-correction factors
  (ACFs). We take a threshold as parameter (which should be a bit larger 
  than 1). Along each line in the sinogram, we search from the edges 
  until we find the first bin above the threshold. We then go back a
  few bins (as set by \a safety-margin). All bins from the edge until
  this point are included in the mask.  

  \author Kris Thielemans
	
  \par Usage:   
  
  \verbatim
   create_tail_mask_from_ACFs --ACF-filename <filename> \\
	    --output-filename <filename> \\
	    [--ACF-threshold <float>] \\
	    [--safety-margin <integer>]
  \endverbatim
  ACF-threshold defaults to 1.1 (should be larger than 1), safety-margin to 4
*/

#include "stir/ProjDataInfo.h"
#include "stir/Sinogram.h"
#include "stir/ProjDataInterfile.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/Bin.h"
#include "boost/lambda/lambda.hpp"
#include <iostream>
#include <fstream>
#include <string>


/***********************************************************/     

static void
print_usage_and_exit(const char * const prog_name)
{
  std::cerr << "\nUsage:\n" << prog_name << "\n"
	    << "\t--ACF-filename <filename>\n"
	    << "\t--output-filename <filename>\n"
	    << "\t[--ACF-threshold <float>]\n"
	    << "\t[--safety-margin <integer>]\n"
	    << "ACF-threshold defaults to 1.1, safety-margin to 4\n";
  exit(EXIT_FAILURE);
}

int main(int argc, const char *argv[])                                  
{         
  USING_NAMESPACE_STIR;
  const char * const prog_name = argv[0];

  // option processing
  float ACF_threshold = 1.1;
  int safety_margin=4;
  std::string ACF_filename;
  std::string output_filename;
  while (argc>2 && argv[1][1] == '-')
    {
      if (strcmp(argv[1], "--ACF-filename")==0)
	{
	  ACF_filename = (argv[2]);
	  argc-=2; argv +=2;
	}
      else if (strcmp(argv[1], "--output-filename")==0)
	{
	  output_filename = (argv[2]);
	  argc-=2; argv +=2;
	}
      else if (strcmp(argv[1], "--ACF-threshold")==0)
	{
	  ACF_threshold = atof(argv[2]);
	  argc-=2; argv +=2;
	}
      else if (strcmp(argv[1], "--safety-margin")==0)
	{
	  safety_margin = atoi(argv[2]);
	  argc-=2; argv +=2;
	}
      else
	{
	  std::cerr << "\nUnknown option: " << argv[1];
	  print_usage_and_exit(prog_name);
	}
    }
    
  if (argc!=1 || ACF_filename.size()==0 || output_filename.size()==0)
    {
      print_usage_and_exit(prog_name);
    }      
  if (ACF_threshold<=1)
    error("ACF-threshold should be larger than 1");

  std::cout << "\nACF-threshold: " << ACF_threshold << " safety-margin : " << safety_margin << std::endl;


  shared_ptr< ProjData >  	
    ACF_sptr(ProjData::read_from_file(ACF_filename));
  
  if (is_null_ptr(ACF_sptr))
    error("Check the attenuation_correct_factors file");
  
  ProjDataInterfile mask_proj_data(ACF_sptr->get_exam_info_sptr(),
				   ACF_sptr->get_proj_data_info_ptr()->create_shared_clone(), 
				   output_filename);
  
  Bin bin;
  {
    for (bin.segment_num()=mask_proj_data.get_min_segment_num();
	 bin.segment_num()<=mask_proj_data.get_max_segment_num();
	 ++bin.segment_num())		
      for (bin.axial_pos_num()=
	     mask_proj_data.get_min_axial_pos_num(bin.segment_num());
	   bin.axial_pos_num()<=mask_proj_data.get_max_axial_pos_num(bin.segment_num());
	   ++bin.axial_pos_num())
	{
	  const Sinogram<float> att_sinogram
	    (ACF_sptr->get_sinogram(bin.axial_pos_num(),bin.segment_num()));
	  Sinogram<float> mask_sinogram
	    (mask_proj_data.get_empty_sinogram(bin.axial_pos_num(),bin.segment_num()));

	  int  count=0;
	  for (bin.view_num()=mask_proj_data.get_min_view_num();
	       bin.view_num()<=mask_proj_data.get_max_view_num();
	       ++bin.view_num())
	    {
#ifdef SCFOLD
	      for (bin.tangential_pos_num()=
		     mask_proj_data.get_min_tangential_pos_num();
		   bin.tangential_pos_num()<=
		     mask_proj_data.get_max_tangential_pos_num();
		   ++bin.tangential_pos_num())
		if (att_sinogram[bin.view_num()][bin.tangential_pos_num()]<ACF_threshold &&
		    (mask_radius_in_mm<0 || mask_radius_in_mm>= std::fabs(scatter_proj_data.get_proj_data_info_ptr()->get_s(bin))))
		  {
		    ++count;
		    mask_sinogram[bin.view_num()][bin.tangential_pos_num()]=1;
		  }
		else
		    mask_sinogram[bin.view_num()][bin.tangential_pos_num()]=0;
#else
	      const Array<1,float>& att_line=att_sinogram[bin.view_num()];

              using boost::lambda::_1;

	      // find left and right mask sizes
	      // sorry: a load of ugly casting to make sure we allow all datatypes
	      std::size_t mask_left_size =
		static_cast<std::size_t>(
					 std::max(0,
						  static_cast<int>
						  (std::find_if(att_line.begin(), att_line.end(),
								_1>=ACF_threshold) -
						   att_line.begin()) -
						  safety_margin)
					 );
	      std::size_t mask_right_size =
		static_cast<std::size_t>(
					 std::max(0,
						  static_cast<int>
						  (std::find_if(att_line.rbegin(), att_line.rend() - mask_left_size,
								_1>=ACF_threshold) -
						   att_line.rbegin()) -
						  safety_margin)
					 );
#if 0
	      std::cout << "mask sizes " << mask_left_size << ", " << mask_right_size << '\n';
#endif
	      std::fill(mask_sinogram[bin.view_num()].begin(),
			mask_sinogram[bin.view_num()].begin() + mask_left_size,
			1.F);
	      std::fill(mask_sinogram[bin.view_num()].rbegin(),
			mask_sinogram[bin.view_num()].rbegin() + mask_right_size,
			1.F);
	      count += mask_left_size + mask_right_size;
#endif
	    }
	  std::cout << count << " bins in mask for sinogram at segment " 
		    << bin.segment_num() << ", axial_pos " << bin.axial_pos_num() << "\n";
	  if (mask_proj_data.set_sinogram(mask_sinogram) != Succeeded::yes)
	    return EXIT_FAILURE;
	}
  }  

  return EXIT_SUCCESS;
}                 
