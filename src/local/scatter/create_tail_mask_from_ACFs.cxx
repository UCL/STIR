//
// $Id$
//
/*
Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
See STIR/LICENSE.txt for details
*/
/*!
\file
\ingroup utilities
\brief   

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
  $Date$
  $Revision$
	
  \par Usage:
  see .cxx file  
\code
  \endcode
  						
  Output: Viewgram with name scaled_scatter_filename
              
  \param ACF_threshold defaults to 1.01 (should be larger than 1)	  
*/

#include <iostream>
#include <fstream>
#include <string>
#include "stir/ProjDataInfo.h"
#include "stir/Sinogram.h"
#include "stir/ProjDataInterfile.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/Bin.h"
#include "boost/lambda/lambda.hpp"


/***********************************************************/     

using boost::lambda::_1;

static void
print_usage_and_exit(const char * const prog_name)
{
  std::cerr << "\nUsage:\n" << prog_name << "\n"
	    << "\t[--min-scale-factor number]\n"
	    << "\t[--max-scale-factor number]\n"
	    << "\tattenuation_correction_factors\n"
	    << "\temission_projdata\n"
	    << "\tscatter_projdata\n" 
	    << "\toutput_filename\n"
	    << "\t[ACF_threshold \n"
	    << "\t[correct_for_interleaving]\n"
	    << "\t[estimate_scale_factor]\n"
#ifdef SCFOLD
	    << "\t[mask_radius_in_mm]]]\n"
#else
	    << "\t[back_off_size]]\n"
#endif
	    << "correct_for_interleaving default to 1\n"
	    << "ACF_threshold defaults to 1.01\n" 
	    << "estimate_scale_factor defaults to 1 for scaling per sinogram\n"
	    << "0 for  no scaling, 1 for by sinogram\n"
	    << "If mask_radius_in_mm is not specified, will use all available data\n";
  exit(EXIT_FAILURE);
}

int main(int argc, const char *argv[])                                  
{         
  USING_NAMESPACE_STIR;
  const char * const prog_name = argv[0];

  // option processing
  float ACF_threshold = 1.01;
  int safety_margin=0;
  std::string ACF_filename;
  std::string output_filename;
  while (argc>1 && argv[1][1] == '-')
    {
      if (strcmp(argv[1], "--ACF-filename")==0)
	{
	  ACF_filename = (argv[2]);
	  argc-=2; argv +=2;
	}
      if (strcmp(argv[1], "--output-filename")==0)
	{
	  output_filename = (argv[2]);
	  argc-=2; argv +=2;
	}
      if (strcmp(argv[1], "--ACF-threshold")==0)
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
    
  if (ACF_filename.size()==0 || output_filename.size()==0)
    {
      print_usage_and_exit(prog_name);
    }      
  std::cout << "\nACF_threshold: " << ACF_threshold << " safety-margin : " << safety_margin << std::endl;


  shared_ptr< ProjData >  	
    ACF_sptr= 
    ProjData::read_from_file(ACF_filename);
  
  if (is_null_ptr(ACF_sptr))
    error("Check the attenuation_correct_factors file\n");
  
  ProjDataInterfile mask_proj_data(ACF_sptr->get_proj_data_info_ptr()->clone(), 
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
	  const Sinogram<float> att_sinogram = 
	    ACF_sptr->get_sinogram(bin.axial_pos_num(),bin.segment_num());
	  Sinogram<float> mask_sinogram = 
	    mask_proj_data.get_empty_sinogram(bin.axial_pos_num(),bin.segment_num());

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
	      // find left and right mask sizes
	      // sorry: a load of ugly casting to make sure we allow all datatypes
	      std::size_t mask_left_size =
		static_cast<std::size_t>(
					 std::max(0,
						  static_cast<int>
						  (std::find_if(att_line.begin(), att_line.end(),
								_1>=ACF_threshold) -
						   att_line.begin()) -
						  static_cast<int>(safety_margin))
					 );
	      std::size_t mask_right_size =
		static_cast<std::size_t>(
					 std::max(0,
						  static_cast<int>
						  (std::find_if(att_line.rbegin(), att_line.rend() - mask_left_size,
								_1>=ACF_threshold) -
						   att_line.rbegin()) -
						  static_cast<int>(safety_margin))
					 );
	      std::cout << "mask sizes " << mask_left_size << ", " << mask_right_size << '\n';
	      std::fill(mask_sinogram[bin.view_num()].begin(),
			mask_sinogram[bin.view_num()].begin() + mask_left_size,
			1.F);
	      std::fill(mask_sinogram[bin.view_num()].rbegin(),
			mask_sinogram[bin.view_num()].rbegin() + mask_right_size,
			1.F);
	      count += mask_left_size + mask_right_size;
#endif
	    }
	  std::cout << count << " bins in mask for this sinogram\n";
	  if (mask_proj_data.set_sinogram(mask_sinogram) != Succeeded::yes)
	    return EXIT_FAILURE;
	}
  }  

  return EXIT_SUCCESS;
}                 
