//
//
/*!

  \file
  \ingroup utilities

  \brief Find sinogram rescaling factors

  \author Sanida Mustafovic
  \author Kris Thielemans
*/
/*
    Copyright (C) 2003- 2004, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInterfile.h"
#include "stir/Succeeded.h"
#include "stir/Sinogram.h"
#include "stir/stream.h"

#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::cerr;
using std::endl;
using std::ofstream;
#endif


int
main( int argc, char* argv[])
{

  USING_NAMESPACE_STIR;

  if ( argc !=4)
    {
      cerr << " Usage: " << argv[0] << " output_filename numerator_proj_data denominator" << endl;
      return EXIT_FAILURE;
    }

  const string scaling_factors_filename =  argv[1];
  shared_ptr<ProjData> numerator_sptr = ProjData::read_from_file(argv[2]);
  shared_ptr<ProjData> denominator_sptr = ProjData::read_from_file(argv[3]);

  if (*numerator_sptr->get_proj_data_info_ptr() != 
      *denominator_sptr->get_proj_data_info_ptr())
    {
      warning("Input files should have same characteristics (such as sizes etc).\n");
      return EXIT_FAILURE;
    }

#if 0
  shared_ptr<ProjData> rescaling_factors_sptr = 
    new ProjDataInterfile(numerator_sptr->get_proj_data_info_ptr()->clone(),
			  scaling_factors_filename);
#endif

  
  const int min_segment_num = numerator_sptr->get_min_segment_num();
  const int max_segment_num = numerator_sptr->get_max_segment_num();
  
  // define array of scaling_factors of appropriate range
  Array<3,float> scaling_factors;
  {
    // sorry, this is pretty horrible...
    IndexRange<3> scaling_factor_index_range;
    scaling_factor_index_range.grow(min_segment_num, max_segment_num);
    for ( int segment_num = numerator_sptr->get_min_segment_num();
	  segment_num <= numerator_sptr->get_max_segment_num();
	  ++segment_num)
      {
	const int min_axial_pos_num =
	  numerator_sptr->get_min_axial_pos_num(segment_num);
	const int max_axial_pos_num =
	  numerator_sptr->get_max_axial_pos_num(segment_num);
	scaling_factor_index_range[segment_num].grow(min_axial_pos_num, max_axial_pos_num);
	for (int axial_pos_num = numerator_sptr->get_min_axial_pos_num(segment_num);
	     axial_pos_num <= numerator_sptr->get_max_axial_pos_num(segment_num);
	     ++axial_pos_num)
	  {
	    scaling_factor_index_range[segment_num][axial_pos_num] =
	      IndexRange<1>(numerator_sptr->get_min_view_num(),
			    numerator_sptr->get_max_view_num());
	  }
      }
    scaling_factors.grow(scaling_factor_index_range);
  }

  for ( int segment_num = numerator_sptr->get_min_segment_num();
	segment_num <= numerator_sptr->get_max_segment_num();
	++segment_num)
    {

      for (int axial_pos = numerator_sptr->get_min_axial_pos_num(segment_num);
	   axial_pos <= numerator_sptr->get_max_axial_pos_num(segment_num);
	   ++axial_pos)
	{
         
	  const Sinogram<float> numerator_sinogram =
	    numerator_sptr->get_sinogram(axial_pos, segment_num);
          const Sinogram<float> denominator_sinogram =
	    denominator_sptr->get_sinogram(axial_pos, segment_num);
          

	  for ( int view_num = numerator_sinogram.get_min_view_num();
		view_num <= numerator_sinogram.get_max_view_num();
		view_num++)
	    {
	      float numerator_sum =0;
	      float denominator_sum =0;

	      for ( int tang_pos = numerator_sinogram.get_min_tangential_pos_num();
		    tang_pos <= numerator_sinogram.get_max_tangential_pos_num();
		    tang_pos++)
		{
		  numerator_sum += numerator_sinogram[view_num][tang_pos];
		  denominator_sum += denominator_sinogram[view_num][tang_pos];	 
		}
	  
	      scaling_factors[segment_num][axial_pos][view_num] = 
		numerator_sum/denominator_sum; 
	    }
	  
	}
  
    }
  
  ofstream outfile(scaling_factors_filename.c_str());
  if(!outfile)
    {
      warning("Error opening output file %d\n", scaling_factors_filename.c_str());
      return EXIT_FAILURE;
    }

  outfile << scaling_factors;
  if(!outfile)
    {
      warning("Error writing to output file %d\n", scaling_factors_filename.c_str());
      return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;

}
 
