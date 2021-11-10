/*!

  \file
  \ingroup projdata

  \brief Implementation of stir::multiply_crystal_factors

  \author Kris Thielemans

*/
/*
  Copyright (C) 2001- 2012, Hammersmith Imanet Ltd
  Copyright (C) 2021, University Copyright London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/

#include "stir/multiply_crystal_factors.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Scanner.h"
#include "stir/Bin.h"
#include "stir/stream.h"
#include "stir/Sinogram.h"
#include "stir/IndexRange2D.h"

START_NAMESPACE_STIR

void multiply_crystal_factors(ProjData& proj_data, const Array<2,float>& efficiencies, const float global_factor)
{
    const ProjDataInfoCylindricalNoArcCorr * const proj_data_info_ptr = 
      dynamic_cast<const ProjDataInfoCylindricalNoArcCorr * const>
      (proj_data.get_proj_data_info_sptr().get());
    if (proj_data_info_ptr == 0)
      {
	error("Can only process not arc-corrected data\n");
      }
    const int max_ring_diff = 
      proj_data_info_ptr->get_max_ring_difference
      (proj_data_info_ptr->get_max_segment_num());

    const int mashing_factor = 
      proj_data_info_ptr->get_view_mashing_factor();

    shared_ptr<Scanner> scanner_sptr(new Scanner(*proj_data_info_ptr->get_scanner_ptr()));
    const int num_detectors_per_ring = 
      scanner_sptr->get_num_detectors_per_ring();
    unique_ptr<ProjDataInfo> uncompressed_proj_data_info_uptr
      (ProjDataInfo::construct_proj_data_info(scanner_sptr,
                                              /*span=*/1, max_ring_diff,
                                              /*num_views=*/ num_detectors_per_ring/2,
                                              scanner_sptr->get_max_num_non_arccorrected_bins(),
                                              /*arccorrection=*/false));
    const ProjDataInfoCylindricalNoArcCorr * const
      uncompressed_proj_data_info_ptr =
      dynamic_cast<const ProjDataInfoCylindricalNoArcCorr * const>
      (uncompressed_proj_data_info_uptr.get());

    
    Bin bin;
    Bin uncompressed_bin;
    // current code makes assumptions about mashing
    if (proj_data.get_min_view_num()!=0)
      error("Can only handle min_view_num==0\n");

    for (bin.segment_num() = proj_data.get_min_segment_num(); 
	 bin.segment_num() <= proj_data.get_max_segment_num();  
	 ++ bin.segment_num())
      {	
    
	for (bin.axial_pos_num() = proj_data.get_min_axial_pos_num(bin.segment_num());
	     bin.axial_pos_num() <= proj_data.get_max_axial_pos_num(bin.segment_num());
	     ++bin.axial_pos_num())
	  {
	    Sinogram<float> sinogram =
	      proj_data_info_ptr->get_empty_sinogram(bin.axial_pos_num(),bin.segment_num());
	    const float out_m = proj_data_info_ptr->get_m(bin);
	    const int in_min_segment_num =
	      proj_data_info_ptr->get_min_ring_difference(bin.segment_num());
	    const int in_max_segment_num =
	      proj_data_info_ptr->get_max_ring_difference(bin.segment_num());

	    // now loop over uncompressed detector-pairs
	    {  
	      for (uncompressed_bin.segment_num() = in_min_segment_num; 
		   uncompressed_bin.segment_num() <= in_max_segment_num;
		   ++uncompressed_bin.segment_num())
		for (uncompressed_bin.axial_pos_num() = uncompressed_proj_data_info_ptr->get_min_axial_pos_num(uncompressed_bin.segment_num()); 
		     uncompressed_bin.axial_pos_num()  <= uncompressed_proj_data_info_ptr->get_max_axial_pos_num(uncompressed_bin.segment_num());
		     ++uncompressed_bin.axial_pos_num() )
		  {
		    const float in_m = uncompressed_proj_data_info_ptr->get_m(uncompressed_bin);
		    if (fabs(out_m - in_m) > 1E-4)
		      continue;
		
#ifdef STIR_OPENMP
#  if _OPENMP >= 200711
#     pragma omp parallel for collapse(2) // OpenMP 3.1
#  else
#     pragma omp parallel for // older versions
#  endif
#endif
		    for (int view_num = proj_data.get_min_view_num();
			 view_num <= proj_data.get_max_view_num();
			 ++ view_num)
		      {

			for (int tangential_pos_num = proj_data_info_ptr->get_min_tangential_pos_num();
			     tangential_pos_num <= proj_data_info_ptr->get_max_tangential_pos_num();
			     ++tangential_pos_num)
			  {
                            Bin parallel_bin(bin);
                            parallel_bin.view_num() = view_num;
                            parallel_bin.tangential_pos_num() = tangential_pos_num;
			    Bin parallel_uncompressed_bin = uncompressed_bin;
			    parallel_uncompressed_bin.tangential_pos_num() = parallel_bin.tangential_pos_num();
                            float result = 0.F;
			    for (parallel_uncompressed_bin.view_num() = parallel_bin.view_num()*mashing_factor;
				 parallel_uncompressed_bin.view_num() < (parallel_bin.view_num()+1)*mashing_factor;
				 ++ parallel_uncompressed_bin.view_num())
			      {

				int ra = 0, a = 0;
				int rb = 0, b = 0;
			      
				uncompressed_proj_data_info_ptr->get_det_pair_for_bin(a, ra, b, rb, 
										      parallel_uncompressed_bin);

                                result += efficiencies[ra][a]*efficiencies[rb][b%num_detectors_per_ring];
			      }
#if defined(STIR_OPENMP)
# if _OPENMP >= 201012
#  pragma omp atomic update
# else
#  pragma omp critical(STIRMULTIPLYCRYSTALFACTORS)
                            {
# endif
#endif
                              /*(*segment_ptr)[bin.axial_pos_num()]*/
                              sinogram[parallel_bin.view_num()][parallel_bin.tangential_pos_num()] += result * global_factor;
#if defined(STIR_OPENMP) and _OPENMP < 201012
                            }
#endif
			  }
		      }
		  
		  
		  }
	    }
	    proj_data.set_sinogram(sinogram);
	  }

      }
}
END_NAMESPACE_STIR
