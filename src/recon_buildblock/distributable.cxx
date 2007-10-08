//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
  \ingroup recon_buildblock

  \brief Implementation of stir::distributable_computation()

  \author Kris Thielemans  
  \author Alexey Zverovich 
  \author Matthew Jacobson
  \author PARAPET project

  $Date$
  $Revision$
*/
/* Modification history:
   KT 30/05/2002
   get rid of dependence on specific symmetries (i.e. views up to 45 degrees)
   */
#include "stir/shared_ptr.h"
#include "stir/recon_buildblock/distributable.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ProjData.h"
#include "stir/DiscretisedDensity.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/CPUTimer.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_STIR


#ifndef PARALLEL
void distributable_computation(
			       const shared_ptr<ForwardProjectorByBin>& forward_projector_ptr,
			       const shared_ptr<BackProjectorByBin>& back_projector_ptr,
			       const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_ptr,
			       DiscretisedDensity<3,float>* output_image_ptr,
				    const DiscretisedDensity<3,float>* input_image_ptr,
				    const shared_ptr<ProjData>& proj_dat_ptr, 
                                    const bool read_from_proj_dat,
				    int subset_num, int num_subsets,
				    int min_segment_num, int max_segment_num,
				    bool zero_seg0_end_planes,
				    float* log_likelihood_ptr,
				    const shared_ptr<ProjData>& binwise_correction,
                                    RPC_process_related_viewgrams_type * RPC_process_related_viewgrams)
{
  assert(min_segment_num <= max_segment_num);
  assert(subset_num >=0);
  assert(subset_num < num_subsets);
  
  assert(proj_dat_ptr.use_count() != 0);
  
  if (output_image_ptr != NULL)
    output_image_ptr->fill(0);
  
  if (log_likelihood_ptr != NULL)
  {
    (*log_likelihood_ptr) = 0.0;
  };
  
  for (int segment_num = min_segment_num; segment_num <= max_segment_num; segment_num++)
  {
    CPUTimer segment_timer;
    segment_timer.start();
            
    int count=0, count2=0;

    // boolean used to see when to write diagnostic message
    bool first_view_in_segment = true;

    for (int view = proj_dat_ptr->get_min_view_num() + subset_num; 
        view <= proj_dat_ptr->get_max_view_num(); 
        view += num_subsets)
    {
      const ViewSegmentNumbers view_segment_num(view, segment_num);
        
      if (!symmetries_ptr->is_basic(view_segment_num))
        continue;

      if (first_view_in_segment)
      {
        cerr << "Starting to process segment " << segment_num << " (and symmetry related segments)" << endl;
        if (segment_num==0 && zero_seg0_end_planes)
          cerr << "End-planes of segment 0 will be zeroed" << endl;
        first_view_in_segment = false;
      }
    
      RelatedViewgrams<float>* additive_binwise_correction_viewgrams = NULL;
      if (binwise_correction.use_count() != 0) 
      {
#ifndef _MSC_VER
        additive_binwise_correction_viewgrams =
          new RelatedViewgrams<float>
	  (binwise_correction->get_related_viewgrams(view_segment_num, symmetries_ptr));
#else
	RelatedViewgrams<float> tmp(binwise_correction->
	  get_related_viewgrams(view_segment_num, symmetries_ptr));
        additive_binwise_correction_viewgrams = new RelatedViewgrams<float>(tmp);
#endif      
      }
      RelatedViewgrams<float>* y = NULL;
      
      if (read_from_proj_dat)
      {
#ifndef _MSC_VER
        y = new RelatedViewgrams<float>
	  (proj_dat_ptr->get_related_viewgrams(view_segment_num, symmetries_ptr));
#else
        // workaround VC++ 6.0 bug
        RelatedViewgrams<float> tmp(proj_dat_ptr->
	  get_related_viewgrams(view_segment_num, symmetries_ptr));
        y = new RelatedViewgrams<float>(tmp);
#endif        
      }
      else
      {
        y = new RelatedViewgrams<float>
	  (proj_dat_ptr->get_empty_related_viewgrams(view_segment_num, symmetries_ptr));
      }
      
#ifndef NDEBUG
      // test if symmetries didn't take us out of the segment range
      for (RelatedViewgrams<float>::iterator r_viewgrams_iter = y->begin();
           r_viewgrams_iter != y->end();
           ++r_viewgrams_iter)
      {
        assert(r_viewgrams_iter->get_segment_num() >= min_segment_num);
        assert(r_viewgrams_iter->get_segment_num() <= max_segment_num);
      }
#endif

      if (segment_num==0 && zero_seg0_end_planes)
      {
      
        if (y != NULL)
        {
          const int min_ax_pos_num = y->get_min_axial_pos_num();
          const int max_ax_pos_num = y->get_max_axial_pos_num();
          for (RelatedViewgrams<float>::iterator r_viewgrams_iter = y->begin();
               r_viewgrams_iter != y->end();
               ++r_viewgrams_iter)
          {
            (*r_viewgrams_iter)[min_ax_pos_num].fill(0);
            (*r_viewgrams_iter)[max_ax_pos_num].fill(0);
          }
        };
        
        if (additive_binwise_correction_viewgrams != NULL)
        {
          const int min_ax_pos_num = additive_binwise_correction_viewgrams->get_min_axial_pos_num();
          const int max_ax_pos_num = additive_binwise_correction_viewgrams->get_max_axial_pos_num();
          for (RelatedViewgrams<float>::iterator r_viewgrams_iter = additive_binwise_correction_viewgrams->begin();
               r_viewgrams_iter != additive_binwise_correction_viewgrams->end();
               ++r_viewgrams_iter)
          {
            (*r_viewgrams_iter)[min_ax_pos_num].fill(0);
            (*r_viewgrams_iter)[max_ax_pos_num].fill(0);
          }
        }
      }
      
      
      RPC_process_related_viewgrams(forward_projector_ptr,
				    back_projector_ptr,
				    output_image_ptr, input_image_ptr, y, count, count2, log_likelihood_ptr, additive_binwise_correction_viewgrams);
      
      if (additive_binwise_correction_viewgrams != NULL)
      {
        delete additive_binwise_correction_viewgrams;
        additive_binwise_correction_viewgrams = NULL;
      };
      delete y;
    }

    if (first_view_in_segment != true) // only write message when at least one view was processed
    {
      // TODO this message relies on knowledge of count, count2 which might be inappropriate for 
      // the call-back function
      cerr<<"\tNumber of (cancelled) singularities: "<<count
          <<"\n\tNumber of (cancelled) negative numerators: "<<count2
          << "\n\tSegment " << segment_num << ": " << segment_timer.value() << "secs" <<endl;
    }
    
  }  
}

#endif // !PARALLEL

END_NAMESPACE_STIR
