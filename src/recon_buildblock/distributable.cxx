//
// $Id$: $Date$
//
/*!

  \file
  \ingroup recon_buildblock

  \brief Implementation of distributable_computation() and related functions

  \author Kris Thielemans  
  \author Alexey Zverovich 
  \author Matthew Jacobson
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#include "shared_ptr.h"
#include "recon_buildblock/distributable.h"
#include "RelatedViewgrams.h"
#include "ProjData.h"
#include "DiscretisedDensity.h"
#include "CPUTimer.h"
#include "recon_buildblock/ForwardProjectorByBin.h"
#include "recon_buildblock/BackProjectorByBin.h"

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_TOMO

bool RPC_slave_sens_zero_seg0_end_planes = false;


shared_ptr<ForwardProjectorByBin> forward_projector_ptr;
shared_ptr<BackProjectorByBin> back_projector_ptr;
shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_ptr;

void set_projectors_and_symmetries(
       const shared_ptr<ForwardProjectorByBin>& current_forward_projector_ptr,
       const shared_ptr<BackProjectorByBin>& current_back_projector_ptr,
       const shared_ptr<DataSymmetriesForViewSegmentNumbers>& current_symmetries_ptr)
{
  forward_projector_ptr = current_forward_projector_ptr;
  back_projector_ptr = current_back_projector_ptr;
  symmetries_ptr = current_symmetries_ptr;
}

#ifndef PARALLEL
void distributable_computation(DiscretisedDensity<3,float>* output_image_ptr,
				    const DiscretisedDensity<3,float>* input_image_ptr,
				    const shared_ptr<ProjData>& proj_dat_ptr, 
                                    const bool read_from_proj_dat,
				    int subset_num, int num_subsets,
				    int min_segment, int max_segment,
				    bool zero_seg0_end_planes,
				    float* log_likelihood_ptr,
				    const shared_ptr<ProjData>& binwise_correction,
                                    RPC_process_related_viewgrams_type * RPC_process_related_viewgrams)
{
  assert(min_segment <= max_segment);
  
  assert(proj_dat_ptr.use_count() != 0);
  
  if (output_image_ptr != NULL)
    output_image_ptr->fill(0);
  
  if (log_likelihood_ptr != NULL)
  {
    (*log_likelihood_ptr) = 0.0;
  };
  
  RPC_slave_sens_zero_seg0_end_planes = zero_seg0_end_planes;
  
  for (int segment_num = min_segment; segment_num <= max_segment; segment_num++)
  {
    
    cerr << "Starting to process segment pair " << segment_num << endl;
    if (segment_num==0 && zero_seg0_end_planes)
      cerr << "\nEnd-planes of segment 0 will be zeroed" << endl;
    

    CPUTimer segment_timer;
    segment_timer.start();
        
    
    int count=0, count2=0;
    // TODO replace by something with symmetries
    const int view45 = proj_dat_ptr->get_num_views()/4;

    // KT 31/05/2000 go now upto view45 because of change in subset scheme
    for (int view = subset_num; view <= view45; view += num_subsets)
    {

      RelatedViewgrams<float>* additive_binwise_correction_viewgrams = NULL;
      if (binwise_correction.use_count() != 0) 
      {
#ifndef _MSC_VER
        const ViewSegmentNumbers view_segmnet_num(view, segment_num);
        additive_binwise_correction_viewgrams =
          new RelatedViewgrams<float>
	  (binwise_correction->get_related_viewgrams
	  (view_segmnet_num.view_num(),view_segmnet_num.segment_num(), symmetries_ptr));
#else
	const ViewSegmentNumbers view_segmnet_num(view, segment_num);
        RelatedViewgrams<float> tmp(binwise_correction->
	  get_related_viewgrams
	  (view_segmnet_num, symmetries_ptr));
        additive_binwise_correction_viewgrams = new RelatedViewgrams<float>(tmp);
#endif      
      }
      RelatedViewgrams<float>* y = NULL;
      
      if (read_from_proj_dat)
      {
#ifndef _MSC_VER
       const ViewSegmentNumbers view_segmnet_num(view, segment_num);
        y = new RelatedViewgrams<float>
	  (proj_dat_ptr->get_related_viewgrams
	  (view_segmnet_num.view_num(),view_segmnet_num.segment_num(), symmetries_ptr));
#else
        // workaround VC++ 6.0 bug
        const ViewSegmentNumbers view_segmnet_num(view, segment_num);
        RelatedViewgrams<float> tmp(proj_dat_ptr->
	  get_related_viewgrams
	  (view_segmnet_num, symmetries_ptr));
        y = new RelatedViewgrams<float>(tmp);
#endif        
      }
      else
      {
      const ViewSegmentNumbers view_segmnet_num(view, segment_num);
        y = new RelatedViewgrams<float>
	  (proj_dat_ptr->get_empty_related_viewgrams(view_segmnet_num, symmetries_ptr));
      }

      
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
      
      
      RPC_process_related_viewgrams(output_image_ptr, input_image_ptr, y, count, count2, log_likelihood_ptr, additive_binwise_correction_viewgrams);
      
      if (additive_binwise_correction_viewgrams != NULL)
      {
        delete additive_binwise_correction_viewgrams;
        additive_binwise_correction_viewgrams = NULL;
      };
      delete y;
    }
    

    
    cerr<<"Number of (cancelled) singularities: "<<count<<endl;
    cerr<<"Number of (cancelled) negative numerators: "<<count2<<endl;
    
    cerr << "Segment " << segment_num << ": " << segment_timer.value() << "secs " <<endl;
    

    
  }  
}

#endif // !PARALLEL

END_NAMESPACE_TOMO
