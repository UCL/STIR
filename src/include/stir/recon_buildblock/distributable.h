//
// $Id$
//

#ifndef __DISTRIBUTABLE_H__
#define __DISTRIBUTABLE_H__

/*!
  \file
  \ingroup recon_buildblock

  \brief Declaration of the main function that performs parallel processing

  \author Alexey Zverovich
  \author Kris Thielemans
  \author Matthew Jacobson
  \author PARAPET project

   $Date$
   $Revision$
*/
#include "tomo/common.h"

START_NAMESPACE_TOMO

template <typename T> class shared_ptr;
template <typename elemT> class RelatedViewgrams;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
class ProjData;
class ProjDataInfo;
class DataSymmetriesForViewSegmentNumbers;
class ForwardProjectorByBin;
class BackProjectorByBin;


extern bool RPC_slave_sens_zero_seg0_end_planes; // = false;

extern shared_ptr<ForwardProjectorByBin> forward_projector_ptr;
extern shared_ptr<BackProjectorByBin> back_projector_ptr;
extern shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_ptr;

void set_projectors_and_symmetries(
       const shared_ptr<ForwardProjectorByBin>& current_forward_projector_ptr,
       const shared_ptr<BackProjectorByBin>& current_back_projector_ptr,
       const shared_ptr<DataSymmetriesForViewSegmentNumbers>& current_symmetries_ptr);

//! typedef for callback functions to be defined by the main program
/*! \warning The data in *measured_viewgrams_ptr are allowed to be overwritten */
typedef  void RPC_process_related_viewgrams_type (DiscretisedDensity<3,float>* output_image_ptr, 
                                             const DiscretisedDensity<3,float>* input_image_ptr, 
			   RelatedViewgrams<float>* measured_viewgrams_ptr,
			   int& count, int& count2, float* log_likelihood_ptr,
			   const RelatedViewgrams<float>* additive_binwise_correction_ptr);

/*!
  \brief A multi-purpose function that computes whatever it needs to !
  Output is in output_image_ptr and in float_out_ptr 
  (but only if they are not NULL).

  Whatever the output is, is really determined by the call-back function
  RPC_process_related_viewgrams.

 */
//TODOdoc
void distributable_computation(DiscretisedDensity<3,float>* output_image_ptr,
				    const DiscretisedDensity<3,float>* input_image_ptr,
				    const shared_ptr<ProjData>& proj_data_ptr,
                                    const bool read_from_proj_dat,
				    int subset_num, int num_subsets,
				    int min_segment, int max_segment,
				    bool zero_seg0_end_planes,
				    float* float_out_ptr,
				    const shared_ptr<ProjData>& additive_binwise_correction,
                                    RPC_process_related_viewgrams_type * RPC_process_related_viewgrams);




END_NAMESPACE_TOMO

#endif // __DISTRIBUTABLE_H__

