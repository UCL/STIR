//
// $Id$: $Date$
//

#ifndef __common_H__
#define __common_H__

/*!

  \file
  \ingroup LogLikBased_buildblock

  \brief Declaration of common routines for LogLikelihoodBased algorithms

  \author Alexey Zverovich
  \author Kris Thielemans
  \author Matthew Jacobson
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#include "Tomography_common.h"


START_NAMESPACE_TOMO

template <typename T> class shared_ptr;
template <typename elemT> class RelatedViewgrams;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
class ProjData;
class ProjDataInfo;
class DataSymmetriesForViewSegmentNumbers;

const int rim_truncation_sino  = 4; // This MUST be const, otherwise it will need
                                    // to be distributed to the slaves
const int rim_truncation_image = 2;


//! computes gradient of (subset of ) loglikelihood 
void distributable_compute_gradient(DiscretisedDensity<3,float>& output_image_ptr,
				    const DiscretisedDensity<3,float>& current_estimate,
				    shared_ptr<ProjData> const& proj_data_ptr,
				    int subset_num, int num_subsets,
				    int min_segment, int max_segment,
				    bool zero_seg0_end_planes,
				    float* log_likelihood = NULL,
				    shared_ptr<ProjData> const& binwise_correction = NULL);

//TODO move somewhere else
void distributable_backproject(DiscretisedDensity<3,float>& output_image,
			       shared_ptr<ProjData> const& proj_data_ptr,
			       const int subset_num,
                               const int num_subsets,
                               const int min_segment,
                               const int max_segment,
                               bool zero_seg0_end_planes);

/*! \brief computes the sensitivity image

  if attenuation_image_ptr==NULL, this skips the forward projection step by taking
  effectively 1 for the attenuation factors.

  result is properly initialised
  */
void distributable_compute_sensitivity_image(DiscretisedDensity<3,float>& result,
					     shared_ptr<ProjDataInfo> const& proj_data_info_ptr,
					     DiscretisedDensity<3,float> const* attenuation_image_ptr,
					     const int subset_num,
					     const int num_subsets,
					     const int min_segment,
					     const int max_segment,
					     bool zero_seg0_end_planes,
					     const shared_ptr<ProjData>& multiplicative_sinogram = NULL);

//! Accumulates the log likelihood value in log_likelihood_ptr
void distributable_accumulate_loglikelihood(const DiscretisedDensity<3,float>& current_estimate,
				    shared_ptr<ProjData> const& proj_dat_ptr,
				    int subset_num, int num_subsets,
				    int min_segment, int max_segment,
				    bool zero_seg0_end_planes,
				    float* log_likelihood_ptr,
				    shared_ptr<ProjData> const& binwise_correction = NULL);


END_NAMESPACE_TOMO

#endif
