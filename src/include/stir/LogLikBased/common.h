//
// $Id$: $Date$
//

#ifndef __LogLikBased_common_H__
#define __LogLikBased_common_H__

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

const int rim_truncation_sino  = 0; // This MUST be const, otherwise it will need
                                    // to be distributed to the slaves
const int rim_truncation_image = 0;


//! computes gradient of (subset of ) loglikelihood 
/*!
  This function performs (over the given subset)
  \verbatim
  backproj(proj_data/(forwproj(current_estimate)+additive_data).
  \endverbatim
  Note that normalisation and attenuation factors cancel in this calculation 
  (at least if included in the additivie term).

  \warning This really computes the gradient+sensitivity_image.

  \param output_image has to be properly initialised (correct sizes etc.) before calling 
  this function
  \param current_estimate is the image at which the gradient will be computed
  \param proj_data_ptr is the measured data
  \param additive_binwise_correction_ptr This terms is added to the forward projected data,
    unless the pointer is 0. It needs to have the same dimensions as given by the 
    proj_dat_ptr argument.
*/
void distributable_compute_gradient(DiscretisedDensity<3,float>& output_image,
				    const DiscretisedDensity<3,float>& current_estimate,
				    shared_ptr<ProjData> const& proj_data_ptr,
				    int subset_num, int num_subsets,
				    int min_segment, int max_segment,
				    bool zero_seg0_end_planes,
				    float* log_likelihood = NULL,
				    shared_ptr<ProjData> const& additive_binwise_correction_ptr = NULL);

//TODO move somewhere else
void distributable_backproject(DiscretisedDensity<3,float>& output_image,
			       shared_ptr<ProjData> const& proj_data_ptr,
			       const int subset_num,
                               const int num_subsets,
                               const int min_segment,
                               const int max_segment,
                               bool zero_seg0_end_planes);

/*! \brief computes the sensitivity image for a given subset

  It does this by filling viewgrams with 1s, applying the given correction factors to it, 
  and then backprojecting them.

  \param result has to be properly initialised (correct sizes etc.) before calling 
  this function
  \param proj_data_info_ptr is used to determine the sampling of projection. 
  \param attenuation_image_ptr has to contain an estimate of the mu-map for the image. It will used
    to estimate attenuation factors as exp(-forw_proj(*attenuation_image_ptr)).
    if attenuation_image_ptr==NULL, the forward projection step is skipped, and 
    the attenuation factors are effectively set to 1.
  \param multiplicative_proj_data_ptr the viewgram is <i> divided</i> before backprojecting, 
    hence this argument is suitable to pass normalisation factors, and/or attenuation
    correction factors. Currently, it needs to have the same dimensions as given by the 
    proj_dat_info_ptr argument.

  */
void distributable_compute_sensitivity_image(DiscretisedDensity<3,float>& result,
					     shared_ptr<ProjDataInfo> const& proj_data_info_ptr,
					     DiscretisedDensity<3,float> const* attenuation_image_ptr,
					     const int subset_num,
					     const int num_subsets,
					     const int min_segment,
					     const int max_segment,
					     bool zero_seg0_end_planes,
					     const shared_ptr<ProjData>& multiplicative_proj_data_ptr = NULL);

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
