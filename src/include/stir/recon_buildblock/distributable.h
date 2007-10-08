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

#ifndef __stir_recon_buildblock_DISTRIBUTABLE_H__
#define __stir_recon_buildblock_DISTRIBUTABLE_H__

/*!
  \file
  \ingroup distributable

  \brief Declaration of the main function that performs parallel processing

  \author Alexey Zverovich
  \author Kris Thielemans
  \author Matthew Jacobson
  \author PARAPET project

   $Date$
   $Revision$
*/
#include "stir/common.h"

START_NAMESPACE_STIR

template <typename T> class shared_ptr;
template <typename elemT> class RelatedViewgrams;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
class ProjData;
class ProjDataInfo;
class DataSymmetriesForViewSegmentNumbers;
class ForwardProjectorByBin;
class BackProjectorByBin;

//!@{
//! \ingroup distributable

//! typedef for callback functions for distributable_computation()
/*! Pointers will be NULL when they are not to be used by the callback function.

    \a count and \a count2 are normally incremental counters that accumulate over the loop
    in distributable_computation().

    \warning The data in *measured_viewgrams_ptr are allowed to be overwritten, but the new data 
    will not be used. 
*/
typedef  void RPC_process_related_viewgrams_type (
						  const shared_ptr<ForwardProjectorByBin>& forward_projector_sptr,
						  const shared_ptr<BackProjectorByBin>& back_projector_sptr,
						  DiscretisedDensity<3,float>* output_image_ptr, 
						  const DiscretisedDensity<3,float>* input_image_ptr, 
						  RelatedViewgrams<float>* measured_viewgrams_ptr,
						  int& count, int& count2, float* log_likelihood_ptr,
						  const RelatedViewgrams<float>* additive_binwise_correction_ptr);

/*!
  \brief This function essentially implements a loop over segments and all views in the current subset.
  
  Output is in output_image_ptr and in float_out_ptr (but only if they are not NULL).
  What the output is, is really determined by the call-back function
  RPC_process_related_viewgrams.

  A parallel version of this function exists which distributes the computation over the slaves.

  Subsets are currently defined on views. A particular \a subset_num contains all views
  which are symmetry related to 
  \code 
  proj_data_ptr->min_view_num()+subset_num + n*num_subsets
  \endcode
  for n=0,1,,.. \c and for which the above view_num is 'basic' (for some segment_num in the range).

  Symmetries are determined by using the 3rd argument to set_projectors_and_symmetries().

  \param output_image_ptr will store the output image if non-zero.
  \param input_image_ptr input when non-zero.
  \param proj_data_ptr input projection data
  \param read_from_proj_data if true, the \a measured_viewgrams_ptr argument of the call_back function 
         will be constructed using ProjData::get_related_viewgrams, otherwise 
         ProjData::get_empty_related_viewgrams is used.
  \param subset_num the number of the current subset (see above). Should be between 0 and num_subsets-1.
  \param num_subsets the number of subsets to consider. 1 will process all data.
  \param min_segment_num Minimum segment_num to process.
  \param max_segment_num Maximum segment_num to process.
  \param zero_seg0_end_planes if true, the end planes for segment_num=0 in measured_viewgrams_ptr
         (and additive_binwise_correction_ptr when applicable) will be set to 0.
  \param float_out_ptr a potential float output parameter for the call back function.
  \param additive_binwise_correction Additional input projection data (when the shared_ptr is not 0).
  \param RPC_process_related_viewgrams function that does the actual work.
  
  \warning There is NO check that the resulting subsets are balanced.

  \warning The function assumes that \a min_segment_num, \a max_segment_num are such that
  symmetries map this range onto itself (i.e. no segment_num is obtained outside the range). 
  This usually means that \a min_segment_num = -\a max_segment_num. This assumption is checked with 
  assert().

  \todo The subset-scheme should be moved somewhere else (a Subset class?).

 */
void distributable_computation(
			       const shared_ptr<ForwardProjectorByBin>& forward_projector_sptr,
			       const shared_ptr<BackProjectorByBin>& back_projector_sptr,
			       const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_sptr,
			       DiscretisedDensity<3,float>* output_image_ptr,
				    const DiscretisedDensity<3,float>* input_image_ptr,
				    const shared_ptr<ProjData>& proj_data_ptr,
                                    const bool read_from_proj_data,
				    int subset_num, int num_subsets,
				    int min_segment_num, int max_segment_num,
				    bool zero_seg0_end_planes,
				    float* float_out_ptr,
				    const shared_ptr<ProjData>& additive_binwise_correction,
                                    RPC_process_related_viewgrams_type * RPC_process_related_viewgrams);


//!@}

END_NAMESPACE_STIR

#endif // __stir_recon_buildblock_DISTRIBUTABLE_H__

