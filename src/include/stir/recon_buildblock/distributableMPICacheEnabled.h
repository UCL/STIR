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

#ifndef __stir_recon_buildblock_DISTRIBUTABLEMPI_H__
#define __stir_recon_buildblock_DISTRIBUTABLEMPI_H__

/*!
  \file
  \ingroup distributable

  \brief Declaration of the main function that performs parallel processing

  \author Alexey Zverovich
  \author Kris Thielemans
  \author Matthew Jacobson
  \author PARAPET project
  \author Tobias Beisel

  $Date$
  $Revision$
*/
#include "stir/recon_buildblock/distributable.h"

START_NAMESPACE_STIR

class DistributedCachingInformation;


//!@{
//! \ingroup distributable


/*!
  \brief This function essentially implements a loop over segments and all views in the current subset in the parallel case

  This provides the same functionality as distributable_computation(), but enables caching of
  RelatedViewgrams such that they don't need to be sent multiple times.

  \warning Do not call this function directly. Use distributable_computation() instead.
  \todo Merge this functionality into distributable_computation()

  \internal
 */
void distributable_computation_cache_enabled(
                                             const shared_ptr<ForwardProjectorByBin>& forward_projector_ptr,
                                             const shared_ptr<BackProjectorByBin>& back_projector_ptr,
                                             const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_ptr,
                                             DiscretisedDensity<3,float>* output_image_ptr,
                                             const DiscretisedDensity<3,float>* input_image_ptr,
                                             const shared_ptr<ProjData>& proj_data_sptr, 
                                             const bool read_from_proj_data,
                                             int subset_num, int num_subsets,
                                             int min_segment_num, int max_segment_num,
                                             bool zero_seg0_end_planes,
                                             double*  double_out_ptr,
                                             const shared_ptr<ProjData>& additive_binwise_correction,
                                             const shared_ptr<BinNormalisation> normalise_sptr,
                                             const double start_time_of_frame,
                                             const double end_time_of_frame,
                                             RPC_process_related_viewgrams_type * RPC_process_related_viewgrams, 
                                             DistributedCachingInformation* caching_info_ptr
                                             );


void test_image_estimate(shared_ptr<stir::DiscretisedDensity<3, float> > input_image_ptr);
//!@}

END_NAMESPACE_STIR

#endif // __stir_recon_buildblock_DISTRIBUTABLEMPI_H__

