//
//
/*!

  \file
  \ingroup projdata
  \brief Declaration of stir::SSRB functions

  \author Kris Thielemans

*/
/*
    Copyright (C) 2002- 2009, Hammersmith Imanet Ltd
    Copyright (C) 2021, 2024 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_SSRB_H__
#define __stir_SSRB_H__

#include "stir/common.h"
#include <string>

START_NAMESPACE_STIR

class ProjData;
class ProjDataInfo;

//! construct new ProjDataInfo that is appropriate for rebinned data
/*!
  \ingroup projdata
  \param in_proj_data_info input projection data information.
  \param num_segments_to_combine how many segments will be combined into 1 output segment.
  \param  num_views_to_combine how many views will be combined in the output
    (i.e. mashing)
  \param num_tangential_poss_to_trim can be used to throw
    away some bins. Half of the bins will be thrown away at each 'side' of a sinogram (see below).
  \param max_in_segment_num_to_process rebinned in_proj_data only upto this segment.
  Default value -1 means 'do all segments'.
  \param num_tof_bins_to_combine can be used to increase TOF mashing.

  The original SSRB algorithm was developed in M.E. Daube-Witherspoon and
  G. Muehllehner, (1987) <i>Treatment of axial data in three-dimensional PET</i>,
  J. Nucl. Med. 28, 171-1724. It essentially ignores the obliqueness of a
  Line of Response and moves data to the axial position in segment 0 such
  that z-resolution on the axis of the scanner is preserved.

  The STIR implementation of SSRB is a generalisation that applies the same
  idea while still allowing preserving some of the obliqueness. For instance,
  for a dataset with 9 segments, SSRB can produce a new dataset with only 3
  segments. This essentially increases the axial compression (or span in CTI
  terminology), see the STIR Glossary on axial compression. In addition, SSRB
  can introduce extra mashing (see the STIR Glossary) of the data, i.e. add
  views together. Finally, it can also be used to combine TOF bins together.

  Here is how to determine which bins are discarded when trimming is used.
  For a certain num_tangential_poss, the range is from
\verbatim
   -(num_tangential_poss/2) to -(num_tangential_poss/2) + num_tangential_poss - 1.
\endverbatim
  The new \c num_tangential_poss is simply set to \c old_num_tangential_poss -
  \a num_tang_poss_to_trim. Note that because of this, if \a num_tang_poss_to_trim is
  negative, more (zero) bins will be added.

  \warning in_proj_data_info has to be (at least) of type ProjDataInfoCylindrical
  \warning This function can only handle in_proj_data_info where all segments have
      identical 'num_segments_to_combine'. So it cannot handle standard
      GE Advance data.
  \todo get rid of both restrictions flagged as warnings in the documentation for this function.
  \todo rename to something much more general than \c SSRB
*/
ProjDataInfo* SSRB(const ProjDataInfo& in_proj_data_info,
                   const int num_segments_to_combine,
                   const int num_views_to_combine = 1,
                   const int num_tangential_poss_to_trim = 0,
                   const int max_in_segment_num_to_process = -1,
                   const int num_tof_bins_to_combine = 1);

//! Perform Single Slice Rebinning and write output to file
/*!
  \ingroup projdata
  \param output_filename filename to write output projection data (will be in
  Interfile format)
  \param in_projdata input data
  \param num_segments_to_combine how many segments will be combined into 1 output segment.
  \param max_in_segment_num_to_process rebinned in_proj_data only upto this segment.
  Default value -1 means 'do all segments'.
  \param do_normalisation (default true) wether to normalise the output sinograms
  corresponding to how many input sinograms contribute to them.
  \param num_tof_bins_to_combine defaults to 1, so TOF bins are not combined.

  \see SSRB(ProjData& out_projdata,
     const ProjData& in_projdata,
     const bool do_normalisation = true)
  */
void SSRB(const std::string& output_filename,
          const ProjData& in_projdata,
          const int num_segments_to_combine,
          const int num_views_to_combine = 1,
          const int num_tangential_poss_to_trim = 0,
          const bool do_normalisation = true,
          const int max_in_segment_num_to_process = -1,
          const int num_tof_bins_to_combine = 1);

//! Perform Single Slice Rebinning and write output to ProjData
/*!
  \ingroup projdata
  \param out_projdata Output projection data. Its projection_data_info is used to
  determine output characteristics. Data will be 'put' in here using
  ProjData::set_sinogram().
  \param in_projdata input data
  \param do_normalisation (default true) wether to normalise the output sinograms
  corresponding to how many (ignoring TOF) input sinograms contribute to them.
  Note that we do not normalise according to the number of TOF bins.
  This is because the projectors will take the width of the TOF bin
  properly into account (by integration over the TOF kernel). (In contrast, in spatial
  direction, projectors are outputting "normalised" data, i.e. corresponding to the
  line integral).


  \warning \a in_projdata has to be (at least) of type ProjDataInfoCylindrical

  \see SSRB(const ProjDataInfo& in_proj_data_info,
     const int num_segments_to_combine,
     const int num_views_to_combine,
     const int num_tang_poss_to_trim,
     const int max_in_segment_num_to_process,
     const int num_tof_bins_to_combine
     ) for information on the rebinning.
*/
void SSRB(ProjData& out_projdata, const ProjData& in_projdata, const bool do_normalisation = true);

END_NAMESPACE_STIR

#endif
