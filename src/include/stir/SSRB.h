//
// $Id$
//
/*!

  \file
  \ingroup projdata
  \brief Declaration of SSRB functions

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_SSRB_H__
#define __stir_SSRB_H__

#include "stir/common.h"
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_STIR

class ProjData;
class ProjDataInfo;

//! construct new ProjDataInfo that is appropriate for rebinned data
/*!
  \ingroup projdata
  \param in_proj_data_info input projection data information.
  \param num_segments_to_combine how many segments will be combined into 1 output segment.
  \param max_in_segment_num_to_process rebinned in_proj_data only upto this segment.
  Default value -1 means 'do all segments'.

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
views together.


  \warning in_proj_data_info has to be (at least) of type ProjDataInfoCylindrical
  \warning This function can only handle in_proj_data_info where all segments have 
      identical 'num_segments_to_combine'. So it cannot handle standard 
      GE Advance data.
  \todo get rid of both restrictions flagged as warnings in the documentation for this function.
*/
ProjDataInfo *
SSRB(const ProjDataInfo& in_proj_data_info,
     const int num_segments_to_combine,
     const int num_views_to_combine = 1,
     const int max_in_segment_num_to_process=-1
     );

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

  \see SSRB(const ProjDataInfo& in_proj_data_info,
     const int num_segments_to_combine,
     const int max_in_segment_num_to_process
     ) for restrictions
  */
void 
SSRB(const string& output_filename,
     const ProjData& in_projdata,
     const int num_segments_to_combine,
     const int num_views_to_combine = 1,
     const bool do_normalisation = true,
     const int max_segment_num_to_process = -1
     );

//! Perform Single Slice Rebinning and write output to ProjData
/*! 
  \ingroup projdata
  \param out_projdata Output projection data. Its projection_data_info is used to 
  determine output characteristics. Data will be 'put' in here using 
  ProjData::set_sinogram().
  \param in_projdata input data
  \param do_normalisation (default true) wether to normalise the output sinograms 
  corresponding to how many input sinograms contribute to them.
  
  \warning in_proj_data_info has to be (at least) of type ProjDataInfoCylindrical
*/  
void 
SSRB(ProjData& out_projdata,
     const ProjData& in_projdata,
     const bool do_normalisation = true
     );

END_NAMESPACE_STIR

#endif

