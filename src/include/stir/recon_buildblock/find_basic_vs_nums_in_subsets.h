//
//
/*
    Copyright (C) 2011, Hammersmith Imanet Ltd
    Copyright (C) 2015, University College London
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

  \brief Implementation for stir::detail::find_basic_vs_nums_in_subset

  \author Kris Thielemans
*/


#include "stir/ViewSegmentNumbers.h"
#include <vector>

START_NAMESPACE_STIR

class ProjDataInfo;
class DataSymmetriesForViewSegmentNumbers;

namespace detail 
{

  /*!
    \brief a helper function to find which view/segments are in a subset
    \ingroup recon_buildblock

    This function is used by projectors and distributable_computation etc
    to construct a list of view/segments that are in a subset, and which are
    "basic" w.r.t the symmetries.
  */
  std::vector<ViewSegmentNumbers> 
  find_basic_vs_nums_in_subset(const ProjDataInfo& proj_data_info,
                               const DataSymmetriesForViewSegmentNumbers& symmetries, 
                               const int min_segment_num, const int max_segment_num,
                               const int subset_num, const int num_subsets);

}

END_NAMESPACE_STIR
