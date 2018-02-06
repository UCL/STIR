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

#include "stir/recon_buildblock/find_basic_vs_nums_in_subsets.h"
#include "stir/DataSymmetriesForViewSegmentNumbers.h"
#include "stir/ProjDataInfo.h"
#include <vector>

START_NAMESPACE_STIR

namespace detail 
{

  std::vector<ViewSegmentNumbers> 
  find_basic_vs_nums_in_subset(const ProjDataInfo& proj_data_info,
                               const DataSymmetriesForViewSegmentNumbers& symmetries, 
                               const int min_segment_num, const int max_segment_num,
                               const int subset_num, const int num_subsets)
  {
    std::vector<ViewSegmentNumbers> vs_nums_to_process;
    for (int segment_num = min_segment_num; segment_num <= max_segment_num; segment_num++)
      {
        for (int view = proj_data_info.get_min_view_num() + subset_num; 
             view <= proj_data_info.get_max_view_num(); 
             view += num_subsets)
          {
            const ViewSegmentNumbers view_segment_num(view, segment_num);

            if (!symmetries.is_basic(view_segment_num))
              continue;

            vs_nums_to_process.push_back(view_segment_num);
      
#ifndef NDEBUG
            // test if symmetries didn't take us out of the segment range
            std::vector<ViewSegmentNumbers> rel_vs;
            symmetries.get_related_view_segment_numbers(rel_vs, view_segment_num);
            for (std::vector<ViewSegmentNumbers>::const_iterator iter = rel_vs.begin(); iter!= rel_vs.end(); ++iter)
              {
                assert(iter->segment_num() >= min_segment_num);
                assert(iter->segment_num() <= max_segment_num);
              }
#endif
          }
      }
    return vs_nums_to_process;
  }

}

END_NAMESPACE_STIR
