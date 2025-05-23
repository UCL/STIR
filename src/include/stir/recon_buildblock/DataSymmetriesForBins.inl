//
//
/*!

  \file
  \ingroup buildblock

  \brief inline implementations for class stir::DataSymmetriesForBins

  \author Kris Thielemans
  \author PARAPET project


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#include "stir/Bin.h"

START_NAMESPACE_STIR
void
DataSymmetriesForBins::
get_related_bins(std::vector<Bin>& rel_b, const Bin& b) const
{
  get_related_bins(rel_b, b,
                   proj_data_info_ptr->get_min_axial_pos_num(b.segment_num()), 
                   proj_data_info_ptr->get_max_axial_pos_num(b.segment_num()),
                   proj_data_info_ptr->get_min_tangential_pos_num(), 
                   proj_data_info_ptr->get_max_tangential_pos_num());
}


void
DataSymmetriesForBins::
get_related_bins_factorised(std::vector<AxTangPosNumbers>& ax_tang_poss, const Bin& b) const
{
   get_related_bins_factorised(ax_tang_poss, b,
                               proj_data_info_ptr->get_min_axial_pos_num(b.segment_num()), 
                               proj_data_info_ptr->get_max_axial_pos_num(b.segment_num()),
                               proj_data_info_ptr->get_min_tangential_pos_num(), 
                               proj_data_info_ptr->get_max_tangential_pos_num());
}

END_NAMESPACE_STIR
