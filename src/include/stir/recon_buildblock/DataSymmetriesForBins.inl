//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief inline implementations for class stir::DataSymmetriesForBins

  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
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

#include "stir/Bin.h"

START_NAMESPACE_STIR
void
DataSymmetriesForBins::
    get_related_bins(vector<Bin>& rel_b, const Bin& b) const
{
  get_related_bins(rel_b, b,
                   proj_data_info_ptr->get_min_axial_pos_num(b.segment_num()), 
                   proj_data_info_ptr->get_max_axial_pos_num(b.segment_num()),
                   proj_data_info_ptr->get_min_tangential_pos_num(), 
                   proj_data_info_ptr->get_max_tangential_pos_num());
}


void
DataSymmetriesForBins::
    get_related_bins_factorised(vector<AxTangPosNumbers>& ax_tang_poss, const Bin& b) const
{
   get_related_bins_factorised(ax_tang_poss, b,
                               proj_data_info_ptr->get_min_axial_pos_num(b.segment_num()), 
                               proj_data_info_ptr->get_max_axial_pos_num(b.segment_num()),
                               proj_data_info_ptr->get_min_tangential_pos_num(), 
                               proj_data_info_ptr->get_max_tangential_pos_num());
}

END_NAMESPACE_STIR
