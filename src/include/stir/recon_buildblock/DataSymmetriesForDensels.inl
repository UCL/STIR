//
//
/*!

  \file
  \ingroup buildblock

  \brief inline implementations for class stir::DataSymmetriesForDensels

  \author Kris Thielemans
  
*/
/*
    Copyright (C) 2001- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/Densel.h"

START_NAMESPACE_STIR
void
DataSymmetriesForDensels::
    get_related_densels(vector<Densel>& rel_b, const Densel& b) const
{
  get_related_densels(rel_b, b,
                   proj_data_info_ptr->get_min_axial_pos_num(b.segment_num()), 
                   proj_data_info_ptr->get_max_axial_pos_num(b.segment_num()),
                   proj_data_info_ptr->get_min_tangential_pos_num(), 
                   proj_data_info_ptr->get_max_tangential_pos_num());
}



END_NAMESPACE_STIR
