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
