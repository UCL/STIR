//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief inline implementations for class DataSymmetriesForDensels

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

#include "Densel.h"

START_NAMESPACE_TOMO
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



END_NAMESPACE_TOMO
