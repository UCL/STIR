//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief inline implementations for class DataSymmetriesForBins

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

#include "Bin.h"

START_NAMESPACE_TOMO
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

END_NAMESPACE_TOMO
