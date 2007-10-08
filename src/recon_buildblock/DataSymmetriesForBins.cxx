//
// $Id$
//
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
/*!

  \file
  \ingroup symmetries

  \brief implementations for class stir::DataSymmetriesForBins

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/

#include "stir/recon_buildblock/DataSymmetriesForBins.h"
#include "stir/Bin.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/BasicCoordinate.h"
#include "stir/recon_buildblock/SymmetryOperation.h"

START_NAMESPACE_STIR

DataSymmetriesForBins::
~DataSymmetriesForBins()
{}

DataSymmetriesForBins::
DataSymmetriesForBins(const shared_ptr<ProjDataInfo>& proj_data_info_ptr)
: proj_data_info_ptr(proj_data_info_ptr)
{}

bool
DataSymmetriesForBins::
blindly_equals(const root_type * const sym_ptr) const
{ 
  if (!base_type::blindly_equals(sym_ptr))
    return false;

  return 
    *this->proj_data_info_ptr == 
    *static_cast<const self_type&>(*sym_ptr).proj_data_info_ptr;
}


/*! default implementation in terms of get_related_bins, will be slow of course */
int
DataSymmetriesForBins::num_related_bins(const Bin& b) const
{
  vector<Bin> rel_b;
  get_related_bins(rel_b, b);
  return rel_b.size();
}

/*! default implementation in terms of find_symmetry_operation_from_basic_bin */
bool DataSymmetriesForBins::find_basic_bin(Bin& b) const
{
  auto_ptr<SymmetryOperation> sym_op =
    find_symmetry_operation_from_basic_bin(b);
  return sym_op->is_trivial();
}


bool
DataSymmetriesForBins::
is_basic(const Bin& b) const
{
  Bin copy = b;
  return !find_basic_bin(copy);
}

/*! default implementation in terms of get_related_bins_factorised */
void
DataSymmetriesForBins::
get_related_bins(vector<Bin>& rel_b, const Bin& b,
                 const int min_axial_pos_num, const int max_axial_pos_num,
                 const int min_tangential_pos_num, const int max_tangential_pos_num) const
{
#ifndef NDEBUG
  Bin bin_copy = b;
  assert(!find_basic_bin(bin_copy));
#endif

  vector<ViewSegmentNumbers> vs;
  vector<AxTangPosNumbers> ax_tang_poss;

  get_related_bins_factorised(ax_tang_poss, b,
                              min_axial_pos_num, max_axial_pos_num,
                              min_tangential_pos_num, max_tangential_pos_num);

  get_related_view_segment_numbers(vs, ViewSegmentNumbers(b.view_num(), b.segment_num()));

  rel_b.reserve(ax_tang_poss.size() * vs.size());
  rel_b.resize(0);

  for (
#ifdef _MSC_VER
    // VC bug work-around...
       std::
#endif
         vector<ViewSegmentNumbers>::const_iterator view_seg_ptr = vs.begin();
       view_seg_ptr != vs.end();
       ++view_seg_ptr)
  {
    for (
#ifdef _MSC_VER
    // VC bug work-around...
         std::
#endif
           vector<AxTangPosNumbers>::const_iterator ax_tang_pos_ptr = ax_tang_poss.begin();
         ax_tang_pos_ptr != ax_tang_poss.end();
         ++ax_tang_pos_ptr)
    {
      rel_b.push_back(Bin(view_seg_ptr->segment_num(), view_seg_ptr->view_num(), 
                          (*ax_tang_pos_ptr)[1], (*ax_tang_pos_ptr)[2]));
    }
  }
}

auto_ptr<SymmetryOperation>
DataSymmetriesForBins::
find_symmetry_operation_from_basic_view_segment_numbers(ViewSegmentNumbers& vs) const
{
  Bin bin(vs.segment_num(), vs.view_num(),0,0);
#ifndef NDEBUG
  Bin bin_copy = bin;
#endif

  auto_ptr<SymmetryOperation> sym_op =
    find_symmetry_operation_from_basic_bin(bin);
  vs.segment_num() = bin.segment_num();
  vs.view_num() = bin.view_num();

#ifndef NDEBUG
  sym_op->transform_bin_coordinates(bin);
  assert(bin == bin_copy);
#endif

  return sym_op;
}

END_NAMESPACE_STIR
