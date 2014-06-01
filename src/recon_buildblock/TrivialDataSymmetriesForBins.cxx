//
//
/*
    Copyright (C) 2003- 2007, Hammersmith Imanet Ltd
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
  \brief non-inline implementations for class 
         stir::TrivialDataSymmetriesForBins

  \author Kris Thielemans

*/
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/recon_buildblock/SymmetryOperation.h"
#include "stir/ViewSegmentNumbers.h"

using std::vector;
using std::auto_ptr;

START_NAMESPACE_STIR
TrivialDataSymmetriesForBins::
TrivialDataSymmetriesForBins
(
 const shared_ptr<ProjDataInfo>& proj_data_info_ptr)
  : DataSymmetriesForBins(proj_data_info_ptr)
{
}


#ifndef STIR_NO_COVARIANT_RETURN_TYPES
    TrivialDataSymmetriesForBins *
#else
    DataSymmetriesForViewSegmentNumbers *
#endif
TrivialDataSymmetriesForBins::
clone() const
{
  return new TrivialDataSymmetriesForBins(*this);
}

int
TrivialDataSymmetriesForBins::num_related_bins(const Bin& b) const
{
  return 1;
}

bool TrivialDataSymmetriesForBins::find_basic_bin(Bin& b) const
{
  return false;
}


bool
TrivialDataSymmetriesForBins::
is_basic(const Bin& b) const
{
  return true;
}


void
TrivialDataSymmetriesForBins::
get_related_bins_factorised(vector<AxTangPosNumbers>& axtan_pos_nums, const Bin& b,
                 const int min_axial_pos_num, const int max_axial_pos_num,
                 const int min_tangential_pos_num, const int max_tangential_pos_num) const
{
  if (b.axial_pos_num() >= min_axial_pos_num &&
      b.axial_pos_num() <= max_axial_pos_num &&
      b.tangential_pos_num() >= min_tangential_pos_num &&
      b.tangential_pos_num() <= max_tangential_pos_num)
    {
      axtan_pos_nums.resize(1);
      axtan_pos_nums[0] = 
	AxTangPosNumbers(b.axial_pos_num(),
			 b.tangential_pos_num());
    }
  else
    {
      axtan_pos_nums.resize(0);
    }
}

void
TrivialDataSymmetriesForBins::
get_related_bins(vector<Bin>& rel_b, const Bin& b,
                 const int min_axial_pos_num, const int max_axial_pos_num,
                 const int min_tangential_pos_num, const int max_tangential_pos_num) const
{
  if (b.axial_pos_num() >= min_axial_pos_num &&
      b.axial_pos_num() <= max_axial_pos_num &&
      b.tangential_pos_num() >= min_tangential_pos_num &&
      b.tangential_pos_num() <= max_tangential_pos_num)
    {
      rel_b.resize(1);
      rel_b[0] = b;
    }
  else
    {
      rel_b.resize(0);
    }
}

auto_ptr<SymmetryOperation>
TrivialDataSymmetriesForBins::
find_symmetry_operation_from_basic_bin(Bin&) const
{
  return auto_ptr<SymmetryOperation>(new TrivialSymmetryOperation);
}

auto_ptr<SymmetryOperation>
TrivialDataSymmetriesForBins::
find_symmetry_operation_from_basic_view_segment_numbers(ViewSegmentNumbers& vs) const
{
  return auto_ptr<SymmetryOperation>(new TrivialSymmetryOperation);
}

void
TrivialDataSymmetriesForBins::
get_related_view_segment_numbers(vector<ViewSegmentNumbers>& all, const ViewSegmentNumbers& v_s) const
{
  all.resize(1);
  all[0] = v_s;
}


int
TrivialDataSymmetriesForBins::
num_related_view_segment_numbers(const ViewSegmentNumbers&) const
{
  return 1;
}

bool
TrivialDataSymmetriesForBins::
find_basic_view_segment_numbers(ViewSegmentNumbers&) const
{
  return false;
}

bool 
TrivialDataSymmetriesForBins::
blindly_equals(const root_type * const) const
{
  return true;
}

END_NAMESPACE_STIR
