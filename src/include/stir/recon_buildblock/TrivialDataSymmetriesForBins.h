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

  \brief Declaration of class stir::TrivialDataSymmetriesForBins

  \author Kris Thielemans

*/
#ifndef __stir_recon_buildblock_TrivialDataSymmetriesForBins_H__
#define __stir_recon_buildblock_TrivialDataSymmetriesForBins_H__

#include "stir/recon_buildblock/DataSymmetriesForBins.h"

START_NAMESPACE_STIR


/*!
  \ingroup symmetries
  \brief A class derived from DataSymmetriesForBins that says that there are
  no symmetries at all.

*/
class TrivialDataSymmetriesForBins : public DataSymmetriesForBins
{
public:
  TrivialDataSymmetriesForBins(const shared_ptr<const ProjDataInfo>& proj_data_info_ptr);

  
#ifndef STIR_NO_COVARIANT_RETURN_TYPES
    TrivialDataSymmetriesForBins 
#else
    DataSymmetriesForViewSegmentNumbers
#endif
    * clone() const override;

  void
    get_related_bins(std::vector<Bin>&, const Bin& b,
                      const int min_axial_pos_num, const int max_axial_pos_num,
                      const int min_tangential_pos_num, const int max_tangential_pos_num) const override;

  void
    get_related_bins_factorised(std::vector<AxTangPosNumbers>&, const Bin& b,
                                const int min_axial_pos_num, const int max_axial_pos_num,
                                const int min_tangential_pos_num, const int max_tangential_pos_num) const override;

  int
    num_related_bins(const Bin& b) const override;

  unique_ptr<SymmetryOperation>
    find_symmetry_operation_from_basic_bin(Bin&) const override;

  bool
    find_basic_bin(Bin& b) const override;

  bool
    is_basic(const Bin& v_s) const override;

  unique_ptr<SymmetryOperation>
    find_symmetry_operation_from_basic_view_segment_numbers(ViewSegmentNumbers&) const override;

  void
    get_related_view_segment_numbers(std::vector<ViewSegmentNumbers>&, const ViewSegmentNumbers&) const override;

  int
    num_related_view_segment_numbers(const ViewSegmentNumbers&) const override;
  bool
    find_basic_view_segment_numbers(ViewSegmentNumbers&) const override;

private:
  bool blindly_equals(const root_type * const) const override;
};

END_NAMESPACE_STIR


#endif

