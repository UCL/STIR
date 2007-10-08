//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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

  $Date$
  $Revision$
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
  TrivialDataSymmetriesForBins(const shared_ptr<ProjDataInfo>& proj_data_info_ptr);

  virtual 
#ifndef STIR_NO_COVARIANT_RETURN_TYPES
    TrivialDataSymmetriesForBins 
#else
    DataSymmetriesForViewSegmentNumbers
#endif
    * clone() const;

  virtual void
    get_related_bins(vector<Bin>&, const Bin& b,
                      const int min_axial_pos_num, const int max_axial_pos_num,
                      const int min_tangential_pos_num, const int max_tangential_pos_num) const;

  virtual void
    get_related_bins_factorised(vector<AxTangPosNumbers>&, const Bin& b,
                                const int min_axial_pos_num, const int max_axial_pos_num,
                                const int min_tangential_pos_num, const int max_tangential_pos_num) const;

  virtual int
    num_related_bins(const Bin& b) const;

  virtual auto_ptr<SymmetryOperation>
    find_symmetry_operation_from_basic_bin(Bin&) const;

  virtual bool
    find_basic_bin(Bin& b) const;

  virtual bool
    is_basic(const Bin& v_s) const;

  virtual auto_ptr<SymmetryOperation>
    find_symmetry_operation_from_basic_view_segment_numbers(ViewSegmentNumbers&) const;

  virtual void
    get_related_view_segment_numbers(vector<ViewSegmentNumbers>&, const ViewSegmentNumbers&) const;

  virtual int
    num_related_view_segment_numbers(const ViewSegmentNumbers&) const;
  virtual bool
    find_basic_view_segment_numbers(ViewSegmentNumbers&) const;

private:
  virtual bool blindly_equals(const root_type * const) const;
};

END_NAMESPACE_STIR


#endif

