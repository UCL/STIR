//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup projdata

  \brief Declaration of class stir::TrivialDataSymmetriesForViewSegmentNumbers

  \author Kris Thielemans


*/
#ifndef __TrivialDataSymmetriesForViewSegmentNumbers_H__
#define __TrivialDataSymmetriesForViewSegmentNumbers_H__

#include "stir/DataSymmetriesForViewSegmentNumbers.h"

START_NAMESPACE_STIR

/*!
  \brief A class for encoding/finding NO symmetries. Works only on
  ViewSegmentNumbers (instead of Bin).

  This class is mainly useful if you need a DataSymmetriesForViewSegmentNumbers
  object (e.g. for RelatedViewgrams), but do not need/have projectors.

*/
class TrivialDataSymmetriesForViewSegmentNumbers : public DataSymmetriesForViewSegmentNumbers
{
public:
  inline DataSymmetriesForViewSegmentNumbers* clone() const override;

#if 0
  // TODO
  //! returns the range of the indices for basic view/segments
  virtual ViewSegmentIndexRange
    get_basic_view_segment_index_range() const;
#endif

  inline void get_related_view_segment_numbers(std::vector<ViewSegmentNumbers>&, const ViewSegmentNumbers& v_s) const override;

  inline int num_related_view_segment_numbers(const ViewSegmentNumbers& v_s) const override;

  /*! \brief given an arbitrary view/segment, find the basic view/segment

  in this class, \a v_s is unchanged, and the return value is always false.
  'v_s' is changed (i.e. it was NOT a basic view/segment).
  */
  inline bool find_basic_view_segment_numbers(ViewSegmentNumbers& v_s) const override;

private:
  bool blindly_equals(const root_type* const) const override;
};

END_NAMESPACE_STIR

#include "stir/TrivialDataSymmetriesForViewSegmentNumbers.inl"

#endif
