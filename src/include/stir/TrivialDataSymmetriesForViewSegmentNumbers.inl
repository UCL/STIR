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

  \brief Implementation of inline-methods of class stir::TrivialDataSymmetriesForViewSegmentNumbers

  \author Kris Thielemans

*/

START_NAMESPACE_STIR

DataSymmetriesForViewSegmentNumbers*
TrivialDataSymmetriesForViewSegmentNumbers::
clone() const
{
  return new TrivialDataSymmetriesForViewSegmentNumbers;
}


void
TrivialDataSymmetriesForViewSegmentNumbers::
get_related_view_segment_numbers(std::vector<ViewSegmentNumbers>& all, const ViewSegmentNumbers& v_s) const
{
  all.resize(1);
  all[0] = v_s;
}


int
TrivialDataSymmetriesForViewSegmentNumbers::
num_related_view_segment_numbers(const ViewSegmentNumbers& v_s) const
{
  return 1;
}

bool
TrivialDataSymmetriesForViewSegmentNumbers::
find_basic_view_segment_numbers(ViewSegmentNumbers& v_s) const
{
  return false;
}

bool 
TrivialDataSymmetriesForViewSegmentNumbers::
blindly_equals(const root_type * const) const
{
  return true;
}

END_NAMESPACE_STIR
