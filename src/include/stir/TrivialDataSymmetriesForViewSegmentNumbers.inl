//
// $Id$
//
/*
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
  \ingroup projdata

  \brief Implementation of inline-methods of class stir::TrivialDataSymmetriesForViewSegmentNumbers

  \author Kris Thielemans

   $Date$
   $Revision$
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
get_related_view_segment_numbers(vector<ViewSegmentNumbers>& all, const ViewSegmentNumbers& v_s) const
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
