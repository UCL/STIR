//
// $Id$
//
/*!
  \file
  \ingroup projdata 
  \brief Implementations for class DataSymmetriesForViewSegmentNumbers

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/DataSymmetriesForViewSegmentNumbers.h"
#include "stir/ViewSegmentNumbers.h"

START_NAMESPACE_STIR

DataSymmetriesForViewSegmentNumbers::
~DataSymmetriesForViewSegmentNumbers()
{}

int
DataSymmetriesForViewSegmentNumbers::num_related_view_segment_numbers(const ViewSegmentNumbers& vs) const
{
  vector<ViewSegmentNumbers> rel_vs;
  get_related_view_segment_numbers(rel_vs, vs);
  return rel_vs.size();
}

bool
DataSymmetriesForViewSegmentNumbers::
is_basic(const ViewSegmentNumbers& v_s) const
{
  ViewSegmentNumbers copy = v_s;
  return !find_basic_view_segment_numbers(copy);
}

END_NAMESPACE_STIR
