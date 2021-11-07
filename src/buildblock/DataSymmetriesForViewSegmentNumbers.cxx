/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup projdata 
  \brief Implementations for class stir::DataSymmetriesForViewSegmentNumbers

  \author Kris Thielemans
  \author PARAPET project
*/

#include "stir/DataSymmetriesForViewSegmentNumbers.h"
#include "stir/ViewSegmentNumbers.h"
#include <typeinfo>

using std::vector;

START_NAMESPACE_STIR

DataSymmetriesForViewSegmentNumbers::
~DataSymmetriesForViewSegmentNumbers()
{}

/*! Default implementation always returns \c true. Needs to be overloaded.
 */
bool
DataSymmetriesForViewSegmentNumbers::
blindly_equals(const root_type * const) const
{ 
  return true;
}

bool
DataSymmetriesForViewSegmentNumbers::
operator ==(const root_type& that) const
{ 
  return
    typeid(*this) == typeid(that) &&
    (this == &that ||
     this->blindly_equals(&that)
     );
}

bool
DataSymmetriesForViewSegmentNumbers::
operator !=(const root_type& that) const
{ 
  return !((*this) == that);
}


int
DataSymmetriesForViewSegmentNumbers::num_related_view_segment_numbers(const ViewSegmentNumbers& vs) const
{
  vector<ViewSegmentNumbers> rel_vs;
  get_related_view_segment_numbers(rel_vs, vs);
  return static_cast<int>(rel_vs.size());
}

bool
DataSymmetriesForViewSegmentNumbers::
is_basic(const ViewSegmentNumbers& v_s) const
{
  ViewSegmentNumbers copy = v_s;
  return !find_basic_view_segment_numbers(copy);
}

END_NAMESPACE_STIR
