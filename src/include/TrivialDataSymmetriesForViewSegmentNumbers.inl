//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementation of inline-methods of class TrivialDataSymmetriesForViewSegmentNumbers

  \author Kris Thielemans

   $Date$
   $Revision$
*/

START_NAMESPACE_TOMO

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

END_NAMESPACE_TOMO
