//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock 
  \brief 

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
START_NAMESPACE_TOMO


/*! default implementation in terms of get_related_view_segment_numbers, will be slow of course */
int
DataSymmetriesForViewSegmentNumbers::num_related_view_segment_numbers(const ViewSegmentNumbers& vs) const
{
  vector<ViewSegmentNumbers> rel_vs;
  get_related_view_segment_numbers(rel_vs, vs);
  return rel_vs.size();
}

#if 0
/*! default implementation in terms of find_symmetry_operation_to_basic_view_segment_numbers */
bool DataSymmetriesForViewSegmentNumbers::find_basic_view_segment_numbers(ViewSegmentNumbers& vs) const
{
  auto_ptr<SymmetryOperation> sym_op =
    find_symmetry_operation_to_basic_view_segment_numbers(vs);
  return !sym_op->is_trivial();
}
#endif
END_NAMESPACE_TOMO
