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

DataSymmetriesForViewSegmentNumbers::
DataSymmetriesForViewSegmentNumbers(const shared_ptr<ProjDataInfo>& proj_data_info_ptr/*,
                        const shared_ptr<DiscretisedDensity>& image_info*/
)
  : proj_data_info_ptr(proj_data_info_ptr)/*,
    image_info(image_info)*/
{}

/*! default implementation in terms of get_related_view_segment_numbers, will be slow of course */
int
DataSymmetriesForViewSegmentNumbers::num_related_view_segment_numbers(const ViewSegmentNumbers& vs) const
{
  vector<ViewSegmentNumbers> rel_vs;
  get_related_view_segment_numbers(rel_vs, vs);
  return rel_vs.size();
}

/*! default implementation in terms of find_symmetry_operation_to_basic_view_segment_numbers */
bool DataSymmetriesForViewSegmentNumbers::find_basic_view_segment_numbers(ViewSegmentNumbers& vs) const
{
  auto_ptr<SymmetryOperation> sym_op =
    find_symmetry_operation_to_basic_view_segment_numbers(vs);
  return sym_op->is_trivial();
}

END_NAMESPACE_TOMO
