//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementations of inline functions of class SegmentByView

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

START_NAMESPACE_TOMO

template <typename elemT>
int
SegmentByView<elemT>::get_num_views() const
{
  return get_length();
}

template <typename elemT>
int 
SegmentByView<elemT>::get_min_view_num() const
{
  return get_min_index();
}

template <typename elemT>
int 
SegmentByView<elemT>::get_max_view_num() const
{
  return get_max_index();
}

template <typename elemT>
int
SegmentByView<elemT>::get_num_axial_poss() const
 {
     return get_length()==0 ? 0 : (*this)[get_min_view_num()].get_length();
 }

template <typename elemT>
int
SegmentByView<elemT>::get_min_axial_pos_num() const
{
   return get_length()==0 ? 0 : (*this)[get_min_view_num()].get_min_index();
}

template <typename elemT>
int
SegmentByView<elemT>::get_max_axial_pos_num() const
{
 return get_length()==0 ? 0 : (*this)[get_min_view_num()].get_max_index();
}

template <typename elemT>
int 
SegmentByView<elemT>::get_num_tangential_poss() const
{
  return get_length()==0 ? 0 : (*this)[get_min_view_num()][get_min_axial_pos_num()].get_length();
}

template <typename elemT>
int
SegmentByView<elemT>::get_min_tangential_pos_num() const
{
return get_length()==0 ? 0 : (*this)[get_min_view_num()][get_min_axial_pos_num()].get_min_index();
}

template <typename elemT>
int 
SegmentByView<elemT>::get_max_tangential_pos_num()const
{
return get_length()==0 ? 0 : (*this)[get_min_view_num()][get_min_axial_pos_num()].get_max_index();
}

template <typename elemT>
SegmentByView<elemT>::StorageOrder 
SegmentByView<elemT>::get_storage_order() const
{ return StorageByView; }

template <typename elemT>
Viewgram<elemT> 
SegmentByView<elemT>::get_viewgram(int view_num) const
{ 
  return Viewgram<elemT>(Array<3,elemT>::operator[](view_num), 
    proj_data_info_ptr->clone(), view_num, 
    get_segment_num()); }

template <typename elemT>
void 
SegmentByView<elemT>::set_sinogram(const Sinogram<elemT> &s)
{ set_sinogram(s, s.get_axial_pos_num()); }

template <typename elemT>
void 
SegmentByView<elemT>::set_viewgram(const Viewgram<elemT> &v/*, int view_num*/)
{ Array<3,elemT>::operator[](v.get_view_num()) = v; }

END_NAMESPACE_TOMO
