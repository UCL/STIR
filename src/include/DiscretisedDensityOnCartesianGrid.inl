//
// $Id$: $Date$
//
/*!
  \file 
 
  \brief  inline implementations for DiscretisedDensityOnCartesianGrid

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author (help from Alexey Zverovich)
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/


START_NAMESPACE_TOMO

template<int num_dimensions, typename elemT>
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
DiscretisedDensityOnCartesianGrid()
: DiscretisedDensity<num_dimensions, elemT>(),grid_spacing()
{
#ifndef TOMO_NO_NAMESPACES
  std::fill(grid_spacing.begin(), grid_spacing.end(), 0.F);
#else
  // hopefully your compiler understands this.
  // It attempts to avoid conflicts with Array::fill
  ::fill(grid_spacing.begin(), grid_spacing.end(), 0.F);
#endif
}

template<int num_dimensions, typename elemT>
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
DiscretisedDensityOnCartesianGrid
(const IndexRange<num_dimensions>& range_v, 
 const CartesianCoordinate3D<float>& origin_v,
 const BasicCoordinate<num_dimensions,float>& grid_spacing_v)
  : DiscretisedDensity<num_dimensions, elemT>(range_v,origin_v),
    grid_spacing(grid_spacing_v)
{}

template<int num_dimensions, typename elemT>
const BasicCoordinate<num_dimensions,float>& 
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
get_grid_spacing() const
{ return grid_spacing; }

template<int num_dimensions, typename elemT>
void 
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
set_grid_spacing(const BasicCoordinate<num_dimensions,float>& grid_spacing_v)
{
  grid_spacing = grid_spacing_v;
}


template<int num_dimensions, typename elemT>
int
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
get_min_z() const
{ return get_min_index();} 


template<int num_dimensions, typename elemT>
int
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
get_min_y() const
{ return get_length()==0 ? 0 : (*this)[get_min_z()].get_min_index(); }

template<int num_dimensions, typename elemT>
int
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
get_min_x() const
{ return get_length()==0 ? 0 : (*this)[get_min_z()][get_min_y()].get_min_index(); }


template<int num_dimensions, typename elemT>
int 
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
get_x_size() const
{ return  get_length()==0 ? 0 : (*this)[get_min_z()][get_min_y()].get_length(); }

template<int num_dimensions, typename elemT>
int
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
get_y_size() const
{ return get_length()==0 ? 0 : (*this)[get_min_z()].get_length();}

template<int num_dimensions, typename elemT>
int
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
get_z_size() const
{ return get_length(); }


template<int num_dimensions, typename elemT>
int
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
get_max_x() const
{ return get_length()==0 ? 0 : (*this)[get_min_z()][get_min_y()].get_max_index();}

template<int num_dimensions, typename elemT>
int
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
get_max_y() const
{ return get_length()==0 ? 0 : (*this)[get_min_z()].get_max_index();}

template<int num_dimensions, typename elemT>
int
DiscretisedDensityOnCartesianGrid<num_dimensions, elemT>::
get_max_z() const
{ return get_max_index(); }


END_NAMESPACE_TOMO					 
