//
// $Id$: $Date$
//
/*!
  \file 
  \ingroup buildblock  
  \brief inline implementations for the VoxelsOnCartesianGrid class 

  \author Sanida Mustafovic 
  \author Kris Thielemans (with help from Alexey Zverovich)
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/
START_NAMESPACE_TOMO


template<class elemT>
VoxelsOnCartesianGrid<elemT> ::VoxelsOnCartesianGrid()
 : DiscretisedDensityOnCartesianGrid<3,elemT>()
{}

template<class elemT>
VoxelsOnCartesianGrid<elemT>::VoxelsOnCartesianGrid
		      (const Array<3,elemT>& v,
		       const CartesianCoordinate3D<float>& origin,
		       const BasicCoordinate<3,float>& grid_spacing)
		       :DiscretisedDensityOnCartesianGrid<3,elemT>
		       (v.get_index_range(),origin,grid_spacing)
{
  Array<3,elemT>::operator=(v);
}


template<class elemT>
VoxelsOnCartesianGrid<elemT>::VoxelsOnCartesianGrid
		      (const IndexRange<3>& range, 
		       const CartesianCoordinate3D<float>& origin,
		       const BasicCoordinate<3,float>& grid_spacing)
		       :DiscretisedDensityOnCartesianGrid<3,elemT>
		       (range,origin,grid_spacing)
{}

template<class elemT>
CartesianCoordinate3D<float> 
VoxelsOnCartesianGrid<elemT>::get_voxel_size() const
{
  return CartesianCoordinate3D<float>(get_grid_spacing());
}

template<class elemT>
void 
VoxelsOnCartesianGrid<elemT>::set_voxel_size(const BasicCoordinate<3,float>& c) 
{
  set_grid_spacing(c);
}

template<class elemT>
#ifdef TOMO_NO_COVARIANT_RETURN_TYPES
DiscretisedDensity<3,elemT>*
#else
VoxelsOnCartesianGrid<elemT>*
#endif
VoxelsOnCartesianGrid<elemT>::get_empty_discretised_density() const
{
  return get_empty_voxels_on_cartesian_grid();
}



END_NAMESPACE_TOMO
