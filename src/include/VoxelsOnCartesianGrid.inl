//
// $Id$
//
/*!
  \file 
  \ingroup buildblock  
  \brief inline implementations for the VoxelsOnCartesianGrid class 

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author PARAPET project

  $Date$
  $Revision$

*/
START_NAMESPACE_TOMO


template<class elemT>
CartesianCoordinate3D<float> 
VoxelsOnCartesianGrid<elemT>::get_voxel_size() const
{
  return CartesianCoordinate3D<float>(get_grid_spacing());
}




END_NAMESPACE_TOMO
