//
// $Id$
//
/*!
  \file 
  \ingroup densitydata  
  \brief inline implementations for the VoxelsOnCartesianGrid class 

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author PARAPET project

  $Date$
  $Revision$

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
START_NAMESPACE_STIR


template<class elemT>
CartesianCoordinate3D<float> 
VoxelsOnCartesianGrid<elemT>::get_voxel_size() const
{
  return CartesianCoordinate3D<float>(this->get_grid_spacing());
}




END_NAMESPACE_STIR
