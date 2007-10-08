//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup densitydata  
  \brief inline implementations for the stir::VoxelsOnCartesianGrid class 

  \author Sanida Mustafovic 
  \author Kris Thielemans 
  \author PARAPET project

  $Date$
  $Revision$

*/
START_NAMESPACE_STIR


template<class elemT>
CartesianCoordinate3D<float> 
VoxelsOnCartesianGrid<elemT>::get_voxel_size() const
{
  return CartesianCoordinate3D<float>(this->get_grid_spacing());
}

template<typename elemT>
int
VoxelsOnCartesianGrid<elemT>::
get_min_z() const
{ return this->get_min_index();} 


template<typename elemT>
int
VoxelsOnCartesianGrid<elemT>::
get_min_y() const
{ return this->get_length()==0 ? 0 : (*this)[get_min_z()].get_min_index(); }

template<typename elemT>
int
VoxelsOnCartesianGrid<elemT>::
get_min_x() const
{ return this->get_length()==0 ? 0 : (*this)[get_min_z()][get_min_y()].get_min_index(); }


template<typename elemT>
int 
VoxelsOnCartesianGrid<elemT>::
get_x_size() const
{ return  this->get_length()==0 ? 0 : (*this)[get_min_z()][get_min_y()].get_length(); }

template<typename elemT>
int
VoxelsOnCartesianGrid<elemT>::
get_y_size() const
{ return this->get_length()==0 ? 0 : (*this)[get_min_z()].get_length();}

template<typename elemT>
int
VoxelsOnCartesianGrid<elemT>::
get_z_size() const
{ return this->get_length(); }


template<typename elemT>
int
VoxelsOnCartesianGrid<elemT>::
get_max_x() const
{ return this->get_length()==0 ? 0 : (*this)[get_min_z()][get_min_y()].get_max_index();}

template<typename elemT>
int
VoxelsOnCartesianGrid<elemT>::
get_max_y() const
{ return this->get_length()==0 ? 0 : (*this)[get_min_z()].get_max_index();}

template<typename elemT>
int
VoxelsOnCartesianGrid<elemT>::
get_max_z() const
{ return this->get_max_index(); }



END_NAMESPACE_STIR
