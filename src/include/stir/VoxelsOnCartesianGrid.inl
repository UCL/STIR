//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup densitydata
  \brief inline implementations for the stir::VoxelsOnCartesianGrid class

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project


*/
START_NAMESPACE_STIR

template <class elemT>
CartesianCoordinate3D<float>
VoxelsOnCartesianGrid<elemT>::get_voxel_size() const
{
  return CartesianCoordinate3D<float>(this->get_grid_spacing());
}

template <typename elemT>
int
VoxelsOnCartesianGrid<elemT>::get_min_z() const
{
  return this->get_min_index(0);
}

template <typename elemT>
int
VoxelsOnCartesianGrid<elemT>::get_min_y() const
{
  #ifdef STIR_WITH_TORCH
  return this->get_min_index(1);
#else
  return this->get_length() == 0 ? 0 : (*this)[get_min_z()].get_min_index();
#endif
}

template <typename elemT>
int
VoxelsOnCartesianGrid<elemT>::get_min_x() const
{
#ifdef STIR_WITH_TORCH
  return this->get_min_index(2);
#else
  return this->get_length() == 0 ? 0 : (*this)[get_min_z()][get_min_y()].get_min_index();
#endif

}

template <typename elemT>
int
VoxelsOnCartesianGrid<elemT>::get_x_size() const
{
#ifdef STIR_WITH_TORCH
  return this->get_length(2);
#else
   return this->get_length() == 0 ? 0 : (*this)[get_min_z()][get_min_y()].get_length();
#endif
}

template <typename elemT>
int
VoxelsOnCartesianGrid<elemT>::get_y_size() const
{
#ifdef STIR_WITH_TORCH
  return this->get_length(1);
#else
  return this->get_length() == 0 ? 0 : (*this)[get_min_z()].get_length();
#endif
}

template <typename elemT>
int
VoxelsOnCartesianGrid<elemT>::get_z_size() const
{
  return this->get_length();
}

template <typename elemT>
int
VoxelsOnCartesianGrid<elemT>::get_max_x() const
{
#ifdef STIR_WITH_TORCH
  return this->get_max_index(1);
#else
    return this->get_length() == 0 ? 0 : (*this)[get_min_z()][get_min_y()].get_max_index();
#endif
}

template <typename elemT>
int
VoxelsOnCartesianGrid<elemT>::get_max_y() const
{
#ifdef STIR_WITH_TORCH
  return this->get_max_index(2);
#else
  return this->get_length() == 0 ? 0 : (*this)[get_min_z()].get_max_index();
#endif

}

template <typename elemT>
int
VoxelsOnCartesianGrid<elemT>::get_max_z() const
{
  return this->get_max_index();
}

END_NAMESPACE_STIR
