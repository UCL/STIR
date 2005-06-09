//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd

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
  \brief inline implementations for the stir::PixelsOnCartesianGrid class 

  \author Sanida Mustafovic 
  \author Kris Thielemans (with help from Alexey Zverovich)
  \author PARAPET project

  $Date$
  $Revision$
*/
START_NAMESPACE_STIR


template<class elemT>
PixelsOnCartesianGrid<elemT> ::PixelsOnCartesianGrid()
 : DiscretisedDensityOnCartesianGrid<2,elemT>()
{}

template<class elemT>
PixelsOnCartesianGrid<elemT>::PixelsOnCartesianGrid
		      (const Array<2,elemT>& v,
		       const CartesianCoordinate3D<float>& origin,
		       const BasicCoordinate<2,float>& grid_spacing)
		       :DiscretisedDensityOnCartesianGrid<2,elemT>
		       (v.get_index_range(),origin,grid_spacing)
{
  Array<2,elemT>::operator=(v);
}


template<class elemT>
PixelsOnCartesianGrid<elemT>::PixelsOnCartesianGrid
		      (const IndexRange<2>& range, 
		       const CartesianCoordinate3D<float>& origin,
		       const BasicCoordinate<2,float>& grid_spacing)
		       :DiscretisedDensityOnCartesianGrid<2,elemT>
		       (range,origin,grid_spacing)
{}


template<class elemT>
int
PixelsOnCartesianGrid<elemT>:: get_min_y() const
    { return this->get_length()==0 ? 0 : this->get_min_index(); }

template<class elemT>
int
PixelsOnCartesianGrid<elemT>::get_min_x() const
    { return this->get_length()==0 ? 0 : (*this)[get_min_y()].get_min_index(); }


template<class elemT>
int 
PixelsOnCartesianGrid<elemT>::get_x_size() const
{ return  this->get_length()==0 ? 0 : (*this)[get_min_y()].get_length(); }

template<class elemT>
int
PixelsOnCartesianGrid<elemT>:: get_y_size() const
{ return this->get_length()==0 ? 0 : this->get_length();}

template<class elemT>
int
PixelsOnCartesianGrid<elemT>::get_max_x() const
{ return this->get_length()==0 ? 0 : (*this)[get_min_y()].get_max_index();}

template<class elemT>
int
PixelsOnCartesianGrid<elemT>:: get_max_y() const
{ return this->get_length()==0 ? 0 : this->get_max_index();}

template<class elemT>
CartesianCoordinate2D<float> 
PixelsOnCartesianGrid<elemT>::get_pixel_size() const
{
  return CartesianCoordinate2D<float>(this->get_grid_spacing());
}

template<class elemT>
void 
PixelsOnCartesianGrid<elemT>::set_pixel_size(const BasicCoordinate<2,float>& s) const
{
  this->set_grid_spacing(s);
}

template<class elemT>
#ifdef STIR_NO_COVARIANT_RETURN_TYPES
DiscretisedDensity<2,elemT>*
#else
PixelsOnCartesianGrid<elemT>*
#endif
PixelsOnCartesianGrid<elemT>::get_empty_discretised_density() const
{
  return get_empty_pixels_on_cartesian_grid();
}

/*!
  This member function will be unnecessary when all compilers can handle
  'covariant' return types. 
  It is a non-virtual counterpart of get_empty_pixels_on_cartesian_grid.
*/
template<class elemT>
PixelsOnCartesianGrid<elemT>*
PixelsOnCartesianGrid<elemT>::get_empty_pixels_on_cartesian_grid() const

{
  return new PixelsOnCartesianGrid(this->get_index_range(),
		                   this->get_origin(), 
		                   this->get_grid_spacing());
}


template<class elemT>
DiscretisedDensity<2, elemT>* 
PixelsOnCartesianGrid<elemT>::clone() const
{
  return new PixelsOnCartesianGrid(*this);
}
END_NAMESPACE_STIR
