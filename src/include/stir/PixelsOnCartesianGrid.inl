//
// $Id$
//
/*!
  \file 
  \ingroup buildblock 
  \brief inline implementations for the PixelsOnCartesianGrid class 

  \author Sanida Mustafovic 
  \author Kris Thielemans (with help from Alexey Zverovich)
  \author PARAPET project

  $Date$

  $Revision$

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
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
    { return get_length()==0 ? 0 : get_min_index(); }

template<class elemT>
int
PixelsOnCartesianGrid<elemT>::get_min_x() const
    { return get_length()==0 ? 0 : (*this)[get_min_y()].get_min_index(); }


template<class elemT>
int 
PixelsOnCartesianGrid<elemT>::get_x_size() const
{ return  get_length()==0 ? 0 : (*this)[get_min_y()].get_length(); }

template<class elemT>
int
PixelsOnCartesianGrid<elemT>:: get_y_size() const
{ return get_length()==0 ? 0 : get_length();}

template<class elemT>
int
PixelsOnCartesianGrid<elemT>::get_max_x() const
{ return get_length()==0 ? 0 : (*this)[get_min_y()].get_max_index();}

template<class elemT>
int
PixelsOnCartesianGrid<elemT>:: get_max_y() const
{ return get_length()==0 ? 0 : get_max_index();}

template<class elemT>
CartesianCoordinate2D<float> 
PixelsOnCartesianGrid<elemT>::get_pixel_size() const
{
  return CartesianCoordinate2D<float>(get_grid_spacing());
}

template<class elemT>
void 
PixelsOnCartesianGrid<elemT>::set_pixel_size(const BasicCoordinate<2,float>& s) const
{
  set_grid_spacing(c);
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
  return new PixelsOnCartesianGrid(get_index_range(),
		                   get_origin(), 
		                   get_grid_spacing());
}


template<class elemT>
DiscretisedDensity<2, elemT>* 
PixelsOnCartesianGrid<elemT>::clone() const
{
  return new PixelsOnCartesianGrid(*this);
}
END_NAMESPACE_STIR
