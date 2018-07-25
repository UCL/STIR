//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
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
#ifndef __stir_ZOOM_H__
#define  __stir_ZOOM_H__

/*!
  \file 
 
  \brief This file declares various zooming functions.
  \ingroup buildblock
  \author Kris Thielemans
  \author Claire Labbe
  \author PARAPET project



  These functions can be used for zooming of projection data or image data.
  Zooming requires interpolation. Currently, this is done using 
  stir::overlap_interpolate.
  
  The first set of functions allows zooming and translation in transaxial
  planes only. These parameters are the same for projection data or image data.
  That is, zoomed projection data are (approximately) the forward projections
  of zoomed images with the same parameters. These functions have the following
  parameters:
  
  \param  zoom  scales the projection bins (zoom larger than 1 means more detail, so smaller pixels)
  \param  x_offset_in_mm  x-coordinate of new origin (in mm)
  \param  y_offset_in_mm  y-coordinate of new origin (in mm)

  \par Projection data

  This allows 2-dimensional
  zooming and translation on arccorrected data, here translation means a translation in
  'image' space which gives a \c sin shift of origin in the \c s - coordinate in
  projection space. 
  The new range of \c s coordinates is given by
  \param  min_tang_pos_num [for projection data only] the minimum tangential position number in the new 
             projection line
  \param  max_tang_pos_num [for projection data only] the maximum tangential position number in the new 
             projection line
  Note that the (projection of the) centre of the scanner axis is supposed to be 
  at \a tang_pos_num = 0.

  \par images

  It allows three-dimensional
  zooming and translation. Parameters are derived either from
  stir::CartesianCoordinate3D objects, or from the information in the \a in and \a out
  images.
	 
  In the case that the offsets are 0, the following holds:
  <UL>
  <LI> If \c size is equal to \c zoom*old_size, the same amount of data is represented.
  <LI> If it is less, data are truncated.
  <LI> If it is larger, the outer ends are filled with 0.
  </UL>
  
  
  \warning Because overlap_interpolate is used, the zooming is 'count-preserving',
  i.e. when the output range is large enough, the in.sum() == out.sum().

  \warning In STIR 1.x, origins were taken relative to the centre of the coordinate range:
\code 
    x_in_mm = (x_index - x_ctr_index) * voxel_size.x() + origin.x() 
\endcode
    where 
\code 
   x_ctr_index = (x_max_index + x_min_index)/2
\endcode
  This is no longer true. Instead we use 
  DiscretisedDensity\<3,float\>::get_physical_coordinates_for_indices.
*/

#include "stir/common.h"
#include "stir/DiscretisedDensity.h"
#include "stir/PostFiltering.h"
#include "stir/DataProcessor.h"
#include "stir/SeparableGaussianImageFilter.h"
#include "stir/ZoomOptions.h"

START_NAMESPACE_STIR

template <typename elemT> class Viewgram;
template <typename elemT> class RelatedViewgrams;
template <typename elemT> class VoxelsOnCartesianGrid;
template <typename elemT> class PixelsOnCartesianGrid;
template <typename elemT> class CartesianCoordinate3D;
template <int num_dimensions, typename elemT> class BasicCoordinate;
/*!
 \ingroup buildblock
  \name Functions for interpolating data to new pixel/bin sizes
*/
//@{

/*!
  \brief zoom a RelatedViewgrams object, replacing it with the new data
   \see zoom.h for parameter info
*/
void
zoom_viewgrams (RelatedViewgrams<float>& viewgrams, 
	       const float zoom, 
	       const int min_tang_pos_num, const int max_tang_pos_num,
	       const float x_offset_in_mm = 0, const float y_offset_in_mm = 0);

/*!
  \brief zoom \a viewgram, replacing it with the new data
  \see zoom.h for parameter info
*/
void
zoom_viewgram (Viewgram<float>& viewgram, 
	       const float zoom, 
	       const int min_tang_pos_num, const int max_tang_pos_num,
	       const float x_offset_in_mm = 0, const float y_offset_in_mm = 0);
/*!
  \brief zoom \a in_viewgram, replacing \a out_viewgram with the new data
  
  This version of zoom_viewgram gets the info on the new sizes, sampling etc. 
  from \a out_viewgram.
 \see zoom.h for parameter info
*/
void
zoom_viewgram (Viewgram<float>& out_viewgram, 
	       const Viewgram<float>& in_viewgram, 
	       const float x_offset_in_mm = 0, const float y_offset_in_mm = 0);



/*!
  \brief zoom \a image, returning the new image
  \see zoom.h for parameter info

  \see zoom_image_in_place
*/
VoxelsOnCartesianGrid<float>
zoom_image(const VoxelsOnCartesianGrid<float> &image,
	   const CartesianCoordinate3D<float>& zooms,
	   const CartesianCoordinate3D<float>& offsets_in_mm,
	   const BasicCoordinate<3,int>& new_sizes);

/*! 
  \brief 
  zoom \a image, replacing the first argument with the new data. 
  Full 3D shifts and zooms. 
  \see zoom.h for parameter info.

  Zooming is done such that the physical coordinates of a point
  (as returned by 
  DiscretisedDensity\<3,float\>::get_physical_coordinates_for_indices)
  remain the same.

  The index range of the new image is according to the standard STIR
  conventions (z starts from 0, but x and y from -(new_size/2)).
  The origin is then chosen such that the geometric centres of the
  images satisfy
  \verbatim
  offsets_in_mm == new_middle - old_middle
  \endverbatim
  The geometric centre is determined by the average of the physical
  coordinates of the min and max indices.

  \warning: For even-sized images, this convention can lead to somewhat
  non-intuitive results (half-pixel shifts etc).
*/
void
zoom_image_in_place(VoxelsOnCartesianGrid<float> &image,
		    const CartesianCoordinate3D<float>& zooms,
		    const CartesianCoordinate3D<float>& offsets_in_mm,
		    const BasicCoordinate<3,int>& new_sizes);

/*!
 \brief zoom \a image_in according to dimensions, origin and voxel_size of \a image_out.
  \see zoom.h for parameter info

  Zooming is done such that the physical coordinates of a point
  (as returned by 
  DiscretisedDensity\<3,float\>::get_physical_coordinates_for_indices)
  remain the same.
*/


void
zoom_image(VoxelsOnCartesianGrid<float> &image_out,
       const VoxelsOnCartesianGrid<float> &image_in, const std::string& postfilter_parameter_filename, const ZoomOptions::ZO &zo = ZoomOptions::preserve_sum);


void
zoom_image(VoxelsOnCartesianGrid<float> &image_out,
       const VoxelsOnCartesianGrid<float> &image_in, const ZoomOptions::ZO &zo = ZoomOptions::preserve_sum);



void
zoom_image_matlab(VoxelsOnCartesianGrid<float> &image_out,
       const VoxelsOnCartesianGrid<float> &image_in, const std::string& postfilter_parameter_filename, const int &zoom_option = 0);


void
zoom_image_matlab(VoxelsOnCartesianGrid<float> &image_out,
       const VoxelsOnCartesianGrid<float> &image_in, const int &zoom_option = 0);
//------------------ 2D zooms---------------------

/*! 
\brief 
zoom \a image2D_in according to dimensions, origin and pixel_size of 
\a image2D_out.
   \see zoom.h for parameter info
*/
void
zoom_image(PixelsOnCartesianGrid<float> &image2D_out, 
           const PixelsOnCartesianGrid<float> &image2D_in);

/*!
  \brief zoom \a image, replacing the first argument with the new data
  \see zoom.h for parameter info
*/
VoxelsOnCartesianGrid<float>
zoom_image(const VoxelsOnCartesianGrid<float> &image,
	   const float zoom,
	   const float x_offset_in_mm, const float y_offset_in_mm, 
	   const int new_size);

/*!
  \brief zoom \a image, replacing the first argument with the new data
  \see zoom.h for parameter info
*/
void
zoom_image_in_place(VoxelsOnCartesianGrid<float> &image,
		    const float zoom,
		    const float x_offset_in_mm, const float y_offset_in_mm, 
		    const int new_size);

//@}

END_NAMESPACE_STIR

#endif

