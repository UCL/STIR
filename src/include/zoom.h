//
// $Id$: $Date$
//
#ifndef __ZOOM_H__
#define  __ZOOM_H__

/*!
  \file 
 
  \brief This file declares various zooming functions.

  \author Kris Thielemans
  \author Claire Labbe
  \author PARAPET project

  \date    $Date$

  \version $Revision$

  These functions can be used for zooming of projection data or image data.
  Zooming requires interpolation. Currently, this is done using 
  overlap_interpolate.
  
  The first set of functions allows zooming and translation in transaxial
  planes only. This parameters are the same for projection data or image data.
  That is, zoomed projection data are (approximately) the forward projections
  of zoomed images with the same parameters. These functions have the following
  parameters:
  
  \param  zoom  scales the projection bins (zoom larger than 1 means more detail, so smaller pixels)
  \param  x_offset_in_mm  x-coordinate of new origin (in mm)
  \param  y_offset_in_mm  y-coordinate of new origin (in mm)
  \param  size  the number of 'bins' in the new projection line (or equivalent for images)
  \param azimuthal_angle_sampling (for projection data only) is used to convert 
         \c view_num to angles (in radians)
		 
  In the case that the offsets are 0, the following holds:
  <UL>
  <LI> If \c size is equal to \c zoom*old_size, the same amount of data is represented.
  <LI> If it is less, data are truncated.
  <LI> If it is larger, the outer ends are filled with 0.
  </UL>
  
  The second class of functions is for images only. It allows three-dimensional
  zooming and translation. Parameters are derived either from
  CartesianCoordinate3D objects, or from the information in the \a in and \a out
  images.

  \warning Because overlap_interpolate is used, the zooming is 'count-preserving',
  i.e. when the output range is large enough, the in.sum() == out.sum().

  \warning Origins are taken relative to the centre of the coordinate range:<BR>
    x_in_mm = (x_index - x_ctr_index) * voxel_size.x() + origin.x() <BR>
    where x_ctr_index = (x_max_index + x_min_index)/2
*/

#ifdef SINO
#include "sinodata.h"
#endif
#include "VoxelsOnCartesianGrid.h" 
#include "PixelsOnCartesianGrid.h" 

START_NAMESPACE_TOMO
#ifdef SINO
/*!
  \brief zoom \c segment, replacing the first argument with the new data

 \see zoom.h for parameter info
*/
void 
zoom_segment (SegmentByView& segment, 
              const float zoom, const float x_offset_in_mm, const float y_offset_in_mm, 
              const int size, const float azimuthal_angle_sampling);

/*!
  \brief zoom \c segment, replacing the first argument with the new data

 \see zoom.h for parameter info
*/
void 
zoom_segment (SegmentBySinogram& segment, 
              const float zoom, const float x_offset_in_mm, const float y_offset_in_mm, 
              const int size, const float azimuthal_angle_sampling);

/*!
  \brief zoom \c viewgram, replacing the first argument with the new data

 \see zoom.h for parameter info
*/
void
zoom_viewgram (Viewgram& viewgram, 
               const float zoom, const float x_offset_in_mm, const float y_offset_in_mm, 
               const int size, const float azimuthal_angle_sampling);

/*!
  \brief zoom \c image, replacing the first argument with the new data

 \see zoom.h for parameter info
*/
#endif

void
zoom_image(VoxelsOnCartesianGrid<float> &image,
	   const float zoom,
	   const float x_offset_in_mm, const float y_offset_in_mm, 
	   const int new_size);

/*! 
  \brief 
  zoom \c image, replacing the first argument with the new data. 
  Full 3D shifts and zooms. 

  \see zoom.h for parameter info.
*/
void
zoom_image(VoxelsOnCartesianGrid<float> &image,
	   const CartesianCoordinate3D<float>& zooms,
	   const CartesianCoordinate3D<float>& offsets_in_mm,
	   const CartesianCoordinate3D<int>& new_sizes);

/*!
 \brief zoom \c image_in according to dimensions, origin and voxel_size of \c image_out.

  \see zoom.h for parameter info
*/
void 
zoom_image(VoxelsOnCartesianGrid<float> &image_out, 
	   const VoxelsOnCartesianGrid<float> &image_in);

/*! 
\brief 
zoom \c image2D_in according to dimensions, origin and pixel_size of 
\c image2D_out.

  \see zoom.h for parameter info
*/
void
zoom_image(PixelsOnCartesianGrid<float> &image2D_out, 
           const PixelsOnCartesianGrid<float> &image2D_in);

END_NAMESPACE_TOMO

#endif

