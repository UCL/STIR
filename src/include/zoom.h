//
// $Id$: $Date$
//

#ifndef __ZOOM_H__
#define  __ZOOM_H__

#include "sinodata.h"
#include "imagedata.h" 
// TODO remove next when used in images
#include "CartesianCoordinate3D.h"

/* 
   zooming of projection data or image data
   parameters :
     zoom : scales the projection bins (zoom > 1 means more detail)
     Xoffp, Yoffp: are new x,y offsets (in mm)
     size : the desired length of the new projection line 

   if size == zoom*get_num_bins(), the same amount of data is represented
           <                     , data are truncated
           >                     , the line is filled with 0 at the outer ends

*/

void 
zoom_segment (PETSegmentByView& segment, 
              const float zoom, const float Xoffp, const float Yoffp, 
              const int size, const float itophi);


void 
zoom_segment (PETSegmentBySinogram& segment, 
              const float zoom, const float Xoffp, const float Yoffp, 
              const int size, const float itophi);

// KT 17/01/2000 const for Xoffp
void
zoom_viewgram (PETViewgram& in_view, 
               const float zoom, const float Xoffp, const float Yoffp, 
               const int size, const float itophi);


void
zoom_image(PETImageOfVolume &image,
                       const float zoom,
                       const float Xoffp, const float Yoffp, 
                       const int new_size);

// KT 17/01/2000 new
// KT 16/02/2000 renamed to CartesianCoordinate
void
zoom_image(PETImageOfVolume &image,
	   const CartesianCoordinate3D<float>& zooms,
	   const CartesianCoordinate3D<float>& offsets,
	   const CartesianCoordinate3D<int>& new_sizes);

// KT 17/01/2000 new
void 
zoom_image(PETImageOfVolume &image_out, const PETImageOfVolume &image_in);

// KT 17/01/2000 reordered args
void
zoom_image(PETPlane &image2D_out, const PETPlane &image2D_in);

#endif

