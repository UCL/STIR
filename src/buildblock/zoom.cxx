//
// $Id$: $Date$
//
/*!
  \file 
 
  \brief Implementations of the zoom functions

  \author Kris Thielemans
  \author Claire Labbe
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/
/* Modification history:
   - First versions by CL and KT (sinogram version based on C code by Matthias Egger (using linear interpolation).
   - CL introduced overlap interpolation.
   - KT moved interpolation to separate function overlap_interpolate, removing bugs.
   - KT introduced 3D zooming for images.
 */
   
#include <cmath>
#include "interpolate.h"
#include "zoom.h"
#include "IndexRange3D.h"
#include "IndexRange2D.h"

START_NAMESPACE_TOMO

// TODO all these are terribly wasteful with memory allocations
// main reason: we cannot have segments with viewgrams of different sizes (et al)
#ifdef SINO
void zoom_segment (PETSegmentByView& segment, 
                   const float zoom, const float azimuthal_angle_sampling, const float y_offset_in_mm, 
                   const int new_size, const float azimuthal_angle_sampling)
{

  // First check if there is anything to do at all, 
  // if not, return original segment.

  if (new_size == segment.get_num_bins() &&
      zoom == 1.0 && x_offset_in_mm == 0.0 && y_offset_in_mm == 0.0) 
    return;

  // KT 17/01/2000 use local copy of scan_info instead of segment.scan_info
  PETScanInfo scan_info = segment.scan_info;
  scan_info.set_num_bins(new_size);
  scan_info.set_bin_size(segment.scan_info.get_bin_size() / zoom);
    
  const int minsize = -new_size/2;
  const int maxsize = minsize+new_size-1;
 
  // TODO replace by get_empty_segment or so
  PETSegmentByView 
    out_segment(Tensor3D<float>(segment.get_min_view(), segment.get_max_view(),
				segment.get_min_ring(), segment.get_max_ring(), 
				minsize, maxsize),
		scan_info,
		segment.get_segment_num(),
		segment.get_min_ring_difference(),
		segment.get_max_ring_difference());

  for (int view = segment.get_min_view(); view <= segment.get_max_view(); view++) 
    {
      PETViewgram viewgram = segment.get_viewgram(view);
      zoom_viewgram(viewgram,
		    zoom, x_offset_in_mm, y_offset_in_mm,
		    new_size, azimuthal_angle_sampling);
      out_segment.set_viewgram(viewgram);
    }

  segment = out_segment;
}

// This function is a copy of the previous lines, except for the 
// segment declaration line.
void 
zoom_segment (PETSegmentBySinogram& segment, 
	      const float zoom, const float x_offset_in_mm, const float y_offset_in_mm, 
	      const int new_size, const float azimuthal_angle_sampling)
{

  if (new_size == segment.get_num_bins() &&
      zoom == 1.0 && x_offset_in_mm == 0.0 && y_offset_in_mm == 0.0) 
    return;

  // const float azimuthal_angle_sampling = _PI / segment.get_num_views();
    
  // KT 17/01/2000 use local copy of scan_info instead of segment.scan_info
  PETScanInfo scan_info = segment.scan_info;
  scan_info.set_num_bins(new_size);
  scan_info.set_bin_size(segment.scan_info.get_bin_size() / zoom);

  const int minsize = -new_size/2;
  const int maxsize = minsize+new_size-1;
    
  // TODO replace by get_empty...
  PETSegmentBySinogram 
    out_segment(Tensor3D<float>(segment.get_min_ring(), segment.get_max_ring(), 
				segment.get_min_view(), segment.get_max_view(),
				minsize, maxsize),
		scan_info,
		segment.get_segment_num(),
		segment.get_min_ring_difference(),
		segment.get_max_ring_difference());

  for (int view = segment.get_min_view(); view <= segment.get_max_view(); view++) 
    {
      PETViewgram viewgram = segment.get_viewgram(view);
      zoom_viewgram(viewgram,
		    zoom, x_offset_in_mm, y_offset_in_mm,
		    new_size, azimuthal_angle_sampling);
      out_segment.set_viewgram(viewgram);
    }

  segment = out_segment;
}


// KT 17/01/2000 const for Xoffp
void
zoom_viewgram (PETViewgram& in_view, 
	       const float zoom, const float x_offset_in_mm, const float y_offset_in_mm, 
	       const int new_size, const float azimuthal_angle_sampling)
{   
  if (new_size == in_view.get_num_bins() &&
      zoom == 1.0 && x_offset_in_mm == 0.0 && y_offset_in_mm == 0.0) 
    return;
    
  // KT 17/01/2000 use local copy of scan_info
  PETScanInfo scan_info = in_view.scan_info;
  scan_info.set_num_bins(new_size);
  scan_info.set_bin_size(in_view.scan_info.get_bin_size() / zoom);

  // TODO replace by get_empty...
  PETViewgram 
    out_view(Tensor2D<float> (in_view.get_min_ring(), in_view.get_max_ring(), 
			      -(new_size-1)/2,-(new_size-1)/2+new_size-1),
	     scan_info, 
	     in_view.get_view_num(),
	     in_view.get_segment_num());
    
    
  const double phi = in_view.get_view_num()*azimuthal_angle_sampling;
  // compute offset in bin_size_in units
  const float offset = 
    static_cast<float>(x_offset_in_mm*cos(phi) +y_offset_in_mm*sin(phi)) 
    / in_view.scan_info.get_bin_size();

  for (int ring= out_view.get_min_ring(); ring <= out_view.get_max_ring(); ring++)
    {
      overlap_interpolate(out_view[ring], in_view[ring], zoom, offset, false);
    }
 
  in_view = out_view;
    
}

#endif SINO

void
zoom_image(VoxelsOnCartesianGrid<float> &image,
           const float zoom,
	   const float x_offset_in_mm, const float y_offset_in_mm, 
	   const int new_size )                          
{
  if(zoom==1 && x_offset_in_mm==0 && y_offset_in_mm==0 && new_size== image.get_x_size()) 
    return;
   
  CartesianCoordinate3D<float>
    voxel_size(image.get_voxel_size().z()/zoom,
	       image.get_voxel_size().y()/zoom,
	       image.get_voxel_size().x());
  CartesianCoordinate3D<float>
    origin(image.get_origin().z(),
	   image.get_origin().y() + y_offset_in_mm,
	   image.get_origin().x() + x_offset_in_mm);

  VoxelsOnCartesianGrid<float> 
    new_image(IndexRange3D(image.get_min_z(), image.get_max_z(),
			   -new_size/2, -new_size/2+new_size-1,
			   -new_size/2, -new_size/2+new_size-1),
	      origin,
	      voxel_size);

  PixelsOnCartesianGrid<float> 
    new_image2D = new_image.get_plane(new_image.get_min_z());
  for (int plane = image.get_min_z(); plane <= image.get_max_z(); plane++)
    {
      zoom_image(new_image2D, image.get_plane(plane));
      new_image.set_plane(new_image2D, plane);
    }
  image=new_image;
    
  assert(image.get_voxel_size() == voxel_size);
}


void
zoom_image(VoxelsOnCartesianGrid<float> &image,
	   const CartesianCoordinate3D<float>& zooms,
	   const CartesianCoordinate3D<float>& offsets_in_mm,
	   const CartesianCoordinate3D<int>& new_sizes)
{

  CartesianCoordinate3D<float>
    voxel_size = image.get_grid_spacing() / zooms;
  
  CartesianCoordinate3D<float>
    origin = image.get_origin() / offsets_in_mm;
  
  VoxelsOnCartesianGrid<float> 
    new_image(IndexRange3D(-new_sizes.z()/2, -new_sizes.z()/2+new_sizes.z()-1,
			   -new_sizes.y()/2, -new_sizes.y()/2+new_sizes.y()-1,
			   -new_sizes.x()/2, -new_sizes.x()/2+new_sizes.x()-1),
	      origin,
	      voxel_size);
  zoom_image(new_image, image);

  image = new_image;
}

void 
zoom_image(VoxelsOnCartesianGrid<float> &image_out, 
	   const VoxelsOnCartesianGrid<float> &image_in)
{

/*
     interpolation routine uses the following relation:
         x_in_index = x_out_index/zoom  + offset

     compare to 'physical' coordinates
         x_phys = (x_index - x_ctr_index) * voxel_size.x + origin.x
       
     as x_in_phys == x_out_phys, we find
         (x_in_index - x_in_ctr_index)* voxel_size_in.x + origin_in.x ==
           (x_out_index - x_out_ctr_index)* voxel_size_out.x + origin_out.x
        <=>
         x_in_index = x_in_ctr_index +
                      ((x_out_index- x_out_ctr_index) * voxel_size_out.x 
	                + origin_out.x - origin_in.x) 
		      / voxel_size_in.x
  */
  const float zoom_x = 
    image_in.get_voxel_size().x() / image_out.get_voxel_size().x();
  const float zoom_y = 
    image_in.get_voxel_size().y() / image_out.get_voxel_size().y();
  const float zoom_z = 
    image_in.get_voxel_size().z() / image_out.get_voxel_size().z();
  const float x_offset = 
    ((image_out.get_origin().x() - image_in.get_origin().x()) 
     / image_in.get_voxel_size().x()) + 
    (image_in.get_max_x() + image_in.get_min_x())/2.F -
    (image_out.get_max_x() + image_out.get_min_x())/2.F / zoom_x;
  const float y_offset = 
    ((image_out.get_origin().y() - image_in.get_origin().y()) 
     / image_in.get_voxel_size().y()
    ) + 
    (image_in.get_max_y() + image_in.get_min_y())/2.F -
    (image_out.get_max_y() + image_out.get_min_y())/2.F / zoom_y;
  const float z_offset = 
    ((image_out.get_origin().z() - image_in.get_origin().z()) 
     / image_in.get_voxel_size().z()
    ) + 
    (image_in.get_max_z() + image_in.get_min_z())/2.F -
    (image_out.get_max_z() + image_out.get_min_z())/2.F / zoom_z;

  
  if(zoom_x==1.0F && zoom_y==1.0F && zoom_z==1.0F &&
     x_offset == 0.F && y_offset == 0.F && z_offset == 0.F &&
     image_in.get_index_range() == image_out.get_index_range()
     )
    {
      image_out = image_in;
      return;
    }

  // TODO creating a lot of new images here...
  Array<3,float> 
    temp(IndexRange3D(image_in.get_min_z(), image_in.get_max_z(),
		      image_in.get_min_y(), image_in.get_max_y(),
		      image_out.get_min_x(), image_out.get_max_x()));

  for (int z=image_in.get_min_z(); z<=image_in.get_max_z(); z++)
    for (int y=image_in.get_min_y(); y<=image_in.get_max_y(); y++)
      overlap_interpolate(temp[z][y], image_in[z][y], zoom_x, x_offset);

  Array<3,float> temp2(IndexRange3D(image_in.get_min_z(), image_in.get_max_z(),
				    image_out.get_min_y(), image_out.get_max_y(),
				    image_out.get_min_x(), image_out.get_max_x()));

  for (int z=image_in.get_min_z(); z<=image_in.get_max_z(); z++)
    overlap_interpolate(temp2[z], temp[z], zoom_y, y_offset);

  overlap_interpolate(image_out, temp2, zoom_z, z_offset);

}

void
zoom_image(PixelsOnCartesianGrid<float> &image2D_out, 
           const PixelsOnCartesianGrid<float> &image2D_in)
{
  /*
    see above for how to find zoom and offsets
  */

  const float zoom_x = 
    image2D_in.get_pixel_size().x() / image2D_out.get_pixel_size().x();
  const float zoom_y = 
    image2D_in.get_pixel_size().y() / image2D_out.get_pixel_size().y();
  const float x_offset = 
    ((image2D_out.get_origin().x() - image2D_in.get_origin().x()) 
     / image2D_in.get_pixel_size().x()
    ) + 
    (image2D_in.get_max_x() + image2D_in.get_min_x())/2.F -
    (image2D_out.get_max_x() + image2D_out.get_min_x())/2.F / zoom_x;
  const float y_offset = 
    ((image2D_out.get_origin().y() - image2D_in.get_origin().y()) 
     / image2D_in.get_pixel_size().y()
    ) + 
    (image2D_in.get_max_y() + image2D_in.get_min_y())/2.F -
    (image2D_out.get_max_y() + image2D_out.get_min_y())/2.F / zoom_y;
  
  if(zoom_x==1.0F && zoom_y==1.0F &&
     x_offset == 0.F && y_offset == 0.F &&
     image2D_in.get_index_range() == image2D_out.get_index_range()
     )
  {
    image2D_out = image2D_in;
    return;
  }

  Array<2,float> 
    temp(IndexRange2D(image2D_in.get_min_y(), image2D_in.get_max_y(),
		      image2D_out.get_min_x(), image2D_out.get_max_x()));

  for (int y=image2D_in.get_min_y(); y<=image2D_in.get_max_y(); y++)
    overlap_interpolate(temp[y], image2D_in[y], zoom_x, x_offset);

  overlap_interpolate(image2D_out, temp, zoom_y, y_offset);   
}

END_NAMESPACE_TOMO
