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

START_NAMESPACE_TOMO

// TODO all these are terribly wasteful with memory allocations
// main reason: we cannot have segments with viewgrams of different sizes (et al)

void zoom_segment (PETSegmentByView& segment, 
                   const float zoom, const float x_offset_in_mm,  const float y_offset_in_mm, 
                   const int size, const float azimuthal_angle_sampling)
{

  // First check if there is anything to do at all, 
  // if not, return original segment.

    if (size == segment.get_num_bins() &&
        zoom == 1.0 && x_offset_in_mm == 0.0 && y_offset_in_mm == 0.0) 
        return;

    // KT 17/01/2000 use local copy of scan_info instead of segment.scan_info
    PETScanInfo scan_info = segment.scan_info;
    scan_info.set_num_bins(size);
    scan_info.set_bin_size(segment.scan_info.get_bin_size() / zoom);
    
    const int minsize = -size/2;
    const int maxsize = minsize+size-1;
 
    // TODO replace by get_empty_segment or so
    PETSegmentByView out_segment(Tensor3D<float>(segment.get_min_view(), segment.get_max_view(),
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
                      size, azimuthal_angle_sampling);
        out_segment.set_viewgram(viewgram);
    }

    segment = out_segment;
}

// This function is a copy of the previous lines, except for the 
// segment declaration line.
void 
zoom_segment (PETSegmentBySinogram& segment, 
	      const float zoom, const float x_offset_in_mm, const float y_offset_in_mm, 
	      const int size, const float azimuthal_angle_sampling)
{

    if (size == segment.get_num_bins() &&
        zoom == 1.0 && x_offset_in_mm == 0.0 && y_offset_in_mm == 0.0) 
        return;

    // const float azimuthal_angle_sampling = _PI / segment.get_num_views();
    
    // KT 17/01/2000 use local copy of scan_info instead of segment.scan_info
    PETScanInfo scan_info = segment.scan_info;
    scan_info.set_num_bins(size);
    scan_info.set_bin_size(segment.scan_info.get_bin_size() / zoom);

    const int minsize = -size/2;
    const int maxsize = minsize+size-1;
    
    // TODO replace by get_empty...
    PETSegmentBySinogram out_segment(Tensor3D<float>(segment.get_min_ring(), segment.get_max_ring(), 
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
                      size, azimuthal_angle_sampling);
        out_segment.set_viewgram(viewgram);
    }

    segment = out_segment;
}


// KT 17/01/2000 const for Xoffp
void
zoom_viewgram (PETViewgram& in_view, 
	       const float zoom, const float x_offset_in_mm, const float y_offset_in_mm, 
	       const int size, const float azimuthal_angle_sampling)
{   
    if (size == in_view.get_num_bins() &&
        zoom == 1.0 && x_offset_in_mm == 0.0 && y_offset_in_mm == 0.0) 
        return;
    
    // KT 17/01/2000 use local copy of scan_info
    PETScanInfo scan_info = in_view.scan_info;
    scan_info.set_num_bins(size);
    scan_info.set_bin_size(in_view.scan_info.get_bin_size() / zoom);

    // TODO replace by get_empty...
    PETViewgram out_view(Tensor2D<float> (in_view.get_min_ring(), in_view.get_max_ring(), 
                                          -(size-1)/2,-(size-1)/2+size-1),
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

void
zoom_image(PETImageOfVolume &image,
           const float zoom,
	   const float x_offset_in_mm, const float y_offset_in_mm, 
	   const int new_size )                          
{
    if(zoom==1 && x_offset_in_mm==0 && y_offset_in_mm==0 && new_size== image.get_x_size()) 
        return;
   
    Point3D voxel_size(image.get_voxel_size().x/zoom,
                       image.get_voxel_size().y/zoom,
                       image.get_voxel_size().z);
    // KT 17/01/2000 update origin as well
    Point3D origin(image.get_origin().x + x_offset_in_mm,
                   image.get_origin().y + y_offset_in_mm,
		   image.get_origin().z);

    // TODO use get_empty...
    PETImageOfVolume new_image(Tensor3D<float>(image.get_min_z(), image.get_max_z(),
                                               -new_size/2, -new_size/2+new_size-1,
                                               -new_size/2, -new_size/2+new_size-1),
                               origin,
                               voxel_size);

    PETPlane new_image2D=new_image.get_plane(new_image.get_min_z());
    for (int plane = image.get_min_z(); plane <= image.get_max_z(); plane++)
    {
      // KT 17/01/2000 reordered args
      zoom_image(new_image2D, image.get_plane(plane));
	new_image.set_plane(new_image2D,plane);
    }
    image=new_image;
    
    // KT 17/01/2000 introduced assert and removed explicit set_voxel_size
    // TODO cannot use == with Point3D yet
    // assert(image.get_voxel_size() == voxel_size);
    assert(image.get_voxel_size().x == voxel_size.x);
    assert(image.get_voxel_size().y == voxel_size.y);
    assert(image.get_voxel_size().z == voxel_size.z);
}


// KT 17/01/2000 new
// KT 16/02/2000 use CartesianCoordinate  
void
zoom_image(PETImageOfVolume &image,
	   const CartesianCoordinate3D<float>& zooms,
	   const CartesianCoordinate3D<float>& offsets_in_mm,
	   const CartesianCoordinate3D<int>& new_sizes)
{

  Point3D voxel_size(image.get_voxel_size().x/zooms.x(),
		     image.get_voxel_size().y/zooms.y(),
		     image.get_voxel_size().z/zooms.z());
  
  Point3D origin(image.get_origin().x + offsets_in_mm.x(),
		 image.get_origin().y + offsets_in_mm.y(),
		 image.get_origin().z + offsets_in_mm.z());
  
  PETImageOfVolume 
    new_image(Tensor3D<float>(-new_sizes.z()/2, -new_sizes.z()/2+new_sizes.z()-1,
                              -new_sizes.y()/2, -new_sizes.y()/2+new_sizes.y()-1,
                              -new_sizes.x()/2, -new_sizes.x()/2+new_sizes.x()-1),
              origin,
              voxel_size);
  zoom_image(new_image, image);
  
  image = new_image;
}

// KT 17/01/2000 new
void 
zoom_image(PETImageOfVolume &image_out, const PETImageOfVolume &image_in)
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
    image_in.get_voxel_size().x / image_out.get_voxel_size().x;
  const float zoom_y = 
    image_in.get_voxel_size().y / image_out.get_voxel_size().y;
  const float zoom_z = 
    image_in.get_voxel_size().z / image_out.get_voxel_size().z;
  const float x_offset = 
    ((image_out.get_origin().x - image_in.get_origin().x) 
     / image_in.get_voxel_size().x) + 
    (image_in.get_max_x() + image_in.get_min_x())/2.F -
    (image_out.get_max_x() + image_out.get_min_x())/2.F / zoom_x;
  const float y_offset = 
    ((image_out.get_origin().y - image_in.get_origin().y) 
     / image_in.get_voxel_size().y
    ) + 
    (image_in.get_max_y() + image_in.get_min_y())/2.F -
    (image_out.get_max_y() + image_out.get_min_y())/2.F / zoom_y;
  const float z_offset = 
    ((image_out.get_origin().z - image_in.get_origin().z) 
     / image_in.get_voxel_size().z
    ) + 
    (image_in.get_max_z() + image_in.get_min_z())/2.F -
    (image_out.get_max_z() + image_out.get_min_z())/2.F / zoom_z;

  // TODO extra tests on get_min_x and y and z
  if(zoom_x==1.0F && zoom_y==1.0F && zoom_z==1.0F &&
     x_offset == 0.F && y_offset == 0.F && z_offset == 0.F &&
     image_in.get_x_size() == image_out.get_x_size() &&
     image_in.get_y_size() == image_out.get_y_size() &&
     image_in.get_z_size() == image_out.get_z_size()
    )
  {
    image_out = image_in;
    return;
  }

  // TODO creating a lot of new images here...
  Tensor3D<float> temp(image_in.get_min_z(), image_in.get_max_z(),
                       image_in.get_min_y(), image_in.get_max_y(),
                       image_out.get_min_x(), image_out.get_max_x());

  for (int z=image_in.get_min_z(); z<=image_in.get_max_z(); z++)
    for (int y=image_in.get_min_y(); y<=image_in.get_max_y(); y++)
      overlap_interpolate(temp[z][y], image_in[z][y], zoom_x, x_offset);

  Tensor3D<float> temp2(image_in.get_min_z(), image_in.get_max_z(),
                       image_out.get_min_y(), image_out.get_max_y(),
                       image_out.get_min_x(), image_out.get_max_x());

  // KT 04/02/2000 corrected to use y zooming
  for (int z=image_in.get_min_z(); z<=image_in.get_max_z(); z++)
    overlap_interpolate(temp2[z], temp[z], zoom_y, y_offset);

  overlap_interpolate(image_out, temp2, zoom_z, z_offset);

}


void
zoom_image(PETPlane &image2D_out, const PETPlane &image2D_in)
{
  /*
    see above for how to find zoom and offsets
  */

  const float zoom_x = 
    image2D_in.get_voxel_size().x / image2D_out.get_voxel_size().x;
  const float zoom_y = 
    image2D_in.get_voxel_size().y / image2D_out.get_voxel_size().y;
  const float x_offset = 
    ((image2D_out.get_origin().x - image2D_in.get_origin().x) 
     / image2D_in.get_voxel_size().x
    ) + 
    (image2D_in.get_max_x() + image2D_in.get_min_x())/2.F -
    (image2D_out.get_max_x() + image2D_out.get_min_x())/2.F / zoom_x;
  const float y_offset = 
    ((image2D_out.get_origin().y - image2D_in.get_origin().y) 
     / image2D_in.get_voxel_size().y
    ) + 
    (image2D_in.get_max_y() + image2D_in.get_min_y())/2.F -
    (image2D_out.get_max_y() + image2D_out.get_min_y())/2.F / zoom_y;
  

  // TODO extra tests on get_min_x and y
  if(zoom_x==1.0F && zoom_y==1.0F &&
     x_offset == 0.F && y_offset == 0.F &&
     image2D_in.get_x_size() == image2D_out.get_x_size() &&
     image2D_in.get_y_size() == image2D_out.get_y_size()
    )
  {
    image2D_out = image2D_in;
    return;
  }

  Tensor2D<float> temp(image2D_in.get_min_y(), image2D_in.get_max_y(),
                       image2D_out.get_min_x(), image2D_out.get_max_x());

  for (int y=image2D_in.get_min_y(); y<=image2D_in.get_max_y(); y++)
    overlap_interpolate(temp[y], image2D_in[y], zoom_x, x_offset);

  overlap_interpolate(image2D_out, temp, zoom_y, y_offset);   
}

END_NAMESPACE_TOMO
