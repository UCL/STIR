//
// $Id$
//
/*!
  \file
  \ingroup Shape

  \brief Non-inline implementations for class Shape3D

  \author Kris Thielemans
  \author Sanida Mustafovic
  $Date$
  $Revision$
*/
#include "local/tomo/Shape/Shape3D.h"
#include "local/tomo/Shape/DiscretisedShape3D.h"
#include "DiscretisedDensity.h"
#include "VoxelsOnCartesianGrid.h"

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


// Check the sampled elements of the voxel

START_NAMESPACE_TOMO
/*
Shape3D* Shape3D::read_from_file(const string& filename)
{
  // at the moment only this one
  return new 
    DiscretisedShape3D(DiscretisedDensity<3,float>::read_from_file(filename));
}
*/

float Shape3D::get_voxel_weight(
   const CartesianCoordinate3D<float>& index,
   const CartesianCoordinate3D<float>& voxel_size,
   const CartesianCoordinate3D<int>& num_samples) const
{ 
  int value=0;
  
  for (float zsmall = -float(num_samples.z()-1)/num_samples.z()/2.F;
  zsmall<=0.5F;
  zsmall+=1./num_samples.z())
  {
    const float zinner=index.z()+zsmall;
    
    for (float ysmall =-float(num_samples.y()-1)/num_samples.y()/2.F;
    ysmall<=0.5F;
    ysmall+=1./num_samples.y())
    {
      const float yinner=index.y()+ysmall;
      
      for(float xsmall=-float(num_samples.x()-1)/num_samples.x()/2.F;
      xsmall<=0.5F;
      xsmall+=1./num_samples.x())
      {
	const float xinner= index.x()+xsmall;
	{
	  CartesianCoordinate3D<float> r(xinner,yinner,zinner);
	  r*=voxel_size;   
	  if(is_inside_shape(r))
	  value += 1;	  	   
	}
      }
    }
    
  }
  return float(value)/(num_samples.z()*num_samples.y()*num_samples.x());
}

/* Construct the volume- use the convexity, e.g
   the inner voxels sampled with num_samples=1, only the outer
   voxels checked with the user defined num_samples

  \bug Objects which are only at the edge of the image can be missed
*/
void 
Shape3D::construct_volume(VoxelsOnCartesianGrid<float> &image, 
                          const CartesianCoordinate3D<int>& num_samples) const
{ 
  // TODO
  //CartesianCoordinate3D<float> voxel_size= image.get_voxel_size();
  
  //Point3D pre_voxel_size= image.get_voxel_size();
  CartesianCoordinate3D<float> pre_voxel_size= image.get_voxel_size();
  CartesianCoordinate3D<float> voxel_size(pre_voxel_size.x(), pre_voxel_size.y(), pre_voxel_size.y());
  
  int min_z = image.get_min_z();
  int min_y = image.get_min_y();
  int min_x = image.get_min_x();
  int max_z = image.get_max_z();
  int max_y = image.get_max_y();
  int max_x = image.get_max_x();

  CartesianCoordinate3D<int> crude_num_samples(1,1,1);

  for(int z = min_z;z<=max_z;z++)
  {
    for(int y =min_y;y<=max_y;y++)
      for(int x=min_x;x<=max_x;x++)
	
      {
   CartesianCoordinate3D<float> current_point(x,y,z);
	
	image[z][y][x] = get_voxel_weight(current_point,voxel_size,crude_num_samples);
      }
  }
      
   if (num_samples.x() == 1 && num_samples.y() == 1 && num_samples.z() == 1)
    return;

  int num_recomputed = 0;
  for(int z =min_z;z<=max_z;z++)  
    for(int y =min_y;y<=max_y;y++)
      for(int x=min_x;x<= max_x;x++)
      {
	const CartesianCoordinate3D<float> current_point(x,y,z);
	const float current_value = image[z][y][x];

	bool all_neighbours_are_equal = true;
	for(int i = z-1;all_neighbours_are_equal && (i<=z+1);i++)
	  for(int j= y-1;all_neighbours_are_equal && (j<=y+1);j++)
	    for(int k=x-1;all_neighbours_are_equal && (k<=x+1);k++)	      
	    {
	      const float value_of_neighbour =
		((i < min_z) || (i> max_z) ||
		 (j < min_y) || (j> max_y) ||
		 (k < min_x) || (k> max_x)
		) ? 0 : image[i][j][k];
	      all_neighbours_are_equal = (value_of_neighbour==current_value);	      
	    }
	if (!all_neighbours_are_equal)
	{
	  num_recomputed++;
	  image[z][y][x] = get_voxel_weight(current_point,voxel_size,num_samples);
	}
      }
  cerr << "Number recomputed : " << num_recomputed << endl;
      
}

#if 0
void Shape3D::construct_slice(PixelsOnCartesianGrid<float> &plane, 
                              const CartesianCoordinate3D<int>& num_samples) const
 {

  // TODO
  // CartesianCoordinate3D<float> voxel_size = plane.get_voxel_size();
  CartesianCoordinate2D<float> pre_voxel_size= plane.get_pixel_size();
  //TODO
  //CartesianCoordinate3D<float> grid_spacing =plane.get_grid_spacing();
  int voxel_size_z=1;// grid_spacing.z();

  CartesianCoordinate3D<float> voxel_size(pre_voxel_size.x(), pre_voxel_size.y(),voxel_size_z);
  CartesianCoordinate3D<float> origin = plane.get_origin();
  int z = origin.z();

  //TODO
   for ( int y = plane.get_min_y(); y<=plane.get_max_y(); y++)
   for ( int x = plane.get_min_x();x<=plane.get_max_x();x++)
   
   { 
    CartesianCoordinate3D<float> current_point(x,y,z);
    plane[y][x]= get_voxel_weight(current_point,voxel_size, num_samples);
   }
 }
 
#endif

void 
Shape3D::
set_defaults()
{
  origin[3] = origin[2] = origin[1] =0;
}

void 
Shape3D::
initialise_keymap()
{
  parser.add_key("origin-z (in mm)", &origin.z());
  parser.add_key("origin-y (in mm)", &origin.y());
  parser.add_key("origin-x (in mm)", &origin.x());
}

END_NAMESPACE_TOMO