//
//
/*
    Copyright (C) 2000- 2002, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "local/stir/Shape/EllipsoidalCylinder.h"

#include "stir/interfile.h"
#include "stir/utilities.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


USING_NAMESPACE_STIR

int 
main(int argc, char **argv)
{
 

 if (argc!=3)
   {
      cerr <<"Usage: " << argv[0] << "  outputfile_name template_image_filename\n";
      return (EXIT_FAILURE);
   }

  char* file_name = argv[1];
  
#if 1

  shared_ptr<DiscretisedDensity<3,float> > template_density_ptr =
    DiscretisedDensity<3,float>::read_from_file(argv[2]);

  VoxelsOnCartesianGrid<float> * vox_template_image_ptr =
    dynamic_cast<VoxelsOnCartesianGrid<float> *>(template_density_ptr.get());

  VoxelsOnCartesianGrid<float>& image = *vox_template_image_ptr;
  image.fill(0);

#else

  float voxel_z= ask_num("Voxel size in z ",0.,5.,1.6);
  float voxel_y= ask_num("Voxel sixe in y ",0.,5.,0.976562);
  float voxel_x= ask_num("Voxel size in x ",0.,5.,0.976562);


  int min_plane= ask_num("Min axial position ",0,1000,1);
  int max_plane= ask_num("Max axial position ",0,1000,114);
  int min_y_dir= ask_num("Min y dir ",-300,300,-128);
  int max_y_dir= ask_num("Max y dir ",-300,300,128);
  int min_x_dir= ask_num("Min x dir ",-300,300,-128);
  int max_x_dir= ask_num("Max x dir ",-300,300,128);
  /*
  int orig_z= ask_num("orig_z",min_plane,max_plane,(min_plane+max_plane)/2);
  int orig_y= ask_num("orig_y",min_y_dir,max_y_dir,(min_y_dir+max_y_dir)/2);
  int orig_x= ask_num("orig_x",min_x_dir,max_x_dir,(min_x_dir+max_x_dir)/2);
  */
 

  CartesianCoordinate3D<float> origin(0,0,0);
  CartesianCoordinate3D<float> voxel_size(voxel_z,voxel_y,voxel_x);

  VoxelsOnCartesianGrid<float> image(IndexRange3D(min_plane,max_plane,
        min_y_dir,max_y_dir,
        min_x_dir,max_x_dir),
        origin,voxel_size);

#endif

  // cylinder data
  const float radius = ask_num("Radius (in mm)", 0.F, 500.F, 100.F);
  const float length = ask_num("Length (in mm)", 0.F, 1000.F, 1000.F);

  const float velocity_x = ask_num("Velocity of x in z",-1000.F,1000.F,0.F);
  const float velocity_y = ask_num("Velocity of y in z",-1000.F,1000.F,0.F);
  const float velocity_z = 1.F;
  CartesianCoordinate3D<float> dir_z(1.F,velocity_y, velocity_x);
  dir_z /= norm(dir_z);
  CartesianCoordinate3D<float> dir_y(-velocity_y, velocity_z,0.F);
  dir_y /= norm(dir_y);
  CartesianCoordinate3D<float> dir_x(-velocity_x*velocity_z,
				     -velocity_x*velocity_y,
				     square(velocity_y)+square(velocity_z)
				     );
  dir_x /= norm(dir_x);

  //  std::cerr<<  "dirx.diry: " << inner_product(dir_x,dir_y) << "\n";
  assert(fabs(inner_product(dir_x,dir_y))<1.E-4);
  assert(fabs(inner_product(dir_x,dir_z))<1.E-4);
  assert(fabs(inner_product(dir_y,dir_z))<1.E-4);
  assert(fabs(norm(dir_x)-1)<1.E-4);
  assert(fabs(norm(dir_y)-1)<1.E-4);
  assert(fabs(norm(dir_z)-1)<1.E-4);

  const float shift_x = ask_num("Shift of x in z (in mm)",-1000.F,1000.F,0.F);
  const float shift_y = ask_num("Shift of y in z (in mm)",-1000.F,1000.F,0.F);

  const float centre_z = image.get_length()*image.get_voxel_size().z()/2;
    
  const float origin_z= ask_num("z of centre of cylinder (w.r.t.z of centre of image)",
				-centre_z, centre_z, 0.F);

  const CartesianCoordinate3D<float> 
    cyl_origin(origin_z + centre_z,
	       shift_y + origin_z*velocity_y,
	       shift_x + origin_z*velocity_x);


  EllipsoidalCylinder cylinder(length, radius, radius,
			       cyl_origin,
			       dir_x, dir_y, dir_z);

  const int num_samples = ask_num("Number of samples",1,10,5);
  const float value = ask_num("Value for cylinder",0.F,10000000.F,1.F);
  const CartesianCoordinate3D<int> num_samples3D(num_samples,num_samples,num_samples);
  cylinder.construct_volume(image, num_samples3D);
  image *= value;
  write_basic_interfile(file_name, image);

  return EXIT_SUCCESS;


}
