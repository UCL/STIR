//
// $Id$
//

/*!
  \file
  \ingroup utilities

  \brief A utility to generate images consistening of uniform objects added/subtracted together

  \par Example .par file
  \code
  generate_image Parameters :=
  output filename:= somefile
  XY output image size (in pixels):= 13
  Z output image size (in pixels):= 15
  XY voxel size (in mm):= 4
  Z voxel size (in mm) := 5
  ; now starts an enumeration of objects
  ; Shape3D hierarchy for possible shapes and their parameters
  shape type:= ellipsoidal cylinder
     Ellipsoidal Cylinder Parameters:=
     radius-x (in mm):= 1 
     radius-y (in mm):= 2
     length-z (in mm):= 3
     origin-x (in mm):= 0
     origin-y (in mm):= 15
     origin-z (in mm):= 10
     END:=
  value := 10

  ; here comes another shape
  next shape:=
  shape type:= ellipsoid
  ; etc

  ; as many shapes as you want
  END:=
  \endcode

  \warning If the shape is smaller than the voxel-size, or the shape
  is at the edge of the image, the current
  mechanism of generating the image might miss the shape entirely.

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "local/stir/Shape/Shape3D.h"
#include "stir/KeyParser.h"
#include "stir/is_null_ptr.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/IndexRange3D.h"
#include "stir/Succeeded.h"
#include "stir/IO/DefaultOutputFileFormat.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include <iostream>

USING_NAMESPACE_STIR


class GenerateImage : public KeyParser
{
public:

  GenerateImage(const char * const par_filename);


  Succeeded compute();
private:

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  vector<shared_ptr<Shape3D> > shape_ptrs;
  shared_ptr<Shape3D> current_shape_ptr;
  vector<float> values;
  float current_value;
  string output_filename;
  void increment_current_shape_num();

  int output_image_size_xy;
  int output_image_size_z;
  float output_voxel_size_xy;
  float output_voxel_size_z;
};


void GenerateImage::
increment_current_shape_num()
{
  if (!is_null_ptr( current_shape_ptr))
    {
      shape_ptrs.push_back(current_shape_ptr);
      values.push_back(current_value);
    }
}

void 
GenerateImage::
set_defaults()
{
  output_image_size_xy=-1;
  output_image_size_z=1;
  output_image_size_xy=1;
  output_image_size_z=1;
  shape_ptrs.resize(0);
  values.resize(0);
  output_filename.resize(0);
}

void 
GenerateImage::
initialise_keymap()
{
  add_start_key("generate_image Parameters");
  add_key("output filename",&output_filename);
  add_key("XY output image size (in pixels)",&output_image_size_xy);
  add_key("Z output image size (in pixels)",&output_image_size_z);
  add_key("XY voxel size (in mm)",&output_voxel_size_xy);
  add_key("Z voxel size (in mm)",&output_voxel_size_z);
  add_parsing_key("shape type", &current_shape_ptr);
  add_key("value", &current_value);
  add_key("next shape", KeyArgument::NONE,
	  (KeywordProcessor)&GenerateImage::increment_current_shape_num);
  add_stop_key("END");

}


bool
GenerateImage::
post_processing()
{
  assert(values.size() == shape_ptrs.size());

  if (!is_null_ptr( current_shape_ptr))
    {
      shape_ptrs.push_back(current_shape_ptr);
      values.push_back(current_value);
    }
  if (output_filename.size()==0)
    {
      warning("You have to specify an output_filename\n");
      return true;
    }

  return false;
}

//! \brief parses parameters
/*! \warning Currently does not support interactive input, due to 
    the use of the 'next shape' keyword.
*/
GenerateImage::
GenerateImage(const char * const par_filename)
{
  initialise_keymap();
  set_defaults();
  if (par_filename!=0)
    parse(par_filename) ;
  else
    ask_parameters();

#if 0
  // doesn't work due to problem  Shape3D::parameter_info being ambiguous
  for (vector<shared_ptr<Shape3D> >::const_iterator iter = shape_ptrs.begin();
       iter != shape_ptrs.end();
       ++iter)
    {
      std::cerr << (**iter).parameter_info() << '\n';
    }
#endif

}
  

Succeeded
GenerateImage::
compute()
{
  const CartesianCoordinate3D<int> num_samples(5,5,5);
#if 0
  shared_ptr<DiscretisedDensity<3,float> > density_ptr = 
    DiscretisedDensity<3,float>::read_from_file(template_filename);
  shared_ptr<DiscretisedDensity<3,float> > out_density_ptr = 
    density_ptr->clone();
  out_density_ptr->fill(0);
  VoxelsOnCartesianGrid<float> current_image =
    dynamic_cast<VoxelsOnCartesianGrid<float>& >(*density_ptr);

#else
  VoxelsOnCartesianGrid<float> 
    current_image(IndexRange3D(0,output_image_size_z-1,
			       -(output_image_size_xy/2),
			       -(output_image_size_xy/2)+output_image_size_xy-1,
			       -(output_image_size_xy/2),
			       -(output_image_size_xy/2)+output_image_size_xy-1),
		  CartesianCoordinate3D<float>(0,0,0),
		  CartesianCoordinate3D<float>(output_voxel_size_z,
					       output_voxel_size_xy,
					       output_voxel_size_xy));
  shared_ptr<DiscretisedDensity<3,float> > out_density_ptr = 
    current_image.clone();
#endif
  vector<float >::const_iterator value_iter = values.begin();
  for (vector<shared_ptr<Shape3D> >::const_iterator iter = shape_ptrs.begin();
       iter != shape_ptrs.end();
       ++iter, ++value_iter)
    {
      std::cerr << "Next shape\n"; //(**iter).parameter_info() << '\n';
      current_image.fill(0);
      (**iter).construct_volume(current_image, num_samples);
      current_image *= *value_iter;
      *out_density_ptr += current_image;
    }
  DefaultOutputFileFormat output_file_format;
  return output_file_format.write_to_file(output_filename, *out_density_ptr);  
  
}



/************************ main ************************/


int main(int argc, char * argv[])
{
  
  if ( argc!=2) {
    cerr << "Usage: " << argv[0] << " par_file\n";
    exit(EXIT_FAILURE);
  }
  GenerateImage application(argc==2 ? argv[1] : 0);
  Succeeded success = application.compute();

  return success==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
