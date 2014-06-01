//
//
/*
    Copyright (C) 2004- 2011, Hammersmith Imanet Ltd
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
/*!

  \file
  \ingroup test
  
  \brief Test program for ROI functionality (and a bit of stir::Shape3D hierarchy)
    
  \author Kris Thielemans
  \author C. Ross Schmidtlein (added stir::Box3D test)
      
*/

/*! 
 \def test_ROIs_DISPLAY
 Enable visual display of the ROIs
*/

// uncomment for visual display of the ROIs
//#define test_ROIs_DISPLAY

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/MinimalImageFilter3D.h"
#include "stir/Shape/EllipsoidalCylinder.h"
#include "stir/Shape/Ellipsoid.h"
#include "stir/Shape/Box3D.h"
#include "stir/Shape/DiscretisedShape3D.h"
#include "stir/evaluation/ROIValues.h"
#include "stir/evaluation/compute_ROI_values.h"
#include "stir/IndexRange.h"
#include "stir/RunTests.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"
#include "stir/make_array.h"
#include "stir/numerics/determinant.h"
#ifdef test_ROIs_DISPLAY
#include "stir/display.h"
#endif
#include <iostream>

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for compute_ROI_values and Shape3D hierarchy

  Visual tests can be enabled by setting the compiler define test_ROIs_DISPLAY.

  \todo Tests are currently somewhat simplistic
  
*/
class ROITests : public RunTests
{
public:
  void run_tests();
private:
  //! Run a series of tests for a shape
  /*!
    This function tests ROI values and Shape3D::geometric_volume before and after 
    translating and scaling the shape.
    \warning fills /changes image and shape
    \warning
    If you want to add new tests, the "scale" and "set_direction_vectors" test-code 
    does not work for all shapes as it assumes that the
    transformed shape is inside the original one.
    You can disable the "set_direction_vectors" test by setting  the do_rotated_ROI_test to false.
  */
  void run_tests_one_shape(Shape3D& shape,
			   VoxelsOnCartesianGrid<float>& image,
			   const bool do_rotated_ROI_test=true);
};

void
ROITests::run_tests_one_shape(Shape3D& shape,
			      VoxelsOnCartesianGrid<float>& image,
			      const bool do_rotated_ROI_test)
{
    shape.construct_volume(image, Coordinate3D<int>(1,1,1));

    if (dynamic_cast<DiscretisedShape3D const *>(&shape) != 0)
    {
#ifdef test_ROIs_DISPLAY
      shared_ptr<DiscretisedDensity<3,float> > copy_sptr =
	static_cast<DiscretisedShape3D&>(shape).get_discretised_density().clone();
#endif
      MinimalImageFilter3D<float> erosion_filter(make_coordinate(2,2,2));
      erosion_filter.apply(static_cast<DiscretisedShape3D&>(shape).get_discretised_density());
#ifdef test_ROIs_DISPLAY
      const float max = copy_sptr->find_max();
      *copy_sptr -= static_cast<DiscretisedShape3D&>(shape).get_discretised_density();
      *copy_sptr += max/2;
      display(*copy_sptr,
	      max,
	      "Original - erosion");
#endif      
    }
    const CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
    const float voxel_volume = voxel_size.x() * voxel_size.y() * voxel_size.z();

    image *= 2;
#ifdef test_ROIs_DISPLAY
    display(image, image.find_max(), "image corresponding to shape (before erosion in the discretised case)");
#endif

    {
      const ROIValues ROI_values =
	compute_total_ROI_values(image, shape, Coordinate3D<int>(1,1,1));
      
      check_if_equal(ROI_values.get_mean(), 2.F, "ROI mean");
      check_if_equal(ROI_values.get_stddev(), 0.F, "ROI stddev");
      check_if_equal(ROI_values.get_max(), 2.F, "ROI max");
      check_if_equal(ROI_values.get_min(), 2.F, "ROI min");
      // check volume 
      // test supposes that shape is entirely within the volume
      const float volume = shape.get_geometric_volume();
      if (volume>=0) // only test it if it's implemented
	{
	  const double old_tolerance = get_tolerance();
	  set_tolerance(pow(volume/voxel_volume,1.F/3)*.1);
	  check_if_equal(ROI_values.get_roi_volume(), volume, "ROI volume");
	  set_tolerance(old_tolerance);
	}
    }
    // test on translation (image and shape translate same way)
    {
      const CartesianCoordinate3D<float> translation (3,1,20);  
      image.set_origin(image.get_origin()+translation);
      shape.translate(translation);
      const ROIValues ROI_values =
	compute_total_ROI_values(image, shape, Coordinate3D<int>(1,1,1));
      
      check_if_equal(ROI_values.get_mean(), 2.F, "ROI mean after translation");
      check_if_equal(ROI_values.get_stddev(), 0.F, "ROI stddev after translation");
      check_if_equal(ROI_values.get_max(), 2.F, "ROI max after translation");
      check_if_equal(ROI_values.get_min(), 2.F, "ROI min after translation");
      // check volume 
      // test supposes that shape is entirely within the volume
      const float volume = shape.get_geometric_volume();
      if (volume>=0) // only test it if it's implemented
	{
	  const double old_tolerance = get_tolerance();
	  set_tolerance(pow(volume/voxel_volume,1.F/3)*.1);
	  check_if_equal(ROI_values.get_roi_volume(), volume, "ROI volume after translation");
	  set_tolerance(old_tolerance);
	}
      image.set_origin(image.get_origin()-translation);
      shape.translate(translation*-1);
    }
    // test on translation (image and shape translate separately)
    {
      const CartesianCoordinate3D<float> translation (3,1,10);  
      image.set_origin(image.get_origin()+translation);
      const ROIValues ROI_values =
	compute_total_ROI_values(image, shape, Coordinate3D<int>(1,1,1));
      image.set_origin(image.get_origin()-translation);
      shape.translate(translation*-1);
      const ROIValues ROI_values2 =
	compute_total_ROI_values(image, shape, Coordinate3D<int>(1,1,1));
      shape.translate(translation);
      
      check_if_equal(ROI_values.get_mean(), ROI_values2.get_mean(), "ROI mean after translation shape vs. image");
      check_if_equal(ROI_values.get_stddev(), ROI_values2.get_stddev(), "ROI stddev after translation shape vs. image");
      check_if_equal(ROI_values.get_max(), ROI_values2.get_max(), "ROI max after translation shape vs. image");
      check_if_equal(ROI_values.get_min(), ROI_values2.get_min(), "ROI min after translation shape vs. image");
    }
    // test on scaling (test only works if all scale factors < 1)
    if (dynamic_cast<DiscretisedShape3D const *>(&shape) == 0)
    {
      const float volume_before_scale = shape.get_geometric_volume();
      const CartesianCoordinate3D<float> scale(.5F,.9F,.8F);
      const float total_scale = scale[1]*scale[2]*scale[3];
      shared_ptr<Shape3D> new_shape_sptr(shape.clone());
      new_shape_sptr->scale_around_origin(scale);
      
      const ROIValues ROI_values =
	compute_total_ROI_values(image, *new_shape_sptr, Coordinate3D<int>(1,1,1));
      
      check_if_equal(ROI_values.get_mean(), 2.F, "ROI mean after scale");
      check_if_equal(ROI_values.get_stddev(), 0.F, "ROI stddev after scale");
      check_if_equal(ROI_values.get_max(), 2.F, "ROI max after scale");
      check_if_equal(ROI_values.get_min(), 2.F, "ROI min after scale");
      // check volume 
      // test supposes that shape is entirely within the volume
      const float volume = new_shape_sptr->get_geometric_volume();
      if (volume>=0) // only test it if it's implemented
	{
	  check_if_equal(volume_before_scale*total_scale, volume, "shape volume after scale");
	  const double old_tolerance = get_tolerance();
	  set_tolerance(pow(volume/voxel_volume,1.F/3)*.1);
	  check_if_equal(ROI_values.get_roi_volume(), volume, "ROI volume after scale");
	  set_tolerance(old_tolerance);
	}
#ifdef test_ROIs_DISPLAY
      VoxelsOnCartesianGrid<float> image2 = image;
      new_shape_sptr->construct_volume(image2, Coordinate3D<int>(1,1,1));
      image2 *= 4;
      image2 -= image;
      image2 += 2;
      display(image2, image2.find_max(), "(image corresponding to scaled (.5,.9,.8) shape)*2 - (original shape) + 1");
#endif
    }

    // test on setting direction vectors (test only works if new shape is smaller than original)
    if (dynamic_cast<Shape3DWithOrientation const *>(&shape) != 0)
    {
      const float volume_before_scale = shape.get_geometric_volume();
      // rotate over 45 degrees around 1 and scale
      const Array<2,float> direction_vectors=
	make_array(make_1d_array(1.F,0.F,0.F),
		   make_1d_array(0.F,1.F,1.F),
		   make_1d_array(0.F,-2.F,2.F));
      const float total_scale = 1/determinant(direction_vectors);
      shared_ptr<Shape3DWithOrientation> 
	new_shape_sptr(dynamic_cast<Shape3DWithOrientation *>(shape.clone()));
      check(new_shape_sptr->set_direction_vectors(direction_vectors) == Succeeded::yes, "set_direction_vectors");
      //std::cerr << new_shape_sptr->parameter_info();

      const ROIValues ROI_values =
	compute_total_ROI_values(image, *new_shape_sptr, Coordinate3D<int>(1,1,1));
      
      if (do_rotated_ROI_test)
	{
	  check_if_equal(ROI_values.get_mean(), 2.F, "ROI mean after changing direction vectors");
	  check_if_equal(ROI_values.get_stddev(), 0.F, "ROI stddev after changing direction vectors");
	  check_if_equal(ROI_values.get_min(), 2.F, "ROI min after changing direction vectors");
	}
      check_if_equal(ROI_values.get_max(), 2.F, "ROI max after changing direction vectors");
      // check volume 
      // test supposes that shape is entirely within the volume
      const float volume = new_shape_sptr->get_geometric_volume();
      if (volume>=0) // only test it if it's implemented
	{
	  check_if_equal(volume_before_scale*total_scale, volume, "shape volume after changing direction vectors");
	  const double old_tolerance = get_tolerance();
	  set_tolerance(pow(volume/voxel_volume,1.F/3)*.1);
	  check_if_equal(ROI_values.get_roi_volume(), volume, "ROI volume after changing direction vectors");
	  set_tolerance(old_tolerance);
	}
#ifdef test_ROIs_DISPLAY
      VoxelsOnCartesianGrid<float> image2 = image;
      new_shape_sptr->construct_volume(image2, Coordinate3D<int>(1,1,1));
      image2 *= 4;
      image2 -= image;
      image2 += 2;
      display(image2, image2.find_max(), "(image corresponding to rotated and scaled (x,y) shape)*2 - (original shape) + 1");
#endif
    }
    // test on parsing
    if (dynamic_cast<DiscretisedShape3D const *>(&shape) == 0)
    {
      shared_ptr<Shape3D> shape_sptr(shape.clone());
      KeyParser parser;
      parser.add_start_key("start");
      parser.add_stop_key("stop");
      parser.add_parsing_key("shape type", &shape_sptr);
      // construct stream with all info
      std::stringstream str;
      str << parser.parameter_info();
      // now read it back in and check
      if (check(parser.parse(str) && !is_null_ptr(shape_sptr), 
	       "parsing parameters failed"))
      {
	// check if it's what we expect
	if(!check(*shape_sptr == shape,
		  "parsed shape not equal to original"))
	  {
	    std::cerr << "Original: \n" << shape.parameter_info()
		      << "\nParsed: \n" << shape_sptr->parameter_info();
	  }
      }
  }
}

void
ROITests::run_tests()

{ 
  std::cerr << "Tests for compute_ROI_values and Shape3D hierarchy\n";
  
  CartesianCoordinate3D<float> origin (0,0,0);  
  CartesianCoordinate3D<float> grid_spacing (3,4,5); 
  
  const IndexRange<3> 
    range(Coordinate3D<int>(0,-45,-44),
          Coordinate3D<int>(24,44,45));
  VoxelsOnCartesianGrid<float>  image(range,origin, grid_spacing);

  /* WARNING:
     If you want to add new tests, the "scale" and "set_direction_vectors" test-code 
     in run_tests_one_shape() does not work for all shapes as it assumes that the
     transformed shape is inside the original one.
     You can disable the "set_direction_vectors" test by setting  the do_rotated_ROI_test to false.
  */
  {
    std::cerr << "\tTests with ellipsoidal cylinder.\n";
    // object at centre of image
    EllipsoidalCylinder
      cylinder(/*length*/image.size()*grid_spacing.z()/3, 
	       /*radius_x*/image[0][0].size()*grid_spacing.x()/4,
	       /*radius_y*/image[0].size()*grid_spacing.y()/4,
	       /*centre*/CartesianCoordinate3D<float>((image.get_min_index()+image.get_max_index())/2*grid_spacing.z(), 0,0));
    this->run_tests_one_shape(cylinder, image);
  }
  image.set_origin(origin);
  {
    std::cerr << "\tTests with ellipsoidal cylinder and wedge.\n";
    // object at centre of image
    EllipsoidalCylinder
      cylinder(/*length*/image.size()*grid_spacing.z()/3, 
	       /*radius_x*/image[0][0].size()*grid_spacing.x()/4,
	       /*radius_y*/image[0].size()*grid_spacing.y()/4,
	       /* theta_1 */ 10.F,
	       /* theta_2 */ 280.F,
	       /*centre*/CartesianCoordinate3D<float>((image.get_min_index()+image.get_max_index())/2*grid_spacing.z(), 0,0));
    this->run_tests_one_shape(cylinder, image, /*do_rotated_ROI_test=*/ false);
  }
  image.set_origin(origin);
  {
    std::cerr << "\tTests with ellipsoid.\n";
    // object at centre of image
    Ellipsoid
      ellipsoid(CartesianCoordinate3D<float>(/*radius_z*/image.size()*grid_spacing.z()/3,
					     /*radius_y*/image[0].size()*grid_spacing.y()/5,
					     /*radius_x*/image[0][0].size()*grid_spacing.x()/4),
		/*centre*/CartesianCoordinate3D<float>((image.get_min_index()+image.get_max_index())/2*grid_spacing.z(), 0,0));
    this->run_tests_one_shape(ellipsoid, image);
  }
  {
    std::cerr << "\tTests with Box3D.\n";
    // object at centre of image
    Box3D
      box(/*length_x*/image[0][0].size()*grid_spacing.x()/4,
	  /*length_y*/image[0].size()*grid_spacing.y()/5,
	  /*length_z*/image.size()*grid_spacing.z()/3,
	  /*centre*/CartesianCoordinate3D<float>((image.get_min_index()+image.get_max_index())/2*grid_spacing.z(), 0,0));
    this->run_tests_one_shape(box, image);
    }
  {
    std::cerr << "\tTests with DiscretisedShape3D.\n";
    // object at centre of image
    Ellipsoid
      ellipsoid(CartesianCoordinate3D<float>(/*radius_z*/image.size()*grid_spacing.z()/3,
					     /*radius_y*/image[0].size()*grid_spacing.y()/5,
					     /*radius_x*/image[0][0].size()*grid_spacing.x()/4),
		/*centre*/CartesianCoordinate3D<float>((image.get_min_index()+image.get_max_index())/2*grid_spacing.z(), 0,0));
    // note: it is important to use num_samples=(1,1,1) here, otherwise tests will fail
    // this is because the shape would have smooth edges, and the tests do not take 
    // that into account
    ellipsoid.construct_volume(image, make_coordinate(1,1,1));
    {
      DiscretisedShape3D discretised_shape(image);
      std::cerr << "\t\tidentical image\n";
      this->run_tests_one_shape(discretised_shape, image);
    }
    // need to fill in image again, as the tests change it
    ellipsoid.construct_volume(image, make_coordinate(1,1,1));
    {
      std::cerr << "\t\tNot-identical image\n";
      DiscretisedShape3D discretised_shape(image);
      CartesianCoordinate3D<float> other_origin (2,4,9);  
      CartesianCoordinate3D<float> other_grid_spacing (3.3,4.4,5.5); 
      
      const IndexRange<3> 
	other_range(Coordinate3D<int>(-1,-40,-43),
		    Coordinate3D<int>(25,45,47));
      VoxelsOnCartesianGrid<float>  other_image(other_range,other_origin, other_grid_spacing);
      this->run_tests_one_shape(discretised_shape, other_image);
    }

  }

}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR


int main()
{
  ROITests tests;
  tests.run_tests();
  return tests.main_return_value();
}
