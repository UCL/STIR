//
// $Id$
//
/*!

  \file
  \ingroup test
  
  \brief Test program for ROI functionality (and  bit of Shape3D hierarchy)
    
  \author Kris Thielemans
      
   $Date$        
   $Revision$
*/
/*
  Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
  See STIR/LICENSE.txt for details
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/Shape/EllipsoidalCylinder.h"
#include "stir/Shape/Ellipsoid.h"
#include "stir/evaluation/ROIValues.h"
#include "stir/evaluation/compute_ROI_values.h"
#include "stir/IndexRange.h"
#include "stir/RunTests.h"
//#include "stir/display.h"
#include <iostream>

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for compute_ROI_values and Shape3D hierarchy

  \todo Tests are currently simplistic
*/
class ROITests : public RunTests
{
public:
  void run_tests();
private:
  // warning fills /changes image and shape
  void run_tests_one_shape(Shape3D& shape,
		      VoxelsOnCartesianGrid<float>& image);
};

void
ROITests::run_tests_one_shape(Shape3D& shape,
			      VoxelsOnCartesianGrid<float>& image)
{
    shape.construct_volume(image, Coordinate3D<int>(1,1,1));

    const CartesianCoordinate3D<float> voxel_size = image.get_voxel_size();
    const float voxel_volume = voxel_size.x() * voxel_size.y() * voxel_size.z();

    image *= 2;
    //display(image);
    {
      const ROIValues ROI_values =
	compute_total_ROI_values(image, shape, Coordinate3D<int>(1,1,1));
      
      check_if_equal(ROI_values.get_mean(), 2.F, "ROI mean");
      check_if_equal(ROI_values.get_stddev(), 0.F, "ROI stddev");
      check_if_equal(ROI_values.get_max(), 2.F, "ROI max");
      check_if_equal(ROI_values.get_min(), 2.F, "ROI min");
      // check volume (needs casts as Shape3D does not have get_geometric_volume()
      // test supposes that shape is entirely within the volume
      const double old_tolerance = get_tolerance();
      if (dynamic_cast<EllipsoidalCylinder*>(&shape)!=0)
	{
	  EllipsoidalCylinder const& geoshape = static_cast<EllipsoidalCylinder const&>(shape);
	  set_tolerance(geoshape.get_geometric_area()/pow(voxel_volume,2.F/3)*.1);
	  check_if_equal(ROI_values.get_roi_volume(), 
			 geoshape.get_geometric_volume(), "ROI volume");
	}
      if (dynamic_cast<Ellipsoid*>(&shape)!=0)
	{
	  Ellipsoid const& geoshape = static_cast<Ellipsoid const&>(shape);
	  set_tolerance(geoshape.get_geometric_area()/pow(voxel_volume,2.F/3)*.1);
	  check_if_equal(ROI_values.get_roi_volume(), 
			 geoshape.get_geometric_volume(), "ROI volume");
	}
      set_tolerance(old_tolerance);
    }

    const CartesianCoordinate3D<float> translation (3,1,20);  
    image.set_origin(image.get_origin()+translation);
    shape.translate(translation);
    {
      const ROIValues ROI_values =
	compute_total_ROI_values(image, shape, Coordinate3D<int>(1,1,1));
      
      check_if_equal(ROI_values.get_mean(), 2.F, "ROI mean after translation");
      check_if_equal(ROI_values.get_stddev(), 0.F, "ROI stddev after translation");
      check_if_equal(ROI_values.get_max(), 2.F, "ROI max after translation");
      check_if_equal(ROI_values.get_min(), 2.F, "ROI min after translation");
      // check volume (needs casts as Shape3D does not have get_geometric_volume()
      // test supposes that shape is entirely within the volume
      const double old_tolerance = get_tolerance();
      if (dynamic_cast<EllipsoidalCylinder*>(&shape)!=0)
	{
	  EllipsoidalCylinder const& geoshape = static_cast<EllipsoidalCylinder const&>(shape);
	  set_tolerance(geoshape.get_geometric_area()/pow(voxel_volume,2.F/3)*.1);
	  //std::cerr << "tol " << get_tolerance();
	  check_if_equal(ROI_values.get_roi_volume(), 
			 geoshape.get_geometric_volume(), "ROI volume");
	}
      if (dynamic_cast<Ellipsoid*>(&shape)!=0)
	{
	  Ellipsoid const& geoshape = static_cast<Ellipsoid const&>(shape);
	  set_tolerance(geoshape.get_geometric_area()/pow(voxel_volume,2.F/3)*.1);
	  //std::cerr << "tol " << get_tolerance();
	  check_if_equal(ROI_values.get_roi_volume(), 
			 geoshape.get_geometric_volume(), "ROI volume");
	}
      set_tolerance(old_tolerance);
    }
}

void
ROITests::run_tests()

{ 
  cerr << "Tests for compute_ROI_values and Shape3D hierarchy\n";
  
  CartesianCoordinate3D<float> origin (0,0,0);  
  CartesianCoordinate3D<float> grid_spacing (3,4,5); 
  
  const IndexRange<3> 
    range(Coordinate3D<int>(0,-45,-44),
          Coordinate3D<int>(24,44,45));
  VoxelsOnCartesianGrid<float>  image(range,origin, grid_spacing);

  {
    // object at centre of image
    EllipsoidalCylinder
      cylinder(/*length*/image.size()*grid_spacing.z()/3, 
	       /*radius_x*/image[0][0].size()*grid_spacing.x()/4,
	       /*radius_y*/image[0].size()*grid_spacing.y()/4,
	       /*centre*/CartesianCoordinate3D<float>((image.get_min_index()+image.get_max_index())/2*grid_spacing.z(), 0,0),
	       /*Euler angles*/0,0,0);
    run_tests_one_shape(cylinder, image);
  }
  image.set_origin(origin);
  {
    // object at centre of image
    Ellipsoid
      ellipsoid(/*radius_x*/image[0][0].size()*grid_spacing.x()/4,
	       /*radius_y*/image[0].size()*grid_spacing.y()/5,
		/*radius_z*/image.size()*grid_spacing.z()/3,
	       /*centre*/CartesianCoordinate3D<float>((image.get_min_index()+image.get_max_index())/2*grid_spacing.z(), 0,0),
	       /*Euler angles*/0,0,0);
    run_tests_one_shape(ellipsoid, image);
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
