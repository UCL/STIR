//
// $Id$
//
/*
  Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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

  \brief A simple program to test the stir::test_find_fwhm_in_image

  \author Pablo Aguiar
  \author Kris Thielemans

  $Date$

  $Revision$

  
  To run the test, you should use a command line argument with the name of a file.
  This should contain a number of test cases for the fit.
  
*/
  
#include "stir/RunTests.h"
#include "stir/ArrayFunction.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/SeparableCartesianMetzImageFilter.h"
//#include "stir/display.h"
#include "stir/find_fwhm_in_image.h"
#include "stir/Coordinate3D.h"
#include <fstream>
#include <iostream>
#include <string>
#include <strstream>
#include <list>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::ifstream;
using std::istream;
#endif

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief A simple class to test the find_fwhm_in_image function.
*/
class find_fwhm_in_imageTests : public RunTests
{
public:
  find_fwhm_in_imageTests() 
  {}
  void run_tests();
private:
  //istream& in;
};

void set_Gaussian_filter_fwhm(SeparableCartesianMetzImageFilter<float>& filter,
			      const float fwhm_z, 
			      const float fwhm_y,
			      const float fwhm_x)
{
  std::string buffer;
  std::stringstream parameterstream(buffer);

  parameterstream << "Separable Cartesian Metz Filter Parameters :=\n"
		    << "x-dir filter FWHM (in mm):= " << fwhm_x << "\n"
		    << "y-dir filter FWHM (in mm):= " << fwhm_y << "\n"
		    << "z-dir filter FWHM (in mm):= " << fwhm_z << "\n"
		    << "x-dir filter Metz power:= .0\n"
		    << "y-dir filter Metz power:= .0\n"
		    << "z-dir filter Metz power:=.0\n"
		    << "END Separable Cartesian Metz Filter Parameters :=\n";
    filter.parse(parameterstream);
}


void find_fwhm_in_imageTests::run_tests()
{  
  cerr << "Testing find_fwhm_in_image function..." << endl;

  set_tolerance(1.0);
 
  CartesianCoordinate3D<float> origin (0,1,2);  
  CartesianCoordinate3D<float> grid_spacing (2,1.4F,2.5F); 
  
  const int z_min=0, z_max=4;
  IndexRange<3> 
    range(CartesianCoordinate3D<int>(0,-65,-64),
          CartesianCoordinate3D<int>(24,64,65));
  


  VoxelsOnCartesianGrid<float>  image(range,origin, grid_spacing);
  SeparableCartesianMetzImageFilter<float> filter;
  set_Gaussian_filter_fwhm(filter, 12,14,12);

  {
    // point source
    image.fill(0);
    Coordinate3D<int> location_of_maximum(12,0,0);

    image[location_of_maximum] = 1;

    check_if_equal(image[location_of_maximum], 1.F,
		   "for parameter constant, should be equal");

    filter.apply(image);

    const std::list<ResolutionIndex<3,float> >  result =
      find_fwhm_in_image(image, 1, 2, 1);

    check(result.size() == 1, "check only 1 maximum for single point source case");

    std::list<ResolutionIndex<3,float> >::const_iterator current_result_iter =
      result.begin();
    
    check_if_equal(current_result_iter->voxel_location, location_of_maximum,
		   "check location of maximum for single point source case");
    check_if_equal(current_result_iter->resolution, 
		   Coordinate3D<float>(12,14,12)/image.get_voxel_size(),
		   "check resolution for single point source case");

    
  } 

  {
    // two spheres in a slice
    image.fill(0);
    Coordinate3D<int> location_of_maximum(12,0,0);
    image[location_of_maximum]=2;
    image[12][0][18]=1;
    set_Gaussian_filter_fwhm(filter, 12,14,12);
    filter.apply(image);
    const std::list<ResolutionIndex<3,float> >  result =
      find_fwhm_in_image(image, 1, 2, 1);

    check(result.size() == 1, "check only 1 maximum for 2 point sources in 1 slice");

    std::list<ResolutionIndex<3,float> >::const_iterator current_result_iter =
      result.begin();
    
    check_if_equal(current_result_iter->voxel_location, location_of_maximum,
		   "check location of maximum for 2 point sources in 1 slice");
    check_if_equal(current_result_iter->resolution, 
		   Coordinate3D<float>(12,14,12)/image.get_voxel_size(),
		   "check resolution for 2 point sources in 1 slice");
    // display(image, image.find_max());
  }

  {
    // two spheres in different slices
    image.fill(0);
    Coordinate3D<int> location_of_maximum(14,0,0);
    image[location_of_maximum]=2;
    image[3][0][0]=1;
    set_Gaussian_filter_fwhm(filter, 12,14,12);
    filter.apply(image);
    const std::list<ResolutionIndex<3,float> >  result =
      find_fwhm_in_image(image, 1, 2, 1);

    check(result.size() == 1, "check only 1 maximum for 2 point sources in different slices");

    std::list<ResolutionIndex<3,float> >::const_iterator current_result_iter =
      result.begin();
    
    check_if_equal(current_result_iter->voxel_location, location_of_maximum,
		   "check location of maximum for 2 point sources in different slices");
    check_if_equal(current_result_iter->resolution, 
		   Coordinate3D<float>(12,14,12)/image.get_voxel_size(),
		   "check resolution for 2 point sources in different slices");
    // display(image, image.find_max());
  }

  {
    // six spheres source
    image.fill(0.0005F);
    Coordinate3D<int> location_of_maximum1(12,0,0);
    image[location_of_maximum1] = 5;
   
    // VoxelsOnCartesianGrid<float>  other_image(image);
    //other_image.fill(0.0005);
    Coordinate3D<int> location_of_maximum2(18,32,32);
    //Coordinate3D<int> location_of_maximum3(6,-32,-32);
    image[location_of_maximum2]=3;
    //image[location_of_maximum3]=1.2;
    // other_image[24][10][0]=1;
    //other_image[0][0][10]=1;
    //other_image[14][64][0]=1;
    //other_image[8][0][65]=1;

    set_Gaussian_filter_fwhm(filter, 12,14,12);
    filter.apply(image);
   
    // image += other_image;
    //display(image, image.find_max());

    const std::list<ResolutionIndex<3,float> >  result =
      find_fwhm_in_image(image, 2, 2, 1);

    check(result.size() == 2, "check only 7 maxima");

    std::list<ResolutionIndex<3,float> >::const_iterator current_result_iter =
      result.begin();
    
    check_if_equal(current_result_iter->voxel_location, location_of_maximum1,
		   "check location of 1st maximum for 7 point source case");
    check_if_equal(current_result_iter->resolution, 
		   Coordinate3D<float>(12,14,12)/image.get_voxel_size(),
		   "check resolution for 1st maximum 7 point source case");
    ++current_result_iter;
    check_if_equal(current_result_iter->voxel_location, location_of_maximum2,
    	   "check location of 2nd maximum for 7 point source case");
    check_if_equal(current_result_iter->resolution, 
    	   Coordinate3D<float>(12,14,12)/image.get_voxel_size(),
    	   "check resolution for 2nd maximum 7 point source case");
    //++current_result_iter;
    //check_if_equal(current_result_iter->voxel_location, location_of_maximum3,
    //		   "check location of 3rd maximum for 7 point source case");
    //check_if_equal(current_result_iter->resolution, 
    //		   Coordinate3D<float>(4,4,4)/image.get_voxel_size(),
    //		   "check resolution for 3rd maximum 7 point source case");

    // display(image, image.find_max());
  }
  
  {
    // test line source in y-direction
    image.fill(0);
    const int z_location = 12;
    const int x_location = 0;
    for (int y=image[z_location].get_min_index(); 
	 y<=image[z_location].get_max_index();++y)
     	image[z_location][y][x_location] = 1;
    set_Gaussian_filter_fwhm(filter, 12,0,12);
    filter.apply(image); 
    const std::list<ResolutionIndex<3,float> >  result =
      find_fwhm_in_image(image, image[z_location].get_length(), 2, 2);
      display(image, image.find_max());
      check(result.size() == static_cast<unsigned>(image[z_location].get_length()) , 
	    "check only maximum in line source");
      int y=image[z_location].get_min_index();
      for(std::list<ResolutionIndex<3,float> >::const_iterator current_result_iter =
	   result.begin(); 
	   current_result_iter != result.end();
	   ++current_result_iter)
	{
	  Coordinate3D<int> location_of_maximum(z_location,y,x_location);
	  check_if_equal(current_result_iter->voxel_location, location_of_maximum,
			 "check location of maximum");
	  check_if_equal(current_result_iter->resolution, 
			 Coordinate3D<float>(12,0,12)/image.get_voxel_size(),
			 "check resolution");
          ++y;
	}

  }

  {
    // test line source in x-direction
     image.fill(0);
     const int z_location = 2;
     const int y_location = 0;
     for (int x=image[z_location][y_location].get_min_index(); 
          x<=image[z_location][y_location].get_max_index();++x)
       {
          image[z_location][y_location][x] = 1;
       }
     //display(image, image.find_max());
  }

  {
    // test line source in z-direction
    image.fill(0);

    const int y_location = 0;
    const int x_location = 0;
    for (int z=z_min; 
	 z<=z_max;++z)
      {
	image[z][y_location][x_location] = 1;
      }  
    // display(image, image.find_max());
  }

    /*
      const double old_tolerance = get_tolerance();
      set_tolerance(error_on_chi_square);
      check_if_equal(expected_chi_square, chi_square, 
		     "for parameter chi_square, should be equal");
    */
}


END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc != 1)
  {
    cerr << "Usage : " << argv[0] << " \n";
    return EXIT_FAILURE;
  }

  find_fwhm_in_imageTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
