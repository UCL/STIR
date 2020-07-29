/*
Copyright 2017 ETH Zurich, Institute of Particle Physics and Astrophysics

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*!

  \file
  \ingroup utilities

  \brief Find count statistics in a cylindrical ROI for image and projection data

  \author Parisa Khateri

*/
#include "stir/utilities.h"
#include <iostream>
#include <fstream>
#include <string>
#include "stir/shared_ptr.h"
#include "stir/ProjData.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/LORCoordinates.h"
#include "stir/Bin.h"
#include "stir/ProjDataInterfile.h"
#include "stir/IO/interfile.h"
#include <limits>
#include "stir/Shape/EllipsoidalCylinder.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/read_from_file.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"


USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  bool is_projdata;
  CartesianCoordinate3D<float> centre(0,0,0);

  if (argc==5 && !strcmp(argv[4],"s"))
  {
      is_projdata = 1;
  }
  else if (argc==5 && !strcmp(argv[4],"v"))
  {
    is_projdata = 0;
  }
  else if (argc==8 && !strcmp(argv[4],"v"))
  {
    is_projdata = 0;
    centre.z() = atof(argv[5]);
    centre.y() = atof(argv[6]);
    centre.x() = atof(argv[7]);
  }
  else
  {
    std::cerr<<"\nError: wrong arguments\n"
        <<"Usage: "<< argv[0]
        <<" input_file cylinder_radius(mm) cylinder_hight(mm) s/v [z(mm) y(mm) x(mm)]\n"
        <<"use either s if the input is a projection data or v if the input is an image\n"
        <<"option: if the input is an image, the center of ROI can be deterimined."
        <<"Note that the center of Cartesian Coordinate is the center of scanner which is the default.\n"
        <<"warning: input order matters\n\n";
    return EXIT_FAILURE;
  }

  const double R = atof(argv[2]); // cylinder radius
  const double h = atof(argv[3]); // cylinder hight
  if (R<=0 || h<=0)
  {
    std::cerr <<"\nError: Radius and hight must be positive and larger than zero\n"
              <<"Usage: "<< argv[0]
              <<" input_file cylinder_radius(mm) cylinder_hight(mm) [p]\n"
              <<"use option p if the input is projection data\n"
              <<"warning: input order matters\n\n";
    return EXIT_FAILURE;
  }

  if (is_projdata)   //if the input is a projection data
  {
    shared_ptr<ProjData> projdata_ptr = ProjData::read_from_file(argv[1]);
    if (is_null_ptr(projdata_ptr))
    {
      std::cerr << "Could not read input file\n"; exit(EXIT_FAILURE);
    }

    CartesianCoordinate3D<float> c1, c2;
    LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;

    double total_count=0;
    double min_count=std::numeric_limits<float>::max(); // minimum number of counts per LOR
    double max_count=std::numeric_limits<float>::min(); // minimum number of counts per LOR
    double mean_count=0; //average number of counts per LOR in the active region
    int num_active_LORs=0; //number of LORs which pass through the cylinder
    for (int seg =projdata_ptr->get_min_segment_num(); seg <=projdata_ptr->get_max_segment_num(); ++seg)
      for (int view =projdata_ptr->get_min_view_num(); view <=projdata_ptr->get_max_view_num(); ++view)
    {
      Viewgram<float> cylinder_viewgram = projdata_ptr->get_viewgram(view, seg);
      for (int ax =projdata_ptr->get_min_axial_pos_num(seg); ax <=projdata_ptr->get_max_axial_pos_num(seg); ++ax)
        for (int tang =projdata_ptr->get_min_tangential_pos_num(); tang <=projdata_ptr->get_max_tangential_pos_num(); ++tang)
      {
        Bin bin(seg, view, ax, tang);
        projdata_ptr->get_proj_data_info_sptr()->get_LOR(lor, bin);
        LORAs2Points<float> lor_as2points(lor);
        LORAs2Points<float> intersection_coords;
        if (find_LOR_intersections_with_cylinder(intersection_coords, lor_as2points, R) ==Succeeded::yes)
        { //this only succeeds if LOR is intersecting with the infinitely long cylinder
          c1 = intersection_coords.p1();
          c2 = intersection_coords.p2();
          if (!( (c1.z()<-h/2. && c2.z()<-h/2.) || (c1.z()>h/2. && c2.z()>h/2.) ))
          {
            double N_lor = cylinder_viewgram[ax][tang]; //counts seen by this lor
            float c12 = sqrt( pow(c1.z()-c2.z(), 2)    // length of intersection of lor with the cylinder
                            + pow(c1.y()-c2.y(), 2)
                            + pow(c1.x()-c2.x(), 2) );
            if (c12>0.5) // if LOR intersection is lager than 0.5 mm, check the count per LOR
            {
              total_count+=N_lor;
              num_active_LORs+=1;
              if (N_lor<min_count) min_count=N_lor;
              if (N_lor>max_count) max_count=N_lor;
            }
          }
        }
      }
    }
    mean_count=total_count/num_active_LORs;
    std::cout<<"num_lor total_count mean_count min_count max_count : "
              <<num_active_LORs<<" "<<total_count<<" "<<mean_count<<" "
              <<min_count<<" "<<max_count<<"\n";
  }
  else   //if the input is an image
  {
    shared_ptr< DiscretisedDensity<3,float> > density_ptr
              = read_from_file<DiscretisedDensity<3,float> >(argv[1]);
    const VoxelsOnCartesianGrid<float> * image_ptr =
            	  dynamic_cast<VoxelsOnCartesianGrid<float> *>(density_ptr.get());
    CartesianCoordinate3D<float> voxel_size = image_ptr->get_voxel_size();

    if (is_null_ptr(density_ptr))
    {
      std::cerr << "Could not read input file\n"; exit(EXIT_FAILURE);
    }

    EllipsoidalCylinder cylinder_shape(h,R,R,centre);

    double total_count=0;
    double min_count=std::numeric_limits<float>::max(); // minimum number of counts per LOR
    double max_count=std::numeric_limits<float>::min(); // minimum number of counts per LOR
    double mean_count=0; //average number of counts per LOR in the active region
    double STD=0; //standard deviation of counts per LOR in the active region
    int num_voxels=0;
    DiscretisedDensity<3,float>& image = *density_ptr;
    const int min_k_index = image.get_min_index();
    const int max_k_index = image.get_max_index();
    for ( int k = min_k_index; k<= max_k_index; ++k)
    {
      const int min_j_index = image[k].get_min_index();
      const int max_j_index = image[k].get_max_index();
      for ( int j = min_j_index; j<= max_j_index; ++j)
      {
        const int min_i_index = image[k][j].get_min_index();
        const int max_i_index = image[k][j].get_max_index();
        for ( int i = min_i_index; i<= max_i_index; ++i)
        {
          /*
            [min_i_index,max_i_index]=[-num_voxel_x/2,num_voxel_x/2]
            [min_j_index,max_j_index]=[-num_voxel_y/2,num_voxel_y/2]
            [min_k_index,max_k_index]=[0,num_voxel_z-1]
          */

          CartesianCoordinate3D<float> voxel((k-max_k_index/2)*voxel_size.z(),
                                              j*voxel_size.y(),
                                              i*voxel_size.x());
          if (cylinder_shape.is_inside_shape(voxel))
          {
            total_count+=image[k][j][i];
            num_voxels++;
            if (image[k][j][i]<min_count) min_count=image[k][j][i];
            if (image[k][j][i]>max_count) max_count=image[k][j][i];
          }
        }
      }
    }
    mean_count=total_count/num_voxels;
    double sum_for_std=0;
    for ( int k = min_k_index; k<= max_k_index; ++k)
    {
      const int min_j_index = image[k].get_min_index();
      const int max_j_index = image[k].get_max_index();
      for ( int j = min_j_index; j<= max_j_index; ++j)
      {
        const int min_i_index = image[k][j].get_min_index();
        const int max_i_index = image[k][j].get_max_index();
        for ( int i = min_i_index; i<= max_i_index; ++i)
        {
          /*
            [min_i_index,max_i_index]=[-num_voxel_x/2,num_voxel_x/2]
            [min_j_index,max_j_index]=[-num_voxel_y/2,num_voxel_y/2]
            [min_k_index,max_k_index]=[0,num_voxel_z-1]
          */

          CartesianCoordinate3D<float> voxel((k-max_k_index/2)*voxel_size.z(),
                                              j*voxel_size.y(),
                                              i*voxel_size.x());
          if (cylinder_shape.is_inside_shape(voxel))
          {
            sum_for_std+=pow((image[k][j][i] - mean_count),2);
          }
        }
      }
    }
    STD=sqrt(sum_for_std/num_voxels);
    std::cout<<"num_voxels total_count mean_count STD min_count max_count : "
              <<num_voxels<<" "<<total_count<<" "<<mean_count<<" "
              <<STD<<" "<<min_count<<" "<<max_count<<"\n";
  }
}
