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

  \brief Utility program for calculating recovery coeficient values in the image
            quality phantom described in NEMA NU 4.

  \author Parisa Khateri

*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Shape/EllipsoidalCylinder.h"
#include "stir/CartesianCoordinate2D.h"
#include "stir/CartesianCoordinate3D.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include "stir/IO/read_from_file.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/Succeeded.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ofstream;
using std::vector;
#endif

USING_NAMESPACE_STIR


class FindRecoveryCoefficient : public ParsingObject
{
public:

  FindRecoveryCoefficient(const char * const par_filename);
  Succeeded compute();

private:
  typedef ParsingObject base_type;
  int min_index_for_average_plane; //min plane number
  int max_index_for_average_plane; //max plane number
  float mean_uniform_region;
  float STD_uniform_region;
  int start_z_of_rods; //axial position of start point of the rods
  int stop_z_of_rods; //axial position of end point of the rods
  std::vector<double> ROIs_x; // order is from the smallest ROI to the largest
  std::vector<double> ROIs_y; // order is from the smallest ROI to the largest
  std::string input_filename;
  std::string output_filename;

  void initialise_keymap();
  void set_defaults();
};

void FindRecoveryCoefficient::
initialise_keymap()
{
  this->parser.add_start_key("FindRecoveryCoefficient Parameters");
  this->parser.add_key("input filename",&input_filename);
  this->parser.add_key("output filename",&output_filename);
  this->parser.add_key("minimun index to calculate the average plane", &min_index_for_average_plane);
  this->parser.add_key("maximum index to calculate the average plane", &max_index_for_average_plane);
  this->parser.add_key("mean value of uniform region", &mean_uniform_region);
  this->parser.add_key("STD of uniform region", &STD_uniform_region);
  this->parser.add_key("start index of rods (z value)", &start_z_of_rods);
  this->parser.add_key("stop index of rods (z value)", &stop_z_of_rods);
  this->parser.add_key("x coordinates of center of ROIs (in mm)", &ROIs_x);
  this->parser.add_key("y coordinates of center of ROIs (in mm)", &ROIs_y);
  this->parser.add_stop_key("END FindRecoveryCoefficient Parameters");
}

void FindRecoveryCoefficient::
set_defaults()
{
  // specify defaults for the parameters in case they are not set.
  min_index_for_average_plane= 88;
  max_index_for_average_plane= 98;
  mean_uniform_region = 0.00569557;
  STD_uniform_region = 0.00107216;
  start_z_of_rods = 86;
  stop_z_of_rods = 102;
  ROIs_x = {2.16312,-5.66312,-5.66312,2.16312,7.};
  ROIs_y = {6.6574,4.1145,-4.1145,-6.6574,0.};
  input_filename.resize(0);
  output_filename.resize(0);
}

FindRecoveryCoefficient::
FindRecoveryCoefficient(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    {
      if (parse(par_filename) == false)
  exit(EXIT_FAILURE);
    }
  else
    ask_parameters();
}


void
compute_average_plane_in_given_range(VoxelsOnCartesianGrid<float>& average_plane,
                      const VoxelsOnCartesianGrid<float>& image,
                      const int& min_index, const int& max_index)
{
  int min_z = image.get_min_z();
  int max_z = image.get_max_z();
  int min_y = image.get_min_y();
  int min_x = image.get_min_x();
  int max_y = image.get_max_y();
  int max_x = image.get_max_x();
  if (min_index<min_z || max_index>max_z)
  {
    error("min_index & max_index are not in the range of provided image!");
  }


  shared_ptr<VoxelsOnCartesianGrid<float> > sum_planes_ptr(image.get_empty_voxels_on_cartesian_grid());
  VoxelsOnCartesianGrid<float> sum_planes = *sum_planes_ptr;
  int num_planes = max_index - min_index + 1;

  for (int z=min_index; z<=max_index; z++)
    for(int y=min_y; y<=max_y; y++)
      for(int x=min_x; x<=max_x; x++)
  {
      sum_planes[0][y][x] += image[z][y][x];
  }
  for(int y=min_y; y<=max_y; y++)
    for(int x=min_x; x<=max_x; x++)
  {
    average_plane[0][y][x] = sum_planes[0][y][x] / num_planes;
  }
}

void
build_ROIs(vector<EllipsoidalCylinder> & ROIs, float length_z, std::vector<double> ROIs_x, std::vector<double> ROIs_y)
{
  //note: the y coordinate is (-1*y) of coordinates in your images. STIR assumes y axis downward.
  vector<CartesianCoordinate2D<float>> ROI_centers{CartesianCoordinate2D<float>(ROIs_y[0],ROIs_x[0]), //(y,x)
                                                   CartesianCoordinate2D<float>(ROIs_y[1],ROIs_x[1]),
                                                   CartesianCoordinate2D<float>(ROIs_y[2],ROIs_x[2]),
                                                   CartesianCoordinate2D<float>(ROIs_y[3],ROIs_x[3]),
                                                   CartesianCoordinate2D<float>(ROIs_y[4],ROIs_x[4])};
  for (int i = 0; i<5; i++)
  {
    float d = 2*(i+1); //ROI diameter in mm
    float radius_y =d/2.;
    float radius_x = d/2.;
    CartesianCoordinate3D<float> centre(0, ROI_centers[i].y(), ROI_centers[i].x());
    EllipsoidalCylinder current_ROI(length_z, radius_y, radius_x, centre);
    ROIs.push_back(current_ROI);
  }
}


void
find_max_in_ROI(float &max, CartesianCoordinate3D<int>& max_coord,
                const VoxelsOnCartesianGrid<float>& image,
                const EllipsoidalCylinder& ROI)
{
  bool change = 0; // to make sure there is any non-zero voxel
  const int min_y = image.get_min_y();
  const int min_x = image.get_min_x();
  const int max_y = image.get_max_y();
  const int max_x = image.get_max_x();

  shared_ptr<VoxelsOnCartesianGrid<float> >
            discretised_shape_ptr(image.get_empty_voxels_on_cartesian_grid());
  ROI.construct_volume(*discretised_shape_ptr, Coordinate3D<int>(1,1,1)); // TODO number_samples=1, make it general
  VoxelsOnCartesianGrid<float> discretised_shape = *discretised_shape_ptr;

  max = std::numeric_limits<float>::lowest();

  //iterate only over y , x. because the image is actualy the average plane in which only z=0 is valid
  for(int y=min_y; y<=max_y; y++)
    for(int x=min_x; x<=max_x; x++)
  {
    int z=0;
    const float weight = discretised_shape[z][y][x];
  	if (weight ==0)
      continue;

    change = 1;
    CartesianCoordinate3D<int> current_index(z,y,x);
    const float current_value = image[z][y][x];
    if (current_value>max)
    {
      max=current_value;
      max_coord = current_index;
    }
  }
  if (change ==0)
    error("Max was not found in this range. All voxels are zero\n");
}

void
find_max_in_all_ROIs(vector<float>& maxs,
                     vector<CartesianCoordinate3D<int>>& max_coords,
                     const VoxelsOnCartesianGrid<float>& image,
                     const vector<EllipsoidalCylinder>& ROIs)
{
  for (unsigned int i=0; i<ROIs.size(); i++)
  {
    float max;
    CartesianCoordinate3D<int> max_coord;
    find_max_in_ROI(max, max_coord, image, ROIs[i]);
    maxs.push_back(max);
    max_coords.push_back(max_coord);
  }
}

void
find_mean_STD_along_lineprofile(float & mean, float & STD,
                               const VoxelsOnCartesianGrid<float>& image,
                               char direction, int min_index, int max_index,
                               const CartesianCoordinate3D<int>& pos)
{
  float sum = 0;
  float variance = 0;
  switch(direction)
  {
    case 'z':
    {
      for (int z=min_index; z<=max_index; z++)
      {
        sum += image[z][pos.y()][pos.x()];
      }

      mean = sum/(max_index-min_index+1);

      for (int z=min_index; z<=max_index; z++)
      {
        variance += pow((image[z][pos.y()][pos.x()] - mean), 2);
      }

      variance /= (max_index-min_index+1);
      STD = sqrt(variance);
      break;
    }
    case 'y':
    {
      for (int y=min_index; y<=max_index; y++)
      {
        sum += image[pos.z()][y][pos.x()];
      }

      mean = sum/(max_index-min_index+1);

      for (int y=min_index; y<=max_index; y++)
      {
        variance += pow((image[pos.z()][y][pos.x()] - mean), 2);
      }

      variance /= (max_index-min_index+1);
      STD = sqrt(variance);
      break;
    }
    case 'x':
    {
      for (int x=min_index; x<=max_index; x++)
      {
        sum += image[pos.z()][pos.y()][x];
      }
      mean = sum/(max_index-min_index+1);

      for (int x=min_index; x<=max_index; x++)
      {
        variance += pow((image[pos.z()][pos.y()][x] - mean), 2);
      }

      variance /= (max_index-min_index+1);
      STD = sqrt(variance);
      break;
    }
  }
}

Succeeded FindRecoveryCoefficient::
compute()
{
  shared_ptr<DiscretisedDensity<3,float> >
    density(read_from_file<DiscretisedDensity<3,float> >(input_filename));

  const VoxelsOnCartesianGrid<float>& image =
     dynamic_cast<const VoxelsOnCartesianGrid<float>&>(*density);

  ofstream out (output_filename +".txt");
  if (!out)
  {
    warning("Cannot open output file.\n");
    return Succeeded::no;
  }

  vector<float> ROI_maxs;
  vector<CartesianCoordinate3D<int>> ROI_max_coords;

  // average over 10 middle slices and save the result in one slice
  shared_ptr<VoxelsOnCartesianGrid<float> > average_plane_ptr(image.get_empty_voxels_on_cartesian_grid());
  VoxelsOnCartesianGrid<float> average_plane = *average_plane_ptr;
  info("Computing average plane");
  compute_average_plane_in_given_range(average_plane, image,
                  min_index_for_average_plane, max_index_for_average_plane);
  info("Done computing average plane");

  // draw circular ROIs around each rod with D=2 * rod_d then find max in each ROIs
  vector<EllipsoidalCylinder> ROIs;
  float length_z =  image.get_voxel_size().z();
  info("Building ROIs");
  build_ROIs(ROIs, length_z, ROIs_x, ROIs_y);
  info("Done building ROIs");

  //find maximum and its corresponding coordinate in ROIs
  info("Finding max in ROIs");
  find_max_in_all_ROIs(ROI_maxs, ROI_max_coords, average_plane, ROIs);
  info("Done find max in ROIs");

  // draw line profiles along the rods on the pixel coordinates with max ROI
  //find pixel values along line profiles
  //determine mean and standard deviation of the valuse.
  out<<"Results of recovery coeficient for "<<input_filename<<'\n';
  out<<"ROI_max"<<'\t'<<"ROI_max_voxel_coord (y,x)"<<'\t'<<"mean\t"<<"RC\t"<<"STD\t"<<"Percentage_STD_of_RC"<<'\n';
  info("Finding mean & STD");
  for (unsigned int i = 0; i < ROIs.size(); i++)
  {
    float mean;
    float RC;
    float STD;
    float percentage_STD;
    char direction = 'z';
    find_mean_STD_along_lineprofile(mean, STD, image,
                                    direction, start_z_of_rods, stop_z_of_rods, ROI_max_coords[i]);
    RC= mean/mean_uniform_region;
    STD /= mean_uniform_region;
    percentage_STD = 100*sqrt(pow(STD/RC, 2)
                                      +pow(STD_uniform_region/mean_uniform_region, 2));

    out<<std::setprecision(6)<<ROI_maxs[i]<<"\t("<<ROI_max_coords[i].y()<<", "<<ROI_max_coords[i].x()<<")\t";
    out<<std::setprecision(6)<<mean<<'\t'<<std::setprecision(6)<<RC
       <<'\t'<<std::setprecision(6)<<STD<<'\t'<<std::setprecision(6)<<percentage_STD<<'\n';
  }
  info("Done finding mean & STD");

  //TODO check when yes
  return Succeeded::yes;
}

int
main(int argc, char *argv[])
{
  const char * const progname = argv[0];

  if(argc!=2)
  {
    cerr<<"\nUsage: " << progname << "  parameter_file\n\n";
    //TODO add ask_parameters()

    exit(EXIT_FAILURE);
  }

  FindRecoveryCoefficient application(argc==2 ? argv[1] : 0);
  Succeeded success = application.compute();

  return success==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;

}
