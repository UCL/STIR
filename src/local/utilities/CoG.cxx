//
// $Id$
//


#include "DiscretisedDensity.h"
#include "VoxelsOnCartesianGrid.h"
#include "shared_ptr.h"
#include "CartesianCoordinate3D.h"
#include "CartesianCoordinate2D.h"
#include <iostream>
#include <iomanip>
#include <numeric>
#include <fstream>


#ifndef TOMO_NO_NAMESPACES
using std::ofstream;
using std::cout;
using std::setw;
using std::cerr;
using std::endl;
#endif

USING_NAMESPACE_TOMO

template <class T> CartesianCoordinate3D<T> 
find_centre_of_gravity(const Array<2,T>& plane);


int
main(int argc, char *argv[])
{
 if (argc!=2)
   {
      cerr <<"Usage: " << argv[0] << "  inputfile_name\n";
      cerr <<"WARNING: output will be in files CoG.x, CoG.y\n";
      return (EXIT_FAILURE);
   }

  shared_ptr<DiscretisedDensity<3,float> > 
    density_ptr =  DiscretisedDensity<3,float>::read_from_file(argv[1]);
  VoxelsOnCartesianGrid<float> * const image_ptr =
    dynamic_cast<VoxelsOnCartesianGrid<float> * const>(density_ptr.get());


  const CartesianCoordinate3D<float> voxel_size = image_ptr->get_voxel_size();

#if 0
  cout << setw(5) <<"Plane" << setw(12) << "x" << setw(12) << "y\n";
  for (int z=image_ptr->get_min_index(); z<=image_ptr->get_max_index(); z++)
  {
    CartesianCoordinate3D<float> CoG = find_centre_of_gravity((*image_ptr)[z]);
    cout << setw(5) << z << setw(12) << CoG.x() << setw(12) << CoG.y() << endl;
  }
#else
  VectorWithOffset< CartesianCoordinate3D<float> > 
    allCoG(image_ptr->get_min_index(), image_ptr->get_max_index());

  for (int z=image_ptr->get_min_index(); z<=image_ptr->get_max_index(); z++)
  {
    allCoG[z] = find_centre_of_gravity((*image_ptr)[z]);
    allCoG[z].z() = z;
    allCoG[z] *= voxel_size;
  }
  {
    ofstream xout("CoG.x");
    xout << "x\n";
    xout << image_ptr->get_length() << "\n";
    for (int z=image_ptr->get_min_index(); z<=image_ptr->get_max_index(); z++)
      xout << allCoG[z].z() << "\n";
    for (int z=image_ptr->get_min_index(); z<=image_ptr->get_max_index(); z++)
      xout << allCoG[z].x() << "\n";
    for (int z=image_ptr->get_min_index(); z<=image_ptr->get_max_index(); z++)
    xout << 1 << "\n";
  }

  {
    ofstream yout("CoG.y");

    yout << image_ptr->get_length() << "\n";
    for (int z=image_ptr->get_min_index(); z<=image_ptr->get_max_index(); z++)
      yout << allCoG[z].z() << "\n";
    for (int z=image_ptr->get_min_index(); z<=image_ptr->get_max_index(); z++)
      yout << allCoG[z].y() << "\n";
    for (int z=image_ptr->get_min_index(); z<=image_ptr->get_max_index(); z++)
      yout << 1 << "\n";
  }
  

#endif

  return EXIT_SUCCESS;

}


  /***************** Miscellaneous Functions  *******/

template <class T>
T
find_unweighted_centre_of_gravity(const Array<1,T>& row)
{
  T CoG = 0;
  for (int x=row.get_min_index(); x<=row.get_max_index(); x++)
    CoG += x*row[x];
  return CoG;
}

template <class T>
CartesianCoordinate3D<T> find_unweighted_centre_of_gravity(const Array<2,T>& plane)
{
  CartesianCoordinate3D<T> CoG(0,0,0);
  
  CartesianCoordinate2D<int> min_indices, max_indices;
  if (!plane.get_regular_range(min_indices, max_indices))
    error("Can handle only square arrays\n");

  Array<1,T> sum_over_y(min_indices.x(), max_indices.x());
  Array<1,T> sum_over_x(min_indices.y(), max_indices.y());


  for (int x=min_indices.x(); x<=max_indices.x(); x++)
    for (int y=min_indices.y(); y<=max_indices.y(); y++)
    {
      sum_over_y[x] += plane[y][x];
      sum_over_x[y] += plane[y][x];
    }

  CoG.x() = find_unweighted_centre_of_gravity(sum_over_y);
  CoG.y() = find_unweighted_centre_of_gravity(sum_over_x);
  
  return CoG;
}

template <class T>
CartesianCoordinate3D<T> find_centre_of_gravity(const Array<2,T>& plane)
{
  T sum = plane.sum();
  // TODO different way of error checking 
  if (sum == 0)
    error("Warning: find_centre_of_gravity cannot properly normalise, as data sum to 0\n");
  CartesianCoordinate3D<T> CoG = find_unweighted_centre_of_gravity(plane);
  CoG.x() /= sum;
  CoG.y() /= sum;
  CoG.z() /= sum;
  return CoG;

}
