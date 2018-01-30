#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/interfile.h"
#include "stir/IO/read_from_file.h"

#include <vector>
#include <algorithm>
#include <iostream>
#ifndef STIR_NO_NAMESPACES
using std::string;
using std::cerr;
using std::min;
using std::max;
using std::endl;
using std::vector;
#endif

USING_NAMESPACE_STIR


int 
main(int argc, char * argv[])
{

  if (argc !=5)
  {
    cerr << " Usage:  " << argv[0]<< " Output filename, input image, number of planes-min, number of planes-right " << endl;
    return EXIT_FAILURE;
  }

  const string input_filename = argv[1];
  shared_ptr<DiscretisedDensity<3,float> > input_image_sptr(read_from_file<DiscretisedDensity<3,float> >(argv[2]));  
  const int number_of_planes_min = atoi(argv[3]);
  const int number_of_planes_max = atoi(argv[4]);

  VoxelsOnCartesianGrid<float> *  input_image_sptr_vox= 
    dynamic_cast<VoxelsOnCartesianGrid<float> *> (input_image_sptr.get());
  
  VoxelsOnCartesianGrid<float> * output_image;
  output_image =input_image_sptr_vox->get_empty_voxels_on_cartesian_grid();
  output_image->grow_z_range(output_image->get_min_z()-number_of_planes_min, output_image->get_max_z()+number_of_planes_max);

  for (int k=input_image_sptr_vox->get_min_z();k<=input_image_sptr_vox->get_max_z();k++)   
    for (int j =input_image_sptr_vox->get_min_y();j<=input_image_sptr_vox->get_max_y();j++)
      for (int i =input_image_sptr_vox->get_min_x();i<=input_image_sptr_vox->get_max_x();i++)	
      {
	(*output_image)[k][j][i] = (*input_image_sptr_vox)[k][j][i];

      }
   
   write_basic_interfile(argv[1], *output_image);

  return EXIT_SUCCESS;
}
