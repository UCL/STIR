/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/interfile.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/utilities.h"

#include "stir/IndexRange2D.h"
#include "stir/Array.h"
#include "stir/recon_array_functions.h"
#include "stir/ArrayFunction.h"

#include <iostream> 
#include <fstream>
#include <numeric>

#define SMALL_VALUE = 0.0000001;

#ifndef TOMO_NO_NAMESPACES
using std::iostream;
using std::ofstream;
using std::ifstream;
using std::ios;

using std::cerr;
using std::endl;
using std::cout;
#endif


// Date:21/11/2001



/* compute_sum_images_to_random_power: image_out = sum_i ( X_i)^power
*/


USING_NAMESPACE_STIR


int main(int argc, char *argv[])
{ 
  VoxelsOnCartesianGrid<float> input_image;
  VoxelsOnCartesianGrid<float> image_to_power;
  VoxelsOnCartesianGrid<float> cum_image;
  
  if (argc!= 3)
  {
    cerr << "Usage: lists of images <list> output_filename power"<<endl<<endl;    
   return EXIT_FAILURE;
  }
  ifstream in (argv[1]);  
  int power = atoi(argv[2]);
  
  const int maxsize_length = 100;
  char filename[80];

  image_to_power = (*input_image.get_empty_voxels_on_cartesian_grid());
  
  if (!in)
  {
    cout<< "Cannot open input files.\n";
    return 1;
  }


  int counter =0;
  while(!in.eof())
  {      
    counter++;
    in.getline(filename,maxsize_length);
    if(strlen(filename)==0)
      break;
    input_image= (*read_interfile_image(filename));
    image_to_power = input_image;

    if ( power !=1 )
    {
    for ( int i = 1;i<=power-1;i++)
    {
     image_to_power *= input_image;
    }

    }

    cum_image +=image_to_power;
      
   
  }

  //cum_image/=counter;
  
  write_basic_interfile(argv[2], cum_image);

  
  
  return EXIT_SUCCESS;
  
}