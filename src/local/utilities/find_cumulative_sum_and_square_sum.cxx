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

/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

// Date:07/02/2001


/* find_cumulative_sum_and_square_sum: Given the lists of images add them up and find
*/


USING_NAMESPACE_STIR


int main(int argc, char *argv[])
{ 
  VoxelsOnCartesianGrid<float> input_image;
  VoxelsOnCartesianGrid<float> cum_image;
  VoxelsOnCartesianGrid<float> cum_square;
  VoxelsOnCartesianGrid<float> square;
  
  if (argc!= 2)
  {
   cerr << "Usage: lists of images <list>"<<endl<<endl;    
   return EXIT_FAILURE;
  }
  ifstream in (argv[1]);  
  
  const int maxsize_length = 100;
  char filename[80];
  
  if (!in )
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
    square = input_image;
    square *= input_image;
    cum_square += square;
    cum_image +=input_image;    
    char buffer[20];    
  }

  cum_image/=counter;
  write_basic_interfile("cum_image", cum_image);
  write_basic_interfile("cum_square", cum_square);


  
  
  return EXIT_SUCCESS;
  
}