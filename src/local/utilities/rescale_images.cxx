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

#ifndef STIR_NO_NAMESPACES
using std::iostream;
using std::ofstream;
using std::ifstream;
using std::ios;

using std::cerr;
using std::endl;
using std::cout;
#endif


// Date:21/11/2001



/* Rescale images: Given the lists of images rescale them with a scaling factor
*/
/*
    Copyright (C) 2000- $Year:$, IRSL
    See STIR/LICENSE.txt for details
*/


USING_NAMESPACE_STIR


int main(int argc, char *argv[])
{ 
  VoxelsOnCartesianGrid<float> input_image_1;
  
  if (argc!= 3)
  {
    cerr << "Usage: lists of images <list> scale factor "<<endl<<endl;    
   return EXIT_FAILURE;
  }
  ifstream in_1 (argv[1]);  
  
  const int maxsize_length = 100;
  char filename[80];
  
  if (!in_1)
  {
    cout<< "Cannot open input files.\n";
    return 1;
  }
  
  while(!in_1.eof())
  {      
    in_1.getline(filename,maxsize_length);
    if(strlen(filename)==0)
      break;
    input_image_1= (*read_interfile_image(filename));
    const int scaling_factor = atoi(argv[2]);
    input_image_1/=scaling_factor;
    
    char buffer[20];
    string file =filename;
    int str_len = strlen(file.c_str());
    file.insert(str_len-3,"rescaled_50");
    int str_lennew = strlen(file.c_str());
    write_basic_interfile(file, input_image_1);

  } 
  
  
  return EXIT_SUCCESS;
  
}