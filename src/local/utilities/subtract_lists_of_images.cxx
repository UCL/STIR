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



/* Subtract lists of images: Given two lists of images find the difference. This will be used for 
resolution measurements.
*/
/*
    Copyright (C) 2000- $Year:$, IRSL
    See STIR/LICENSE.txt for details
*/


USING_NAMESPACE_STIR


int main(int argc, char *argv[])
{ 
  VoxelsOnCartesianGrid<float> input_image_1;
  VoxelsOnCartesianGrid<float> input_image_2;
   
  if (argc!= 4)
    cerr << "Usage: subtract lists of images list_1 list_2 output_file_name"<<endl<<endl;    
  
  ifstream in_1 (argv[1]);  
  ifstream in_2 (argv[2]);  
  
  const int maxsize_length = 100;
  char filename_1[80];
  char filename_2[80];
  
  if (!in_1 || !in_2)
  {
    cout<< "Cannot open input files.\n";
    return 1;
  }
  
  int counter = 10;
  while(!in_1.eof())
  {      
    in_1.getline(filename_1,maxsize_length);
    if(strlen(filename_1)==0)
      break;
    // int file_counter_1 = atoi(get_file_index(filename_1).c_str());
    
    in_2.getline(filename_2,maxsize_length);
    if(strlen(filename_2)==0)
      break;
    //int file_counter_2 = atoi(get_file_index(filename_2).c_str());
    
    input_image_1= (*read_interfile_image(filename_1));
    input_image_2= (*read_interfile_image(filename_2));
    
    input_image_1 -= input_image_2;
    const float delta = 0.01;
    input_image_1/=delta;
    
    char buffer[20];
    string file = argv[3];
   // file.append(itoa(counter,buffer,10));
    
    write_basic_interfile(file, input_image_1);
    counter+=10;
  
  }

  return EXIT_SUCCESS;
  
}