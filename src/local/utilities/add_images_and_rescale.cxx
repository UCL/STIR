//
// $Id$
//

/*!
  \file
  \ingroup utilities

  \brief  Rescale images: Given the lists of images rescale them with a scaling factor

  \deprecated. use stir_math

  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/

#include "stir/IO/interfile.h"
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









int main(int argc, char *argv[])
{ 
USING_NAMESPACE_STIR

  VoxelsOnCartesianGrid<float> input_image_1;
  VoxelsOnCartesianGrid<float> cum_image;
  
  if (argc!= 3)
  {
    cerr << "Usage: lists of images <list> output_filename"<<endl<<endl;    
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


  int counter =0;
  while(!in_1.eof())
  {      
    cerr << " Doing: " << counter <<endl;
    counter++;
    in_1.getline(filename,maxsize_length);
    if(strlen(filename)==0)
      break;
    input_image_1= (*read_interfile_image(filename));
    cum_image +=input_image_1;
    
    char buffer[20];
    //string file =filename;
    //int str_len = strlen(file.c_str());
    //file.insert(str_len-3,"rescaled_400");
    //int str_lennew = strlen(file.c_str());
   
  }
  cerr << "dividing with" << counter << endl;
  cum_image/=counter;
  
   write_basic_interfile(argv[2], cum_image);

  
  
  return EXIT_SUCCESS;
  
}

