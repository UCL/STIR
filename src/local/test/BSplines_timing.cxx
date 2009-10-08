//
// $Id$
//
/*
 Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
\brief Allows executing timing tests for the BSplinesRegularGrid class

\author Tim Borgeaud
\author Kris Thielemans
  
$Date$
$Revision$
*/  
//#define MYLINEAR

#include "stir/Array.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange3D.h"
#include "stir/IndexRange4D.h"

#include "local/stir/numerics/more_interpolators.h"

#include "stir/numerics/BSplines.h"
#include "stir/numerics/BSplinesRegularGrid.h"


#include <fstream>
#include <libgen.h>
#include <stdlib.h>
#include <sys/time.h>

using std::cerr;

using namespace stir;
using namespace BSpline;



const int MAX_DIMS = 4;
const int DEF_DIMS = 1;


const int MAX_NUM = 10000000;
const int MIN_SIZE = 5;
const int MAX_TOTAL_GRID_ELEMENTS = 10000000;

const int DEF_NUM = 1000;
const int DEF_SIZE = 10;


void usage(char *name) {
    
  cerr << "A program to test the BSplines interpolation implementation\n\n";
  
  cerr << "Usage: " << name << " [-n number of interpolations] [-s grid size]\n";
  cerr << "             [-i interpolator] [-2]\n\n";
  
  cerr << "   interpolator: near, linear, cubic, quadratic, omoms\n\n";
  
}



template<int dims> 
double interpolate(int num, int size, BSplineType bs_type, Array<dims, float> grid) {
  
  // Create interpolator.
  
  struct timeval start;
  gettimeofday(&start, NULL);  

  BSplinesRegularGrid<dims, float, float> bsi(grid, bs_type);
  
  struct timeval end;
  gettimeofday(&end, NULL);
  
  double total = (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec)/1000000.0);
  
  cout << "Constructed " << dims << " dimensional, " << size;
  for (int d = 1 ; d < dims ; d++ ) {
    cout << " x " << size;
  }
  cout << " element grid.\n";
  cout << "Time: " << total << " seconds\n";
  cout << "\n";
  

  BasicCoordinate<dims, double> pos;


  
  gettimeofday(&start, NULL);

  double foo = 0.0;

  // Now try some interpolation.
  for (int i = 0 ; i < num ; i++ ) {

    // Set position
    for (int d = 1 ; d < dims + 1 ; d++ ) {
      pos[d] = (static_cast<double>(size - 1)/(RAND_MAX)) * random();
      //cout << pos[d] << "\n";
    }

    // Get interpolation.
    foo += bsi(pos);
  }    
  

  gettimeofday(&end, NULL);

  total = (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec)/1000000.0);
  

  cout << "Interpolations: " << num << "\n";
  cout << "Time: " << total << " seconds\n";
  cout << "\n";

  //////////////
#ifdef MYLINEAR  
  gettimeofday(&start, NULL);

  foo = 0.0;

  // Now try some interpolation.
  for (int i = 0 ; i < num ; i++ ) {

    // Set position
    for (int d = 1 ; d < dims + 1 ; d++ ) {
      pos[d] = (static_cast<double>(size - 1)/(RAND_MAX)) * random();
      //cout << pos[d] << "\n";
    }

    // Get interpolation.
    foo += pull_linear_interpolate(grid,pos);
  }    

  gettimeofday(&end, NULL);

  total = (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec)/1000000.0);
  

  cout << "linear Time: " << total << " seconds\n";
  cout << "\n";
#endif //MYLINEAR

  //////////////
  cout << "foo: " << foo << "\n\n";

  return(total);
}






int main(int argc, char **argv) {

  int grid_size = DEF_SIZE;
  BSplineType bs_type = near_n;
  int num = DEF_NUM;
  int dimensions = DEF_DIMS;

  char opt_ch;

  // Options.
  while ((opt_ch = getopt(argc, argv, "n:s:i:d:h")) != -1)
        
        switch (opt_ch) {
	case 'h':
	  usage(basename(argv[0]));
	  exit(EXIT_SUCCESS);

        case 'n':
          // Number of interpolations.
          num = atoi(optarg);
          break;

        case 's':
          // Grid_Size
          grid_size = atoi(optarg);
          break;

        case 'i':
          // Interpolation.
          
          if ( strcasecmp(optarg, "near") == 0 ) {
            bs_type = near_n;
          } else if ( strcasecmp(optarg, "linear") == 0 ) { 
            bs_type = linear;
          } else if ( strcasecmp(optarg, "quadratic") == 0 ) { 
            bs_type = quadratic;
          } else if ( strcasecmp(optarg, "cubic") == 0 ) { 
            bs_type = cubic;
          } else if ( strcasecmp(optarg, "quartic") == 0 ) { 
            bs_type = quartic;
          } else if ( strcasecmp(optarg, "quintic") == 0 ) { 
            bs_type = quintic;
          } else if ( strcasecmp(optarg, "omoms") == 0 ) { 
            bs_type = oMoms;
          }
          
          break;
          
        case 'd':
          dimensions = atoi(optarg);
          break;

        case '?':
        default:
          usage(basename(argv[0]));
            exit(0);
            break;
        }
  
  argc -= optind;
    
  if ( argc > 0 ) {
    usage(basename(argv[0]));
    exit(EXIT_FAILURE);
  }

  
  
  // Sanity check on grid dimensions and number of interpolations to carry out.
  if ( num < 1 || num > MAX_NUM ) {
    num = DEF_NUM;
    cerr << "Number of interpolations out of range. Reset to: " << num << "\n";
  }



  if ( grid_size < MIN_SIZE ) {
    grid_size = DEF_SIZE;
    cerr << "Grid size too small. Reset to: " << grid_size << "\n";
  } else {
    
    int total_elements = grid_size;
    
    for (int d = 1 ; d < dimensions ; d++) {
      total_elements *= grid_size;
    }
    
    if ( total_elements > MAX_TOTAL_GRID_ELEMENTS ) {
      grid_size = DEF_SIZE;
      cerr << "Grid size too large for " << dimensions << " dimensions. Reset to: " 
           << grid_size << "\n";
    }
  }



  if ( dimensions < 1 || dimensions > MAX_DIMS ) {
    dimensions = DEF_DIMS;
    cerr << "Dimensions out of range. Reset to: " << dimensions << "\n";
  }

  
  int size[4] = {0, 0, 0, 0};
  for (int d = 0 ; d < dimensions ; d++) {
    size[d] = grid_size;
  }
  
  // Test arrays - Double precision.
  Array<1, float> data_1(size[0]);

  IndexRange2D range2(size[1], size[1]);
  Array<2, float> data_2(range2);
  
  IndexRange3D range3(size[2], size[2], size[2]);
  Array<3, float> data_3(range3);

  IndexRange4D range4(size[3], size[3], size[3], size[3]);
  Array<4, float> data_4(range4);
  
  
  // Fill test arrays.
  for (int i = 0 ; i < size[0] ; i++ ) {
    data_1[i] = i;
    
    for (int j = 0 ; j < size[1] ; j++ ) {
      data_2[i][j] = i * j;
        
      
      for (int k = 0 ; k < size[2] ; k++) {
        data_3[i][j][k] = i * j * k;
            
        for (int l = 0 ; l < size[3] ; l++) {
          data_4[i][j][k][l] = i * j * k * l;
        }
      }
    }
  }




  switch ( dimensions ) {
#ifndef MYLINEAR
  case 1:
    interpolate(num, size[0], bs_type, data_1);
    break;
    
  case 2:
    interpolate(num, size[1], bs_type, data_2);
    break;
#endif    
  case 3:
    interpolate(num, size[2], bs_type, data_3);
    break;
#ifndef MYLINEAR
  case 4:
    interpolate(num, size[3], bs_type, data_4);
    break;
#endif
  default:
    // Do nothing.
    break;

  }
  
  return(0);

} /* End of Main */

