//
/*
 Copyright (C) 2019 National Physical Laboratory
 This file is part of STIR.
 
 This file is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.3 of the License, or
 (at your option) any later version.
 
 This file is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
 
 See STIR/LICENSE.txt for details
 */  
/*!
 \file
 \ingroup buildblock
 \brief Implementation of function stir::invert_axis
 \author Daniel Deidda
*/

#include "stir/spatial_transformation/InvertAxis.h"

START_NAMESPACE_STIR

void
InvertAxis::invert_axis(DiscretisedDensity<3,float>  & inverted_image,
                        const DiscretisedDensity<3,float>  & input_image,
                        const std::string &axis_name){
//change all the pointers
    const int min_z = input_image.get_min_index();
    const int max_z = input_image.get_max_index();


        for (int z=min_z; z<=max_z; z++){

            const int min_y = input_image[z].get_min_index();
            const int max_y = input_image[z].get_max_index();

              for (int y=min_y;y<= max_y;y++){

                  const int min_x = input_image[z][y].get_min_index();
                  const int max_x = input_image[z][y].get_max_index();

                    for (int x=min_x;x<= max_x;x++){

                      if (axis_name=="x"){
//                        checking whether the size is odd
                          if(((max_x-min_x+1) % 2)==0)
                              inverted_image[z][y][x]=input_image[z][y][-x-1];
                          else
                              inverted_image[z][y][x]=input_image[z][y][-x];
                          }

                      else if (axis_name=="y"){
                          if(((max_y-min_y+1) % 2)==0)
                              inverted_image[z][y][x]=input_image[z][-y-1][x];
                          else
                              inverted_image[z][y][x]=input_image[z][-y][x];
                          }

                      else if (axis_name=="z"){
                          inverted_image[z][y][x]=input_image[max_z-z][y][x];
                          }
                      }
                    }
              }
        }

int
InvertAxis::invert_axis_index(const int input_index,
                              const int size,
                              const std::string& axis_name){

    if (axis_name=="x" || axis_name=="y"){

//      checking whether the size is odd
        if((size % 2)==0)
            return -input_index -1;
        else
            return -input_index;
        }

    else if (axis_name=="z")
        return size -1 -input_index;

}
END_NAMESPACE_STIR

