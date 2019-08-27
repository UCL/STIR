/*
 Copyright (C) 2011 - 2013, King's College London
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
 \ingroup utilities

 \brief This program inverts x y or z axis. It is also able to remove nan from the image.
 \author Daniel Deidda
 */
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/IO/OutputFileFormat.h"
USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if(argc!=4) {
    std::cerr << "Usage: " << argv[0]
              << " <axis name>  <output filename> <input filename>\n"
              << " if <axis name>= nan the utility will substitute every nan value with zero  ";
    exit(EXIT_FAILURE);
  }
  char const * const axis_name = argv[1];
  char const * const output_filename_prefix = argv[2];
  char const * const  input_filename= argv[3];
  std::string name(axis_name);
 // const int num_planes = atoi(argv[3]);
std::cout<<" the axis to invert is "<< name<<std::endl;
  const std::auto_ptr<DiscretisedDensity<3,float> > image_aptr(DiscretisedDensity<3,float>::read_from_file(input_filename));
  const std::auto_ptr<DiscretisedDensity<3,float> > out_image_aptr(image_aptr->clone());

  DiscretisedDensity<3,float>& image = *image_aptr.get ();
  DiscretisedDensity<3,float>& output = *out_image_aptr.get ();

  const int min_z = image.get_min_index();
  const int max_z = image.get_max_index();


      for (int z=min_z; z<=max_z; z++)
        {
//std::cout<<" pos "<< z<< " "<<std::endl;
          const int min_y = image[z].get_min_index();
          const int max_y = image[z].get_max_index();



            for (int y=min_y;y<= max_y;y++)
              {

                const int min_x = image[z][y].get_min_index();
                const int max_x = image[z][y].get_max_index();



                  for (int x=min_x;x<= max_x;x++)
                  {
                     // std::cout<<" pos "<< z<< " "<<-y-1 <<" "<<y<<std::endl;
                    if (name=="x"){
                        output[z][y][x]=image[z][y][-x-1];
                    }
                  else if (name=="y"){

                        //std::cout<<" image "<< image[0][-172][171]<<std::endl;
                        //if (image[z][-y-1][x]==image[0][-172][171]){
                            //std::cout<<" image "<< image[z][-y-1][x]<<std::endl;
//                            std::cout<<" pos "<< max_y<< " "<<-y-1 <<" "<<y<<std::endl;
                           // image[z][-y-1][x]=0;
                            //std::cout<<" image "<< image[z][-y-1][x]<<std::endl;
                       // }
                            if(((max_y-min_y+1) % 2)==0)
                                output[z][y][x]=image[z][-y-1][x];
                            else
                                output[z][y][x]=image[z][-y][x];

                    }
                    else if (name=="z"){
                        //std::cout<<" pos "<< z<< " "<<max_z-z <<" "<<y<<std::endl;
                        output[z][y][x]=image[max_z-z][y][x];
                    }
                    else if (name=="nan"){
                    if(image[z][y][x]>=0 && image[z][y][x]<=1000000){
                        //if(image[z][y][x]!=1 ){
                        //output[z][y][x] = std::numeric_limits<double>::quiet_NaN();
                        //std::cout<<"nan"<<image[z][y][x]<<std::endl;
                      continue;
                    }
                    else{
                      output[z][y][x]=0;
                    }

                  }}
              }
          }
  //std::cout<<" image "<< image[0][-172][172]<<std::endl;
  const Succeeded res = OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
    write_to_file(output_filename_prefix, output);

  return res==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
