//
// $Id$
//



#include "pet_common.h" 
#include <iostream> 
#include <fstream>

#include <numeric>
#include "imagedata.h"


//MJ 2/11/98 include Tensor functions
#include "TensorFunction.h"


#include "display.h"
// KT 13/10/98 include interfile 
#include "interfile.h"
// KT 23/10/98 use ask_* version all over the place
#include "utilities.h"
//MJ 17/11/98 include new functions
#include "recon_array_functions.h"

#include "ImageFilter.h"

#define ZERO_TOL 0.000001


PETImageOfVolume ask_interfile_image(char *input_query);


main(int argc, char *argv[]){


  if (argc<2) error("Usage: postfilter <image file name> \n"); 

  PETImageOfVolume input_image=read_interfile_image(argv[1]);

  //TODO Only require header file?

  if(ask("Filter Dirac function?",true))
    {

   
      int zs,ys,xs, ze,ye,xe, zm,ym,xm;

      zs=input_image.get_min_z();
      ys=input_image.get_min_y();
      xs=input_image.get_min_x(); 
      
      ze=input_image.get_max_z();  
      ye=input_image.get_max_y(); 
      xe=input_image.get_max_x();
      
      zm=(zs+ze)/2;
      ym=(ys+ye)/2;
      xm=(xs+xe)/2;

      input_image.fill(0.0);
      input_image[zm][ym][xm]=1.0F;

    }

 

  double fwhmx_dir=ask_num(" Full-width half-maximum (x-dir) ?", 0.0,20.0,0.5);
  double fwhmz_dir=ask_num(" Full-width half-maximum (z-dir) ?", 0.0,20.0,0.5);
  float Nx_dir=ask_num(" Metz power (x-dir) ?", 0.0,5.0,0.0);   
  float Nz_dir=ask_num(" Metz power (z-dir) ?", 0.0,5.0,0.0);	

  ImageFilter filter;

  filter.build(input_image,fwhmx_dir,fwhmz_dir,Nx_dir,Nz_dir);
  filter.apply(input_image);
 
  char outfile[max_filename_length];

  ask_filename_with_extension(outfile,
			      "Output to which file: ", "");



  write_basic_interfile(outfile, input_image);

  return EXIT_SUCCESS;

}

  /***************** Miscellaneous Functions  *******/





PETImageOfVolume ask_interfile_image(char *input_query){


    char filename[max_filename_length];
    ask_filename_with_extension(filename, 
				input_query,
				"");

    return read_interfile_image(filename);

}

