//
// $Id$
//


//MJ 9/11/98 Introduced math mode


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

#include "bbfilters.h"

#define ZERO_TOL 0.000001


PETImageOfVolume ask_interfile_image(char *input_query);
PETImageOfVolume get_interfile_image(char *filename);



main(int argc, char *argv[]){



  int scanner_num = 0;
  PETScannerInfo scanner;
  char scanner_char[200];


  scanner_num = ask_num("Enter scanner number (0: RPT, 1: 953, 2: 966, 3: GE) ? ", 0,3,0);
  switch( scanner_num )
    {
    case 0:
      scanner = (PETScannerInfo::RPT);
      sprintf(scanner_char,"%s","RPT");
      break;
    case 1:
      scanner = (PETScannerInfo::E953);
      sprintf(scanner_char,"%s","953");   
      break;
    case 2:
      scanner = (PETScannerInfo::E966);
      sprintf(scanner_char,"%s","966"); 
      break;
    case 3:
      scanner = (PETScannerInfo::Advance);
      sprintf(scanner_char,"%s","Advance");   
      break;
    default:
      PETerror("Wrong scanner number\n"); cerr<<endl;Abort();
    }

 

  Point3D origin(0,0,0);
  Point3D voxel_size(scanner.bin_size,
                     scanner.bin_size,
                     scanner.ring_spacing/2); 
  /*
    Point3D voxel_size(scanner.FOV_radius*2/scanner.num_bins,
    scanner.FOV_radius*2/scanner.num_bins,
    scanner.FOV_axial/(2*scanner.num_rings - 1));
  */
  // Create the image
  // two  manipulations are needed now: 
  // -it needs to have an odd number of elements
  // - it needs to be larger than expected because of some overflow in the projectors

  
  int max_bin = (-scanner.num_bins/2) + scanner.num_bins-1;
  if (scanner.num_bins % 2 == 0)
    max_bin++;
   



  PETImageOfVolume 
    input_image(Tensor3D<float>(
				0, 2*scanner.num_rings-2,
				(-scanner.num_bins/2), max_bin,
				(-scanner.num_bins/2), max_bin),
		origin,
		voxel_size);


 if (argc>1) input_image = get_interfile_image(argv[1]);

 else if(ask("Filter Dirac function?",true))
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

 input_image[zm][ym][xm]=1.0F;
   }

 else input_image=ask_interfile_image("Image to filter: ");

double fwhmx_dir=ask_num(" Full-width half-maximum (x-dir) ?", 0.0,20.0,0.5);
double fwhmz_dir=ask_num(" Full-width half-maximum (z-dir) ?", 0.0,20.0,0.5);
float Nx_dir=ask_num(" Metz power (x-dir) ?", 0.0,5.0,0.0);
float Nz_dir=ask_num(" Metz power (z-dir) ?", 0.0,5.0,0.0);	

 ImageFilter filter;

 filter.build(2,scanner,fwhmx_dir,fwhmz_dir,Nx_dir,Nz_dir);
 filter.apply(input_image);

 char outfile[max_filename_length];

 ask_filename_with_extension(
				outfile,
				"Output to which file: ", "");



  write_basic_interfile(outfile, input_image);



}

  /***************** Miscellaneous Functions  *******/





PETImageOfVolume ask_interfile_image(char *input_query){


    char filename[max_filename_length];
    ask_filename_with_extension(filename, 
				input_query,
				"");

    ifstream image_stream(filename);
    if (!image_stream)
    { 
      PETerror("Couldn't open file %s", filename);
      cerr<<endl;
      exit(1);
    }

return read_interfile_image(image_stream);

}

//MJ 12/9/98 added for command line input capability

PETImageOfVolume get_interfile_image(char *filename){


    ifstream image_stream(filename);
    if (!image_stream)
    { 
      PETerror("Couldn't open file %s", filename);
      cerr<<endl;
      exit(1);
    }

return read_interfile_image(image_stream);

}


