//
// $Id$
//

/*!
  \file 
  \ingroup utilities
 
  \brief  This programme performs filtering on image data

  \author Matthew Jacobson
  \author (with help from Kris Thielemans)
  \author PARAPET project

  \date $Date$  
  \version $Revision$

  \warning It only supports VoxelsOnCartesianGrid type of images.
*/


#include "interfile.h"
#include "utilities.h"
#include "DiscretisedDensity.h"
#include "VoxelsOnCartesianGrid.h"
#include "ImageFilter.h"

#include <iostream> 
#include <fstream>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_TOMO

VoxelsOnCartesianGrid<float>* ask_interfile_image(char *input_query);




  /***************** Miscellaneous Functions  *******/



VoxelsOnCartesianGrid<float>* ask_interfile_image(char *input_query){


    char filename[max_filename_length];
    ask_filename_with_extension(filename, 
				input_query,
				"");

    return read_interfile_image(filename);

}

END_NAMESPACE_TOMO

USING_NAMESPACE_TOMO

int
main(int argc, char *argv[])
{

 
  VoxelsOnCartesianGrid<float> input_image;
  if (argc>1)
  {
     input_image = 
	* dynamic_cast<VoxelsOnCartesianGrid<float> *>(
         DiscretisedDensity<3,float>::read_from_file(argv[1]).get());
  }
  else
  {

    cerr<<endl<<"Usage: postfilter <header file name> (*.hv)"<<endl<<endl;
    input_image= *ask_interfile_image("Image to process?");
  }
   
  //TODO Only require header file?



  // MJ 10/07/2000 Made thresholding optional

  bool applying_threshold=false;


  if(ask("Output filter PSF (instead of filtered image)?",false))
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
      (input_image)[zm][ym][xm]=1.0F;

    }

  else applying_threshold=ask("Truncate mininum to small (relative to image max) positive value?",false);
 

  double fwhmx_dir=ask_num(" Full-width half-maximum (x,y-dir) (in mm)?", 0.0,20.0,0.5);
  double fwhmz_dir=ask_num(" Full-width half-maximum (z-dir) (in mm)?", 0.0,20.0,0.5);
  float Nx_dir=ask_num(" Metz power (x-dir) ?", 0.0,5.0,0.0);   
  float Nz_dir=ask_num(" Metz power (z-dir) ?", 0.0,5.0,0.0);	

  ImageFilter filter;

  filter.build(input_image,fwhmx_dir,fwhmz_dir,Nx_dir,Nz_dir);
  filter.apply(input_image, applying_threshold);

  char outfile[max_filename_length];
  ask_filename_with_extension(outfile,
                               "Output to which file: ", "");



  write_basic_interfile(outfile,input_image);

  return EXIT_SUCCESS;

}

