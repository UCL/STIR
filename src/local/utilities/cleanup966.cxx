//
// $Id$
//
/*!
  \file 
  \ingroup utilities
 
  \brief This programme attempts to get rid of the central artefact
  caused by normalisation+scatter problems on the 966. 

  Multiplies the central pixel in the volume by 0.80. Multiplies the
  surrounding pixels which share a common edge (as opposed to a corner)
  by 0.9. These factors empirically derived from one long scan of the
  uniform cylinder CylC.

  This program is a rewrite in STIR of Dale Bailey's CAPP code. 
  KT takes no credit/blame for it.

  \bug Just as Dale's version, this does not handle shift for non-zero x and y offsets. 
   As the CTI offset is currently not passed on to Interfile images,
   this is a dangerous thing, as I cannot properply check for it.

  \author Kris Thielemans
  \author Dale Bailey
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/interfile.h"
#include "stir/Succeeded.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif

USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
  if(argc!=3) {
    cerr<<"Usage: " << argv[0] << " <output image filename> <input image filename> \n";
    exit(EXIT_FAILURE);
  }
  
  // get parameters from command line

  char const * const output_filename = argv[1];
  char const * const input_filename = argv[2];
  
  // read image 

  shared_ptr<DiscretisedDensity<3,float> >  density_ptr = 
    DiscretisedDensity<3,float>::read_from_file(input_filename);

  if (density_ptr->get_origin().x()!=0 ||
      density_ptr->get_origin().z()!=0)
    {
      warning("%s: cannot handle non-zero offset in x or y in file %s\n",
	      argv[0], input_filename);
      return EXIT_FAILURE;
    }
  // TODO insert try() catch() to check if density type is wrong
  VoxelsOnCartesianGrid<float>& volume = 
    dynamic_cast< VoxelsOnCartesianGrid<float>&>(*density_ptr);
  const float fiddle1 = 0.80F;
  const float fiddle2 = 0.90F;

  /* original IDL code

  xdim = shdr.x_dimension
  ydim = shdr.y_dimension
  xc = xdim/2
  yc = ydim/2
  ; Fix pixels in 3 planes only
  volume(xc,yc,*)         = volume(xc,yc,*)*fiddle1
  volume(xc-1,yc-1,*)     = volume(xc-1,yc-1,*)*fiddle2
  volume(xc-1,yc,*)       = volume(xc-1,yc,*)*fiddle2
  volume(xc+1,yc-1,*)     = volume(xc+1,yc-1,*)*fiddle2
  volume(xc+1,yc,*)       = volume(xc+1,yc,*)*fiddle2
  volume(xc+1,yc+1,*)     = volume(xc+1,yc+1,*)*fiddle2
  volume(xc,yc-1,*)       = volume(xc,yc-1,*)*fiddle2
  volume(xc,yc+1,*)       = volume(xc,yc+1,*)*fiddle2
  */

  for (int z=volume.get_min_z(); z<= volume.get_max_z(); ++z)
    {
      volume[z][0][0] *= fiddle1;
      volume[z][-1][-1] *= fiddle2;
      volume[z][ 0][-1] *= fiddle2;
      //volume[z][+1][-1] *= fiddle2; // missing above! TODO
      volume[z][-1][+1] *= fiddle2;
      volume[z][ 0][+1] *= fiddle2;
      volume[z][+1][+1] *= fiddle2;
      volume[z][-1][ 0] *= fiddle2;
      volume[z][+1][ 0] *= fiddle2;
    }
  // write image
  Succeeded res = write_basic_interfile(output_filename, *density_ptr);

  return res==Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;

}




