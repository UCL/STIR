//
// $Id:
//

/*!
  \file
  \ingroup 

  \brief Calculte variance image:
   This program calculate variance image accoording to the following equation :
   var(x) = E(x-m)^2 => var(x)= E(x^2)-(E(x))^2.   ( eq.1)
   However, when the sample mean is calculated ( eq.1) becomes
   var (x) =1/(n-1) ( sum(x^2) - n (1/sum (x /(n)))^2

   given the square of the cum sum and the mean image calculate a variance image
    
  \author Sanida Mustafovic
  
*/
/*
    Copyright (C) 2002- $Year:$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/interfile.h"
#include "stir/utilities.h"
#include "stir/ProjData.h"
#include "stir/ProjDataFromStream.h"
#include "stir/RelatedViewgrams.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Viewgram.h"
#include "stir/ArrayFunction.h"
#include "stir/recon_array_functions.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"

#include "stir/SegmentByView.h"

#include <fstream>
#include <list>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::fstream;
using std::iostream;
using std::endl;
using std::list;
using std::find;
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_STIR


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int 
main (int argc, char * argv[])
{
  
  VoxelsOnCartesianGrid<float> * output_variance_image;
  
  if (argc!=5)
  {
    cerr<<"\nUsage: calculate variance image : <output variance image > <input square sum> <mean image> number of trials \n"<<endl;
  }
  
   ofstream output_file;
   string out_file = "variances_for_pixels_of_interset";
   output_file.open(out_file.c_str(),ios::out);
   output_file << " z    y     x    " <<                " variance" <<"          " << "std" << endl;

   shared_ptr< DiscretisedDensity<3,float> >  square_sum = 
   DiscretisedDensity<3,float>::read_from_file(argv[2]);
   shared_ptr< DiscretisedDensity<3,float> >  mean_image = 
   DiscretisedDensity<3,float>::read_from_file(argv[3]);

   VoxelsOnCartesianGrid<float> *  square_sum_vox= 
    dynamic_cast<VoxelsOnCartesianGrid<float> *> (square_sum.get());
   VoxelsOnCartesianGrid<float> *  mean_image_vox = 
    dynamic_cast<VoxelsOnCartesianGrid<float> *> (mean_image.get());
   int num_trails = atoi(argv[4]);

   output_variance_image = square_sum_vox->get_empty_voxels_on_cartesian_grid();

   *mean_image_vox *=*mean_image_vox; 
   *mean_image_vox *=num_trails;
   *square_sum_vox -= *mean_image_vox;
   *output_variance_image = *square_sum_vox ;
   *output_variance_image /=num_trails-1;
   output_file << " 0   1   5    " << (*output_variance_image)[0][1][5]<< "    " << sqrt((*output_variance_image)[0][1][5]) << endl;
   output_file << " 1   1   5    " << (*output_variance_image)[1][1][5] << "    " << sqrt((*output_variance_image)[1][1][5]) << endl;
   output_file << " 0  14   7    " << (*output_variance_image)[0][14][7] << "    " << sqrt((*output_variance_image)[0][14][7]) << endl;
   output_file << " 0 -13   1    " << (*output_variance_image)[0][-13][1] << "    " << sqrt((*output_variance_image)[0][-13][1]) << endl;





   
   //*output_variance_image = 1/(num_trails -1)((*square_sum_vox) - *mean_image_vox);

   write_basic_interfile(argv[1], *output_variance_image);


   return EXIT_SUCCESS;
}


