//
// $Id:
//

/*!
  \file
  \ingroup 

  \brief Calculate variance on a variance image - gives the error bar on 
   the variance estimate:
   
   formula: 1/(N-1)(N-2) ( s[4] -4* mean*s[3] - (N^2 -3)/N*unbiasedvariance^2
    + (6N-6) unbiasedvariance*mean^2 + 3N *mean ^4)

  where N is a number of trials and
  mean = 1/N * sum_i x(i)
  unbiasedvariance = 1/(N-1) ( sum(x^2) - N (1/sum (x /(N)))^2
    
  \ author Sanida Mustafovic 
  \ author Kris Thielemas
  
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
  
  VoxelsOnCartesianGrid<float> * output_variance_on_variance_image;
  VoxelsOnCartesianGrid<float> * mean_to_power_four;
  VoxelsOnCartesianGrid<float> * mean_to_power_two;
  
  if (argc!=5)
  {
    cerr<<"\nUsage: calculate variance image : <output variance on variance estimate image > <input cum_sum_power_three> <input cum_sum_power_four> <mean image>  <unbiasedvariance> number of trials \n"<<endl;
    return EXIT_FAILURE;
  }
  
   ofstream output_file;
   string out_file = "variances_on_variance_estimate_for_pixels_of_interset";
   output_file.open(out_file.c_str(),ios::out);
   output_file << " z    y     x    " <<                " variance" <<"          " << "std" << endl;

   shared_ptr< DiscretisedDensity<3,float> > cum_sum_power_three = 
   DiscretisedDensity<3,float>::read_from_file(argv[2]);
   shared_ptr< DiscretisedDensity<3,float> >  cum_sum_power_four = 
   DiscretisedDensity<3,float>::read_from_file(argv[3]);
    shared_ptr< DiscretisedDensity<3,float> >  mean_image = 
   DiscretisedDensity<3,float>::read_from_file(argv[4]);
   shared_ptr< DiscretisedDensity<3,float> >  unbiasedvariance = 
   DiscretisedDensity<3,float>::read_from_file(argv[5]);

   VoxelsOnCartesianGrid<float> *  cum_sum_power_three_vox= 
    dynamic_cast<VoxelsOnCartesianGrid<float> *> (cum_sum_power_three.get());
   VoxelsOnCartesianGrid<float> *  cum_sum_power_four_vox = 
    dynamic_cast<VoxelsOnCartesianGrid<float> *> (cum_sum_power_four.get());
    VoxelsOnCartesianGrid<float> *  mean_image_vox= 
    dynamic_cast<VoxelsOnCartesianGrid<float> *> (mean_image.get());
   VoxelsOnCartesianGrid<float> *  unbiasedvariance_vox = 
    dynamic_cast<VoxelsOnCartesianGrid<float> *> (unbiasedvariance.get());
   int num_trails = atoi(argv[6]);

   output_variance_on_variance_image = cum_sum_power_three_vox->get_empty_voxels_on_cartesian_grid();
   mean_to_power_four = cum_sum_power_three_vox->get_empty_voxels_on_cartesian_grid();
   mean_to_power_two =cum_sum_power_three_vox->get_empty_voxels_on_cartesian_grid();
     

   *mean_to_power_two = *mean_image_vox;
   *mean_to_power_two *= *mean_image_vox;

   *mean_to_power_four = *mean_to_power_two;
   *mean_to_power_four *= *mean_to_power_two;

   *mean_to_power_four *= 3*num_trails;

   *mean_to_power_two *= *unbiasedvariance_vox;
   *mean_to_power_two *= (6*num_trails -6);

   *unbiasedvariance_vox *= *unbiasedvariance_vox;
   *unbiasedvariance_vox *=(square(num_trails)-3)/num_trails;


   *mean_image_vox *=*cum_sum_power_three*4;

  // now combine all together 
   *output_variance_on_variance_image  = *cum_sum_power_four_vox;
   *output_variance_on_variance_image -= *mean_image_vox;
   *output_variance_on_variance_image -= *unbiasedvariance_vox;
   *output_variance_on_variance_image += *mean_to_power_two;
   *output_variance_on_variance_image += *mean_to_power_four;



   *output_variance_on_variance_image /= 1/(num_trails-3)*(num_trails-2);

   output_file << " 0   1   5    " << (*output_variance_on_variance_image)[0][1][5]<< "    " << sqrt((*output_variance_on_variance_image)[0][1][5]) << endl;
   output_file << " 1   1   5    " << (*output_variance_on_variance_image)[1][1][5] << "    " << sqrt((*output_variance_on_variance_image)[1][1][5]) << endl;
   output_file << " 0  14   7    " << (*output_variance_on_variance_image)[0][14][7] << "    " << sqrt((*output_variance_on_variance_image)[0][14][7]) << endl;
   output_file << " 0 -13   1    " << (*output_variance_on_variance_image)[0][-13][1] << "    " << sqrt((*output_variance_on_variance_image)[0][-13][1]) << endl;





   
   //*output_variance_image = 1/(num_trails -1)((*square_sum_vox) - *mean_image_vox);

   write_basic_interfile(argv[1], *output_variance_on_variance_image);


   return EXIT_SUCCESS;
}


