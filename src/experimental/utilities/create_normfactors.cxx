//
//
/*!

  \file
  \ingroup utilities

  \brief Create sample normalisation factors for the ML approach to normalisation

  \author Kris Thielemans

*/
/*
    Copyright (C) 2001- 2008, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/ML_norm.h"

#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Scanner.h"
#include "stir/Bin.h"
#include "stir/stream.h"
#include "stir/Sinogram.h"
#include "stir/IndexRange2D.h"
#include "stir/display.h"
#include <iostream>
#include <fstream>
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ofstream;
using std::string;
#endif

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc!=4)
    {
      cerr << "Usage: " << argv[0] 
	   << " out_filename_prefix measured_data noise\n"
	   << "\t noise is a float factor: 0: effs all 1 etc.\n";
      return EXIT_FAILURE;
    }

  const float noise = static_cast<float>(atof(argv[3]));
  shared_ptr<ProjData> measured_data = ProjData::read_from_file(argv[2]);
  const string out_filename_prefix = argv[1];
  const int num_detectors = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_detectors_per_ring();
  const int num_crystals_per_block = 8;
  const int num_blocks = num_detectors/num_crystals_per_block;

  const int segment_num = 0;
  DetPairData det_pair_data;
  Array<1,float> efficiencies(num_detectors);
  assert(num_crystals_per_block%2 == 0);
  GeoData norm_geo_data(IndexRange2D(num_crystals_per_block/2, num_detectors));
  BlockData norm_block_data(IndexRange2D(num_blocks, num_blocks));

  const int iter_num=0;
  const int eff_iter_num = 0;

  for (int ax_pos_num = measured_data->get_min_axial_pos_num(segment_num);
       ax_pos_num <= measured_data->get_max_axial_pos_num(segment_num);
       ++ax_pos_num)
    {


	  // efficiencies
	  {
	    for (int b = 0; b < num_detectors; ++b)      
	      efficiencies[b] = exp(noise*((2.F*rand())/RAND_MAX - 1));
	    {
	      char *out_filename = new char[out_filename_prefix.size() + 30];
	      sprintf(out_filename, "%s_%s_%d_%d_%d.out", 
		      out_filename_prefix.c_str(), "eff", ax_pos_num, iter_num, eff_iter_num);
	      ofstream out(out_filename);
	      out << efficiencies;
	      delete[] out_filename;
	    }
	  // geo norm
	  {
	    // insert known geo factors
	    for (int a = 0; a < num_crystals_per_block/2; ++a)
	      for (int b = 0; b < num_detectors; ++b)      
		{
		  norm_geo_data[a][b] =(a+1)*cos(exp(-ax_pos_num)*(b-num_detectors/2)*_PI/num_detectors)+.1;
		}
            // it's for direct sinograms, so apply transpose symmetry
            {
              GeoData tmp = norm_geo_data;
              for (int a = 0; a < num_crystals_per_block/2; ++a)
                for (int b = 0; b < num_detectors; ++b)      
		{
                  int transposeda=b;
                  int transposedb=a;
                  int newa = transposeda % num_crystals_per_block;
                  int newb = transposedb - (transposeda - newa); 
                  if (newa > num_crystals_per_block - 1 - newa)
                  { 
                    newa = num_crystals_per_block - 1 - newa; 
                    newb = - newb + num_crystals_per_block - 1;
                  }
		  norm_geo_data[a][b] =
                    (tmp[a][b] + 
                     tmp[newa][(2*num_detectors + newb)%num_detectors])/2;
		}
            }
            {
	      char *out_filename = new char[out_filename_prefix.size() + 30];
	      sprintf(out_filename, "%s_%s_%d_%d.out", 
		      out_filename_prefix.c_str(), "geo", ax_pos_num, iter_num);
	      ofstream out(out_filename);
	      out << norm_geo_data;
	      delete[] out_filename;
	    }
	  }
          // block norm
	  {
            // it's for direct sinograms, so apply transpose symmetry
	    for (int a = 0; a < num_blocks; ++a)
	      for (int b = 0; b <=a; ++b)      
		{
                  
		  norm_block_data[a][b] = 
                    norm_block_data[b][a] =
                    exp(noise*((1.F*rand())/RAND_MAX - 0.5F));
		}
              
	    {
	      char *out_filename = new char[out_filename_prefix.size() + 30];
	      sprintf(out_filename, "%s_%s_%d_%d.out", 
		      out_filename_prefix.c_str(), "block", ax_pos_num, iter_num);
	      ofstream out(out_filename);
	      out << norm_block_data;
	      delete[] out_filename;
	    }
	  }
	}
    }

  return EXIT_SUCCESS;
}
