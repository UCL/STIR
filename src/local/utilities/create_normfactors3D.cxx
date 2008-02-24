//
// $Id$
//
/*!

  \file
  \ingroup utilities

  \brief Create sample normalisation factors for the ML approach to normalisation

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2001- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/ML_norm.h"

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
  const int num_rings = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings();
  const int num_detectors_per_ring = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_detectors_per_ring();
  const int num_tangential_crystals_per_block = 8;
  const int num_tangential_blocks = num_detectors_per_ring/num_tangential_crystals_per_block;
  const int num_axial_crystals_per_block = num_rings/2;
  warning("TODO num_axial_crystals_per_block == num_rings/2\n");
  const int num_axial_blocks = num_rings/num_axial_crystals_per_block;

  BlockData3D norm_block_data(num_axial_blocks, num_tangential_blocks,
                              num_axial_blocks-1, num_tangential_blocks-1);
  DetectorEfficiencies efficiencies(IndexRange2D(num_rings, num_detectors_per_ring));

  const int iter_num=1;
  const int eff_iter_num = 0;

    {


	  // efficiencies
	  {
	    for (int ra = 0; ra < num_rings; ++ra)
              for (int a = 0; a < num_detectors_per_ring; ++a)
	        efficiencies[ra][a] = 
		  static_cast<float>((2+sin(2*_PI*a/num_detectors_per_ring))*
				     exp(noise*((2.F*rand())/RAND_MAX - 1)));
	    {
	      char *out_filename = new char[out_filename_prefix.size() + 30];
	      sprintf(out_filename, "%s_%s_%d_%d.out", 
		      out_filename_prefix.c_str(), "eff", iter_num, eff_iter_num);
	      ofstream out(out_filename);
	      out << efficiencies;
	      delete[] out_filename;
	    }
          } // end efficiencies
          // block norm
	  {
             for (int ra = norm_block_data.get_min_ra(); ra <= norm_block_data.get_max_ra(); ++ra)
              for (int a = norm_block_data.get_min_a(); a <= norm_block_data.get_max_a(); ++a)
                // loop rb from ra to avoid double counting
                for (int rb = max(ra,norm_block_data.get_min_rb(ra)); rb <= norm_block_data.get_max_rb(ra); ++rb)
                  for (int b = norm_block_data.get_min_b(a); b <= norm_block_data.get_max_b(a); ++b)      
                  {                  
		    norm_block_data(ra,a,rb,b) =
                      exp(noise*((1.F*rand())/RAND_MAX - 0.5F));
                    if (ra==rb) // it's for direct sinograms, so apply transpose symmetry
                      norm_block_data(ra,b,rb,a) = norm_block_data(ra,a,rb,b);
                  } 
              
	    {
	      char *out_filename = new char[out_filename_prefix.size() + 30];
	      sprintf(out_filename, "%s_%s_%d.out", 
		      out_filename_prefix.c_str(), "block",  iter_num);
	      ofstream out(out_filename);
	      out << norm_block_data;
	      delete[] out_filename;
	    }
	  
          } // end block

    }

  return EXIT_SUCCESS;
}
