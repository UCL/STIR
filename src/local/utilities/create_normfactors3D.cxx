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
  const int num_detectors = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_detectors_per_ring();
  const int num_crystals_per_block = 8;
  const int num_blocks = num_detectors/num_crystals_per_block;

  const int num_rings = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings();
  
  FanProjData fan_data;
  DetectorEfficiencies efficiencies(IndexRange2D(num_rings, num_detectors));

  const int iter_num=1;
  const int eff_iter_num = 0;

    {


	  // efficiencies
	  {
	    for (int ra = 0; ra < num_rings; ++ra)
              for (int a = 0; a < num_detectors; ++a)
	        efficiencies[ra][a] = exp(noise*((2.F*rand())/RAND_MAX - 1));
	    {
	      char *out_filename = new char[out_filename_prefix.size() + 30];
	      sprintf(out_filename, "%s_%s_%d_%d.out", 
		      out_filename_prefix.c_str(), "eff", iter_num, eff_iter_num);
	      ofstream out(out_filename);
	      out << efficiencies;
	      delete out_filename;
	    }
	}
    }

  return EXIT_SUCCESS;
}
