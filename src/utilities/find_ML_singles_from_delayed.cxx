/*
    Copyright (C) 2002- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
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

  \brief Find singles (and hence randoms) from delayed events using an ML approach

  \author Kris Thielemans

*/

#include "stir/ML_norm.h"

#include "stir/stream.h"
#include "stir/display.h"
#include "stir/CPUTimer.h"
#include "stir/utilities.h"
#include <iostream>
#include <fstream>
#include <string>

START_NAMESPACE_STIR

static unsigned long compute_num_bins(const int num_rings, const int num_detectors_per_ring,
	             const int max_ring_diff, const int half_fan_size)
{
  unsigned long num = 0;
  for (int ra = 0; ra < num_rings; ++ra)
    for (int a =0; a < num_detectors_per_ring; ++a)
    {
      for (int rb = std::max(ra-max_ring_diff, 0); rb <= std::min(ra+max_ring_diff, num_rings-1); ++rb)
        for (int b = a+num_detectors_per_ring/2-half_fan_size; b <= a+num_detectors_per_ring/2+half_fan_size; ++b)
  	   ++num;
    }
  return num;
}


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{  
  if (!(argc==4 || (argc==7 && strcmp(argv[1],"-f")==0)))
    {
      std::cerr << "Usage: \n" 
                << '\t' << argv[0] << " -f out_filename_prefix measured_fan_sum_data  num_iterations max_ring_diff fan_size\n"
                << "or\n"
                << '\t' << argv[0] << " out_filename_prefix measured_projdata  num_iterations\n"
                << "If the -f option is used, the 2nd arg should be a file with fan_sums. "
                << "Otherwise, it has to be projection data.\n";

      return EXIT_FAILURE;
    }
  const int num_eff_iterations = atoi(argv[argc==4?3:4]);
  const std::string out_filename_prefix = argv[argc==4?1:2];

  const int do_display_interval = 
    ask_num("Display iterations which are a multiple of ",0,num_eff_iterations,0);
  const int do_KL_interval = 
    ask_num("Compute KL distance between fan-sums at iterations which are a multiple of ",0,num_eff_iterations,0);
  const int do_save_interval = 
    ask_num("Write output at iterations which are a multiple of ",0,num_eff_iterations,num_eff_iterations);


  
  int num_rings;
  int num_detectors_per_ring;
  int fan_size;
  int max_ring_diff;
  Array<2,float> data_fan_sums;

  if (argc==4)
    {
      shared_ptr<ProjData> measured_data = ProjData::read_from_file(argv[2]);
      get_fan_info(num_rings, num_detectors_per_ring, max_ring_diff, fan_size, 
		     *measured_data->get_proj_data_info_ptr());
      data_fan_sums.grow(IndexRange2D(num_rings, num_detectors_per_ring));
#if 0
      FanProjData measured_fan_data;
      make_fan_data(measured_fan_data, *measured_data);
      make_fan_sum_data(data_fan_sums, measured_fan_data);
#else
      make_fan_sum_data(data_fan_sums, *measured_data);
#endif
      // write fan sums to file
      {
        std::string fan_sum_name = "fansums_for_";
	fan_sum_name += argv[2];
	fan_sum_name.erase(fan_sum_name.begin() + fan_sum_name.rfind('.'), 
			   fan_sum_name.end());
	fan_sum_name += ".dat"; 
	std::ofstream out(fan_sum_name.c_str());
	if (!out)
	  {
	    warning("Error opening output file %s\n", fan_sum_name.c_str());
	    exit(EXIT_FAILURE);
	  }
	out << data_fan_sums;
	if (!out)
	  {
	    warning("Error writing data to output file %s\n", fan_sum_name.c_str());
	    exit(EXIT_FAILURE);
	  }
      }
    }
  else
    {
      max_ring_diff = atoi(argv[5]);
      fan_size = atoi(argv[6]);
      std::ifstream in(argv[3]);
      if (!in)
	{
	  warning("Error opening input file %s\n", argv[3]);
	  exit(EXIT_FAILURE);
	}
      in >> data_fan_sums;
      num_rings = data_fan_sums.get_length();
      if (num_rings==0)
	{
	  warning("input file %s should be a  2d list of numbers but I found "
		  "a list of length 0 (or no list at all).\n", argv[3]);
	  exit(EXIT_FAILURE);
	}
      assert(data_fan_sums.get_min_index()==0);
      num_detectors_per_ring = data_fan_sums[0].get_length();
      if (!data_fan_sums.is_regular())
	{
	  warning("input file %s should be a (rectangular) matrix of numbers\n", argv[3]);
	  exit(EXIT_FAILURE);
	}

      if (num_detectors_per_ring==0)
	{
	  warning("input file %s should be a 2d list of numbers but I found something else (zero number of columns?)\n", argv[3]);
	  exit(EXIT_FAILURE);
	}
      if (num_rings<max_ring_diff || num_detectors_per_ring<fan_size)
	{
	  warning("input file %s is a matrix with sizes %dx%d, but this is "
		  "too small compared to max_ring_diff (%d) and/or fan_size (%d)\n", 
		  argv[3],num_rings,num_detectors_per_ring,
		  max_ring_diff, fan_size);
	  exit(EXIT_FAILURE);
	}
    }
  const int half_fan_size = fan_size/2;


  CPUTimer timer;
  timer.start();
  
  DetectorEfficiencies efficiencies(IndexRange2D(num_rings, num_detectors_per_ring));
  {

    float threshold_for_KL = data_fan_sums.find_max()/100000.F;    
    const int iter_num = 1;
    {
      if (iter_num== 1)
      {
        efficiencies.fill(sqrt(data_fan_sums.sum()/
                               compute_num_bins(num_rings, num_detectors_per_ring, max_ring_diff, half_fan_size)));
      }
      // efficiencies
      {
        for (int eff_iter_num = 1; eff_iter_num<=num_eff_iterations; ++eff_iter_num)
        {
          std::cout << "Starting iteration " << eff_iter_num;
          iterate_efficiencies(efficiencies, data_fan_sums, max_ring_diff, half_fan_size);
          if (eff_iter_num==num_eff_iterations || (do_save_interval>0 && eff_iter_num%do_save_interval==0))
          {
            char *out_filename = new char[out_filename_prefix.size() + 30];
            sprintf(out_filename, "%s_%s_%d_%d.out", 
		    out_filename_prefix.c_str(), "eff", iter_num, eff_iter_num);
            std::ofstream out(out_filename);
	    if (!out)
	      {
		warning("Error opening output file %s\n", out_filename);
		exit(EXIT_FAILURE);
	      }
            out << efficiencies;
	    if (!out)
	      {
		warning("Error writing data to output file %s\n", out_filename);
		exit(EXIT_FAILURE);
	      }

            delete[] out_filename;
          }
          if (eff_iter_num==num_eff_iterations || (do_KL_interval>0 && eff_iter_num%do_KL_interval==0))
          {
	    Array<2,float> estimated_fan_sums(data_fan_sums.get_index_range());
	    make_fan_sum_data(estimated_fan_sums, efficiencies, max_ring_diff, half_fan_size);
	    std::cout << "\tKL " << KL(data_fan_sums, estimated_fan_sums, threshold_for_KL);

          }
          std::cout << std::endl;
          if (do_display_interval>0 && eff_iter_num%do_display_interval==0)		 
          {
            display(efficiencies, "efficiencies");
          }
          
        }
      } // end efficiencies
    
    }
  }    
  timer.stop();
  std::cout << "CPU time " << timer.value() << " secs" << std::endl;
  return EXIT_SUCCESS;
}
