//
// $Id$
//
/*!

  \file
  \ingroup utilities

  \brief Find singles (and hence randoms) from delayed events using an ML approach

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "local/stir/ML_norm.h"

#include "stir/stream.h"
#include "stir/display.h"
#include "stir/CPUTimer.h"
#include "stir/utilities.h"
#include <iostream>
#include <fstream>
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;
#endif
START_NAMESPACE_STIR


//************** 3D

// version without model
void iterate_efficiencies(DetectorEfficiencies& efficiencies,
			  const Array<2,float>& data_fan_sums,
			  const int max_ring_diff, const int half_fan_size)
{
  const int num_rings = data_fan_sums.get_length();
  const int num_detectors_per_ring = data_fan_sums[data_fan_sums.get_min_index()].get_length();
  for (int ra = data_fan_sums.get_min_index(); ra <= data_fan_sums.get_max_index(); ++ra)
    for (int a = data_fan_sums[ra].get_min_index(); a <= data_fan_sums[ra].get_max_index(); ++a)
    {
      if (data_fan_sums[ra][a] == 0)
	efficiencies[ra][a] = 0;
      else
	{
     	  float denominator = 0;
	  for (int rb = max(ra-max_ring_diff, 0); rb <= min(ra+max_ring_diff, num_rings-1); ++rb)
             for (int b = a+num_detectors_per_ring/2-half_fan_size; b <= a+num_detectors_per_ring/2+half_fan_size; ++b)
  	       denominator += efficiencies[rb][b%num_detectors_per_ring];
	  efficiencies[ra][a] = data_fan_sums[ra][a] / denominator;
	}
    }
}

unsigned long compute_num_bins(const int num_rings, const int num_detectors_per_ring,
	             const int max_ring_diff, const int half_fan_size)
{
  unsigned long num = 0;
  for (int ra = 0; ra < num_rings; ++ra)
    for (int a =0; a < num_detectors_per_ring; ++a)
    {
      for (int rb = max(ra-max_ring_diff, 0); rb <= min(ra+max_ring_diff, num_rings-1); ++rb)
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
      cerr << "Usage: \n" 
	   << '\t' << argv[0] << " -f out_filename_prefix measured_fan_sum_data  num_iterations max_ring_diff fan_size\n"
	   << "or\n"
	   << '\t' << argv[0] << " out_filename_prefix measured_projdata  num_iterations\n"
	   << "If the -f option is used, the 2nd arg should be a file with fan_sums. "
	   << "Otherwise, it has to be projection data.\n";

      return EXIT_FAILURE;
    }
  const int num_eff_iterations = atoi(argv[argc==4?3:4]);
  const string out_filename_prefix = argv[argc==4?1:2];

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
	string fan_sum_name = "fansums_for_";
	fan_sum_name += argv[2];
	fan_sum_name.erase(fan_sum_name.begin() + fan_sum_name.rfind('.'), 
			   fan_sum_name.end());
	fan_sum_name += ".dat"; 
	ofstream out(fan_sum_name.c_str());
	out << data_fan_sums;
      }
    }
  else
    {
      max_ring_diff = atoi(argv[5]);
      fan_size = atoi(argv[6]);
      ifstream in(argv[3]);
      in >> data_fan_sums;
      num_rings = data_fan_sums.get_length();
      assert(data_fan_sums.get_min_index()==0);
      num_detectors_per_ring = data_fan_sums[0].get_length();
      if (!data_fan_sums.is_regular())
	error("Error reading fan-sum file (not a square matrix)\n");

      if (num_rings==0 || num_detectors_per_ring==0)
	error("Error reading fan-sum file (0 size)\n");
      if (num_rings<max_ring_diff || num_detectors_per_ring<fan_size)
	error("Error reading fan-sum file (sizes too small compared to max_ring_diff and/or fan_size)\n");
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
          cout << "Starting iteration " << eff_iter_num << endl;
          iterate_efficiencies(efficiencies, data_fan_sums, max_ring_diff, half_fan_size);
          if (eff_iter_num==num_eff_iterations || (do_save_interval>0 && eff_iter_num%do_save_interval==0))
          {
            char *out_filename = new char[out_filename_prefix.size() + 30];
            sprintf(out_filename, "%s_%s_%d_%d.out", 
		    out_filename_prefix.c_str(), "eff", iter_num, eff_iter_num);
            ofstream out(out_filename);
            out << efficiencies;
            delete out_filename;
          }
          if (eff_iter_num==num_eff_iterations || (do_KL_interval>0 && eff_iter_num%do_KL_interval==0))
          {
	    Array<2,float> estimated_fan_sums(data_fan_sums.get_index_range());
	    make_fan_sum_data(estimated_fan_sums, efficiencies, max_ring_diff, half_fan_size);
	    cout << "\tKL " << KL(data_fan_sums, estimated_fan_sums, threshold_for_KL) << endl;

          }
          if (do_display_interval>0 && eff_iter_num%do_display_interval==0)		 
          {
            display(efficiencies, "efficiencies");
          }
          
        }
      } // end efficiencies
    
    }
  }    
  timer.stop();
  cout << "CPU time " << timer.value() << " secs" << endl;
  return EXIT_SUCCESS;
}
