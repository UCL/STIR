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

#include "stir/Scanner.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/stream.h"
#include "stir/display.h"
#include "stir/CPUTimer.h"
#include "stir/utilities.h"
#include <iostream>
#include <fstream>
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ofstream;
using std::fstream;
using std::string;
#endif
#include "stir/ProjDataFromStream.h"
#include "stir/interfile.h"

START_NAMESPACE_STIR


//************** 3D
void make_fan_sum_data(Array<2,float>& data_fan_sums, const FanProjData& fan_data)
{
  for (int ra = fan_data.get_min_ra(); ra <= fan_data.get_max_ra(); ++ra)
    for (int a = fan_data.get_min_a(); a <= fan_data.get_max_a(); ++a)
      data_fan_sums[ra][a] = fan_data.sum(ra,a);
}


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
  if (argc!=4)
    {
      cerr << "Usage: " << argv[0] 
	   << " out_filename_prefix measured_data  num_eff_iterations\n";
      return EXIT_FAILURE;
    }
  const bool do_display = ask("Display",false);
  const bool do_KL = ask("Compute KL distances?",false);
  const int num_eff_iterations = atoi(argv[3]);
  shared_ptr<ProjData> measured_data = ProjData::read_from_file(argv[2]);
  const string out_filename_prefix = argv[1];
  const int num_rings = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings();
  const int num_detectors_per_ring = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_detectors_per_ring();

  const ProjDataInfoCylindricalNoArcCorr * const proj_data_info_ptr = 
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr * const>(measured_data->get_proj_data_info_ptr());
  if (proj_data_info_ptr == 0)
  {
    cerr << "Can only process not arc-corrected data\n";
    return EXIT_FAILURE;
  }
  if (proj_data_info_ptr->get_view_mashing_factor()>1)
  {
    cerr << "Can only process data without mashing of views\n";
    return EXIT_FAILURE;
  }
  if (proj_data_info_ptr->get_max_ring_difference(0)>0)
  {
    cerr << "Can only process data without axial compression (i.e. span=1)\n";
    return EXIT_FAILURE;
  }


  const int max_ring_diff = proj_data_info_ptr->get_max_ring_difference(measured_data->get_max_segment_num());;
  const int half_fan_size = 
    min(proj_data_info_ptr->get_max_tangential_pos_num(),
        -proj_data_info_ptr->get_min_tangential_pos_num());
  CPUTimer timer;
  timer.start();


  Array<2,float> data_fan_sums(IndexRange2D(num_rings, num_detectors_per_ring));
  DetectorEfficiencies efficiencies(IndexRange2D(num_rings, num_detectors_per_ring));
  {

    // next could be local if KL is not computed below
    FanProjData measured_fan_data;
    float threshold_for_KL;    
    // compute factors dependent on the data
    {
      make_fan_data(measured_fan_data, *measured_data);
      threshold_for_KL = measured_fan_data.find_max()/100000.F;
      //display(measured_fan_data, "measured data");
      
      make_fan_sum_data(data_fan_sums, measured_fan_data);
    }

    // next only necessary for KL
    FanProjData fan_data;
    if (do_KL)
      {
	fan_data = measured_fan_data;
	fan_data.fill(1);
      }
    
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
          iterate_efficiencies(efficiencies, data_fan_sums, max_ring_diff, half_fan_size);
          {
            char *out_filename = new char[out_filename_prefix.size() + 30];
            sprintf(out_filename, "%s_%s_%d_%d.out", 
		    out_filename_prefix.c_str(), "eff", iter_num, eff_iter_num);
            ofstream out(out_filename);
            out << efficiencies;
            delete out_filename;
          }
          if (do_KL)
          {
	    fan_data.fill(1);
            apply_efficiencies(fan_data, efficiencies);
            //cerr << "model*norm min " << fan_data.find_min() << " ,max " << fan_data.find_max() << endl; 
            if (do_display)
              display( fan_data, "model_times_norm");
            cerr << "KL " << KL(measured_fan_data, fan_data, threshold_for_KL) << endl;
          }
          if (do_display)		 
          {
            display(efficiencies, "efficiencies");
          }
          
        }
      } // end efficiencies
    
    }
  }    
  timer.stop();
  cerr << "CPU time " << timer.value() << " secs" << endl;
  return EXIT_SUCCESS;
}
