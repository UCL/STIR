//
// $Id$
//
/*!

  \file
  \ingroup utilities

  \brief Find normalisation factors using an ML approach

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2001- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "local/stir/ML_norm.h"

#include "stir/Scanner.h"
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
  for (int ra = fan_data.get_min_index(); ra <= fan_data.get_max_index(); ++ra)
    for (int a = fan_data.get_min_a(); a <= fan_data.get_max_a(); ++a)
      data_fan_sums[ra][a] = fan_data.sum(ra,a);
}

void iterate_efficiencies(DetectorEfficiencies& efficiencies,
			  const Array<2,float>& data_fan_sums,
			  const FanProjData& model)
{
#if 0
  const int num_detectors = model.get_num_detectors();

  for (int ra = model.get_min_index(); ra <= model.get_max_index(); ++ra)
    for (int a = model.get_min_a(); a <= model.get_max_a(); ++a)
    {
      if (data_fan_sums[ra][a] == 0)
	efficiencies[ra][a] = 0;
      else
	{
     	  float denominator = 0;
           for (int rb = model.get_min_rb(ra); rb <= model.get_max_rb(ra); ++rb)
             for (int b = model.get_min_b(a); b <= model.get_max_b(a); ++b)
  	       denominator += efficiencies[rb][b%num_detectors]*model(ra,a,rb,b);
	  efficiencies[ra][a] = data_fan_sums[ra][a] / denominator;
	}
    }
#else
  FanProjData estimate = model;
  Array<2,float> estimate_fan_sums(data_fan_sums.get_index_range());
  for (int ra = model.get_min_index(); ra <= model.get_max_index(); ++ra)
    for (int a = model.get_min_a(); a <= model.get_max_a(); ++a)
      {
	estimate= model;
	apply_efficiencies(estimate, efficiencies);
	make_fan_sum_data(estimate_fan_sums, estimate);
	efficiencies[ra][a] *= data_fan_sums[ra][a];
	// TODO this has problems with div through 0
	efficiencies[ra][a] /= estimate_fan_sums[ra][a];
      }
#endif
}

float KL(const FanProjData& d1, const FanProjData& d2, const float threshold = 0)
{
  float sum=0;
  for (int ra = d1.get_min_index(); ra <= d1.get_max_index(); ++ra)
    for (int a = d1.get_min_a(); a <= d1.get_max_a(); ++a)
      for (int rb = d1.get_min_rb(ra); rb <= d1.get_max_rb(ra); ++rb)
        for (int b = d1.get_min_b(a); b <= d1.get_max_b(a); ++b)      
          sum += KL(d1(ra,a,rb,b), d2(ra,a,rb,b), threshold);
  return sum;
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{  
  //check_geo_data();
  if (argc!=6)
    {
      cerr << "Usage: " << argv[0] 
	   << " out_filename_prefix measured_data model num_iterations num_eff_iterations\n"
	   << " set num_iterations to 0 to do only efficiencies\n";
      return EXIT_FAILURE;
    }
  const bool do_display = ask("Display",false);
  const bool do_KL = ask("Compute KL distances?",false);
  const int num_eff_iterations = atoi(argv[5]);
  const int num_iterations = atoi(argv[4]);
  shared_ptr<ProjData> model_data = ProjData::read_from_file(argv[3]);
  shared_ptr<ProjData> measured_data = ProjData::read_from_file(argv[2]);
  const string out_filename_prefix = argv[1];
  const int num_rings = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings();
  const int num_detectors = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_detectors_per_ring();
  const int num_crystals_per_block = 8;
  const int num_blocks = num_detectors/num_crystals_per_block;

  CPUTimer timer;
  timer.start();

  FanProjData model_fan_data;
  Array<2,float> data_fan_sums(IndexRange2D(num_rings, num_detectors));
  DetectorEfficiencies efficiencies(IndexRange2D(num_rings, num_detectors));

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
	/*{
	  char *out_filename = new char[20];
	  sprintf(out_filename, "%s_%d.out", 
	  "fan", ax_pos_num);
	  ofstream out(out_filename);
	  out << data_fan_sums;
	  delete out_filename;
	  }
	*/
      }

      make_fan_data(model_fan_data, *model_data);
      //cerr << "model min " << model_fan_data.find_min() << " ,max " << model_fan_data.find_max() << endl; 		   
      if (do_display)
	display(model_fan_data, "model");
#if 0
      {
	const string output_file_name = "testfan.s";
	shared_ptr<iostream> sino_stream = new fstream (output_file_name.c_str(), ios::out|ios::binary);
	if (!sino_stream->good())
	  {
	    error("%s: error opening file %s\n",argv[0],output_file_name.c_str());
	  }
  
	shared_ptr<ProjDataFromStream> out_proj_data_ptr =
	  new ProjDataFromStream(model_data->get_proj_data_info_ptr()->clone(),sino_stream);
  
	write_basic_interfile_PDFS_header(output_file_name, *out_proj_data_ptr);
      
	set_fan_data(*out_proj_data_ptr, model_fan_data);
      }
#endif

      for (int iter_num = 1; iter_num<=max(num_iterations, 1); ++iter_num)
	{
	  if (iter_num== 1)
	    {
	      efficiencies.fill(sqrt(data_fan_sums.sum()/model_fan_data.sum()));
	    }
	  // efficiencies
	  {
	    //fan_data = model_fan_data;
	    //if (do_display)
	    //  display(fan_data,  "model");
	    for (int eff_iter_num = 1; eff_iter_num<=num_eff_iterations; ++eff_iter_num)
	      {
		iterate_efficiencies(efficiencies, data_fan_sums, model_fan_data);
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
		    FanProjData model_times_norm = model_fan_data;
		    apply_efficiencies(model_times_norm, efficiencies);
		    //cerr << "model*norm min " << model_times_norm.find_min() << " ,max " << model_times_norm.find_max() << endl; 
		    if (do_display)
		      display( model_times_norm, "model_times_norm");

		    cerr << "KL " << KL(measured_fan_data, model_times_norm, threshold_for_KL) << endl;		  
		  }
		if (do_display)		 
		  {
		    FanProjData norm = model_fan_data;
		    norm.fill(1);
		    apply_efficiencies(norm, efficiencies);
		    display(norm, "eff norm");
		  }
		  
	    }
	  }
	}
    }

  timer.stop();
  cerr << "CPU time " << timer.value() << " secs" << endl;
  return EXIT_SUCCESS;
}
