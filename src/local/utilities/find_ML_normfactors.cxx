//
// $Id$
//
/*
    Copyright (C) 2001- $Date$, Hammersmith Imanet Ltd
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

  \brief Find normalisation factors using an ML approach

  \author Kris Thielemans

  $Date$
  $Revision$
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


START_NAMESPACE_STIR


void make_fan_sum_data(Array<1,float>& data_fan_sums, const DetPairData& det_pair_data)
{
  for (int a = det_pair_data.get_min_index(); a <= det_pair_data.get_max_index(); ++a)
    data_fan_sums[a] = det_pair_data.sum(a);
}

void make_geo_data(GeoData& geo_data, const DetPairData& det_pair_data)
{
  const int num_detectors = det_pair_data.get_num_detectors();
  const int num_crystals_per_block = geo_data.get_length()*2;
  const int num_blocks = num_detectors / num_crystals_per_block;
  assert(num_blocks * num_crystals_per_block == num_detectors);

  // TODO optimise
  DetPairData work = det_pair_data;
  work.fill(0);

  for (int a = det_pair_data.get_min_index(); a <= det_pair_data.get_max_index(); ++a)
    for (int b = det_pair_data.get_min_index(a); b <= det_pair_data.get_max_index(a); ++b)      
        {
	// mirror symmetry
	work(a,b) = 
	  det_pair_data(a,b) + 
	  det_pair_data(num_detectors-1-a,(2*num_detectors-1-b)%num_detectors);
      }

  geo_data.fill(0);

  for (int crystal_num_a = 0; crystal_num_a < num_crystals_per_block/2; ++crystal_num_a)
    for (int det_num_b = det_pair_data.get_min_index(crystal_num_a); det_num_b <= det_pair_data.get_max_index(crystal_num_a); ++det_num_b)      
      {
	for (int block_num = 0; block_num<num_blocks; ++block_num)
	  {
	    const int det_inc = block_num * num_crystals_per_block;
            const int new_det_num_a = (crystal_num_a+det_inc)%num_detectors;
            const int new_det_num_b = (det_num_b+det_inc)%num_detectors;
            if (det_pair_data.is_in_data(new_det_num_a,new_det_num_b))
	      geo_data[crystal_num_a][det_num_b] +=
	        work(new_det_num_a,new_det_num_b);
	  }
      }
  geo_data /= 2*num_blocks;
  // TODO why normalisation?
}
 
void make_block_data(BlockData& block_data, const DetPairData& det_pair_data)
{
  const int num_detectors = det_pair_data.get_num_detectors();
  const int num_blocks = block_data.get_length();
  const int num_crystals_per_block = num_detectors/num_blocks;
  assert(num_blocks * num_crystals_per_block == num_detectors);

  block_data.fill(0);
  for (int a = det_pair_data.get_min_index(); a <= det_pair_data.get_max_index(); ++a)
    for (int b = det_pair_data.get_min_index(a); b <= det_pair_data.get_max_index(a); ++b)      
      {
	block_data[a/num_crystals_per_block][(b/num_crystals_per_block)%num_blocks] += 
	  det_pair_data(a,b);
      }
  
  block_data /= square(num_crystals_per_block);
  // TODO why normalisation?
}
 
void iterate_efficiencies(Array<1,float>& efficiencies,
			  const Array<1,float>& data_fan_sums,
			  const DetPairData& model)
{
  const int num_detectors = efficiencies.get_length();

  for (int a = 0; a < num_detectors; ++a)
    {
      if (data_fan_sums[a] == 0)
	efficiencies[a] = 0;
      else
	{
          //const float denominator = inner_product(efficiencies,model[a]);
	  float denominator = 0;
           for (int b = model.get_min_index(a); b <= model.get_max_index(a); ++b)      
  	    denominator += efficiencies[b%num_detectors]*model(a,b);
	  efficiencies[a] = data_fan_sums[a] / denominator;
	}
    }
}

void iterate_geo_norm(GeoData& norm_geo_data,
		      const GeoData& measured_geo_data,
		      const DetPairData& model)
{
  make_geo_data(norm_geo_data, model);
  //norm_geo_data = measured_geo_data / norm_geo_data;
  const int num_detectors = model.get_num_detectors();
  const int num_crystals_per_block = measured_geo_data.get_length()*2;
  const float threshold = measured_geo_data.find_max()/10000.F;
  for (int a = 0; a < num_crystals_per_block/2; ++a)
    for (int b = 0; b < num_detectors; ++b)      
      {
	norm_geo_data[a][b] =
	  (measured_geo_data[a][b]>=threshold ||
	   measured_geo_data[a][b] < 10000*norm_geo_data[a][b])
	  ? measured_geo_data[a][b] / norm_geo_data[a][b]
	  : 0;
      }
}
  
void iterate_block_norm(BlockData& norm_block_data,
		      const BlockData& measured_block_data,
		      const DetPairData& model)
{
  make_block_data(norm_block_data, model);
  //norm_block_data = measured_block_data / norm_block_data;
  const int num_blocks = norm_block_data.get_length();
  const float threshold = measured_block_data.find_max()/10000.F;
  for (int a = 0; a < num_blocks; ++a)
    for (int b = 0; b < num_blocks; ++b)      
      {
	norm_block_data[a][b] =
	  (measured_block_data[a][b]>=threshold ||
	   measured_block_data[a][b] < 10000*norm_block_data[a][b])
	  ? measured_block_data[a][b] / norm_block_data[a][b]
	  : 0;
      }
}
  
#if 0
// this is a test routine for the code
// should really be in a test class
void check_geo_data()
{
  const int num_detectors = 392;
  const int num_crystals_per_block = 8;
  GeoData measured_geo_data(IndexRange2D(num_crystals_per_block/2, num_detectors));
  GeoData norm_geo_data(IndexRange2D(num_crystals_per_block/2, num_detectors));
  
  DetPairData det_pair_data(IndexRange2D(num_detectors, num_detectors));
  det_pair_data.fill(1);
  for (int a = 0; a < num_crystals_per_block/2; ++a)
    for (int b = 0; b < num_detectors; ++b)      
      {
	norm_geo_data[a][b] =(a+1)*cos((b-num_detectors/2)*_PI/num_detectors);
      }
  apply_geo_norm(det_pair_data, norm_geo_data);
  //display(det_pair_data,  "1*geo");
  make_geo_data(measured_geo_data, det_pair_data);
  {
    GeoData diff = measured_geo_data-norm_geo_data;
    std::cerr << "(org geo) min max: " << norm_geo_data.find_min() << ',' << norm_geo_data.find_max() << std::endl;
    std::cerr << "(-org geo + make geo) min max: " << diff.find_min() << ',' << diff.find_max() << std::endl;
  }

  {
    DetPairData model_det_pair_data = det_pair_data;
    for (int a = 0; a < num_detectors; ++a)
      for (int b = 0; b < num_detectors; ++b)      
	model_det_pair_data(a,b)=square((a+b-2*num_detectors)/2.F);
    {
      det_pair_data = model_det_pair_data;
      apply_geo_norm(det_pair_data, norm_geo_data);
      apply_geo_norm(det_pair_data, norm_geo_data,false);
      DetPairData diff = det_pair_data - model_det_pair_data;
      std::cerr << "(org model) min max: " << model_det_pair_data.find_min() << ',' << model_det_pair_data.find_max() << std::endl;
      std::cerr << "(-org model + apply/undo model) min max: " << diff.find_min() << ',' << diff.find_max() << std::endl;
      //display(diff,  "model- apply/undo model");
      //display(det_pair_data,  "apply/undo model");
      //display(model_det_pair_data,  " model");

    }   
    det_pair_data = model_det_pair_data;   
    apply_geo_norm(det_pair_data, norm_geo_data);
    //display(det_pair_data,  "model*geo");
    make_geo_data(measured_geo_data, det_pair_data);
    GeoData new_norm_geo_data = norm_geo_data;
    new_norm_geo_data.fill(0);
    iterate_geo_norm(new_norm_geo_data,
		     measured_geo_data,
		     model_det_pair_data);
    GeoData diff = new_norm_geo_data-norm_geo_data;
    std::cerr << "(org geo) min max: " << norm_geo_data.find_min() << ',' << norm_geo_data.find_max() << std::endl;
    std::cerr << "(-org geo + iterate geo) min max: " << diff.find_min() << ',' << diff.find_max() << std::endl;
  }

}

#endif



END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{  
  //check_geo_data();
  if (argc!=6)
    {
      std::cerr << "Usage: " << argv[0] 
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
  const std::string out_filename_prefix = argv[1];
  /*  const int num_rings = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings();
  */
  const int num_detectors = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_detectors_per_ring();
  const int num_crystals_per_block = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->
    get_num_transaxial_crystals_per_block();
  const int num_blocks = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->
    get_num_transaxial_blocks();

  CPUTimer timer;
  timer.start();

  const int segment_num = 0;
  DetPairData det_pair_data;
  DetPairData model_det_pair_data;
  Array<1,float> data_fan_sums(num_detectors);
  Array<1,float> efficiencies(num_detectors);
  assert(num_crystals_per_block%2 == 0);
  GeoData measured_geo_data(IndexRange2D(num_crystals_per_block/2, num_detectors));
  GeoData norm_geo_data(IndexRange2D(num_crystals_per_block/2, num_detectors));
  BlockData measured_block_data(IndexRange2D(num_blocks, num_blocks));
  BlockData norm_block_data(IndexRange2D(num_blocks, num_blocks));

  for (int ax_pos_num = measured_data->get_min_axial_pos_num(segment_num);
       ax_pos_num <= measured_data->get_max_axial_pos_num(segment_num);
       ++ax_pos_num)
    {
      // next could be local if KL is not computed below
      DetPairData measured_det_pair_data;
      float threshold_for_KL;
      // compute factors dependent on the data
      {
	make_det_pair_data(measured_det_pair_data, *measured_data, segment_num, ax_pos_num);
	threshold_for_KL = measured_det_pair_data.find_max()/100000.F;
	std::cerr << "ax_pos " << ax_pos_num << std::endl;
	//display(measured_det_pair_data, "measured data");
	
	make_fan_sum_data(data_fan_sums, measured_det_pair_data);
	make_geo_data(measured_geo_data, measured_det_pair_data);
	make_block_data(measured_block_data, measured_det_pair_data);
	if (do_display)
	  display(measured_block_data, "raw block data from measurements");	
	/*{
	  char *out_filename = new char[20];
	  sprintf(out_filename, "%s_%d.out", 
	  "fan", ax_pos_num);
	  std::ofstream out(out_filename);
	  out << data_fan_sums;
	  delete out_filename;
	  }
	*/
      }

      make_det_pair_data(model_det_pair_data, *model_data, segment_num, ax_pos_num);
      //display(model_det_pair_data, "model");

      for (int iter_num = 1; iter_num<=max(num_iterations, 1); ++iter_num)
	{
	  if (iter_num== 1)
	    {
	      efficiencies.fill(sqrt(data_fan_sums.sum()/model_det_pair_data.sum()));
	      norm_geo_data.fill(1);
	      norm_block_data.fill(1);
	    }
	  // efficiencies
	  {
	    det_pair_data = model_det_pair_data;
	    apply_geo_norm(det_pair_data, norm_geo_data);
            apply_block_norm(det_pair_data, norm_block_data);
	    if (do_display)
	      display(det_pair_data,  "model*geo*block");
	    for (int eff_iter_num = 1; eff_iter_num<=num_eff_iterations; ++eff_iter_num)
	      {
		iterate_efficiencies(efficiencies, data_fan_sums, det_pair_data);
		{
		  char *out_filename = new char[out_filename_prefix.size() + 30];
		  sprintf(out_filename, "%s_%s_%d_%d_%d.out", 
			  out_filename_prefix.c_str(), "eff", ax_pos_num, iter_num, eff_iter_num);
		  std::ofstream out(out_filename);
		  out << efficiencies;
		  delete out_filename;
		}
		if (do_KL)
		  {
		    DetPairData model_times_norm = det_pair_data;
		    apply_efficiencies(model_times_norm, efficiencies);
		    if (do_display)
		      display( model_times_norm, "model_times_norm");
		    //std::cerr << "model_times_norm min max: " << model_times_norm.find_min() << ',' << model_times_norm.find_max() << std::endl;

		    std::cerr << "KL " << KL(measured_det_pair_data, model_times_norm, threshold_for_KL) << std::endl;		  
		  }
		if (do_display)		 
		  {
		    DetPairData norm = det_pair_data;
		    norm.fill(1);
		    apply_efficiencies(norm, efficiencies);
		    display(norm, "eff norm");
		  }
		  
	    }
	  }
	  if (num_iterations==0)
	    break;
	  // geo norm
	  {
	    det_pair_data = model_det_pair_data;
	    apply_efficiencies(det_pair_data, efficiencies);
	    apply_block_norm(det_pair_data, norm_block_data);
            iterate_geo_norm(norm_geo_data, measured_geo_data, det_pair_data);
	    { // check 
	      for (int a=0; a<measured_geo_data.get_length(); ++a)
		  for (int b=0; b<num_detectors; ++b)
		    if (norm_geo_data[a][b]==0 && measured_geo_data[a][b]!=0)
		      warning("norm geo 0 at a=%d b=%d measured value=%g\n",
			      a,b,measured_geo_data[a][b]);
	    }
	    {
	      char *out_filename = new char[out_filename_prefix.size() + 30];
	      sprintf(out_filename, "%s_%s_%d_%d.out", 
		      out_filename_prefix.c_str(), "geo", ax_pos_num, iter_num);
	      std::ofstream out(out_filename);
	      out << norm_geo_data;
	      delete out_filename;
	    }
	    if (do_KL)
	      {
		apply_geo_norm(det_pair_data, norm_geo_data);
                for (int a = det_pair_data.get_min_index(); a <= det_pair_data.get_max_index(); ++a)
                  for (int b = det_pair_data.get_min_index(a); b <= det_pair_data.get_max_index(a); ++b)      
		    if (det_pair_data(a,b)==0 && measured_det_pair_data(a,b)!=0)
		      warning("geo 0 at a=%d b=%d measured value=%g\n",
			      a,b,measured_det_pair_data(a,b));
		std::cerr << "KL " << KL(measured_det_pair_data, det_pair_data, threshold_for_KL) << std::endl;
	      }
	    if (do_display)		 
	      {
		DetPairData norm = det_pair_data;
		norm.fill(1);
		apply_geo_norm(norm, norm_geo_data);
		display(norm, "geo norm");
	      }
	  }
          // block norm
	  {
	    det_pair_data = model_det_pair_data;
	    apply_efficiencies(det_pair_data, efficiencies);
	    apply_geo_norm(det_pair_data, norm_geo_data);
            iterate_block_norm(norm_block_data, measured_block_data, det_pair_data);
	    { // check 
	      for (int a=0; a<measured_block_data.get_length(); ++a)
		  for (int b=0; b<measured_block_data[0].get_length(); ++b)
		    if (norm_block_data[a][b]==0 && measured_block_data[a][b]!=0)
		      warning("block norm 0 at a=%d b=%d measured value=%g\n",
			      a,b,measured_block_data[a][b]);
	    }
	    {
	      char *out_filename = new char[out_filename_prefix.size() + 30];
	      sprintf(out_filename, "%s_%s_%d_%d.out", 
		      out_filename_prefix.c_str(), "block", ax_pos_num, iter_num);
	      std::ofstream out(out_filename);
	      out << norm_block_data;
	      delete out_filename;
	    }
	    if (do_KL)
	      {
		apply_block_norm(det_pair_data, norm_block_data);
		std::cerr << "KL " << KL(measured_det_pair_data, det_pair_data, threshold_for_KL) << std::endl;
	      }
	    if (do_display)		 
	      {
		DetPairData norm = det_pair_data;
		norm.fill(1);
		apply_block_norm(norm, norm_block_data);
		display(norm_block_data, "raw block norm");
		display(norm, "block norm");
	      }
	  }
	}
    }
  timer.stop();
  std::cerr << "CPU time " << timer.value() << " secs" << std::endl;
  return EXIT_SUCCESS;
}
