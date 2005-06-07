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
#include "stir/ProjData.h"
//#include "stir/ProjDataInterfile.h"

START_NAMESPACE_STIR


//************** 3D

void make_block_data(BlockData3D& block_data, const FanProjData& fan_data)
{
  const int num_axial_detectors = fan_data.get_num_rings();
  const int num_transaxial_detectors = fan_data.get_num_detectors_per_ring();
  const int num_axial_blocks = block_data.get_num_rings();
  const int num_transaxial_blocks = block_data.get_num_detectors_per_ring();
  const int num_axial_crystals_per_block = num_axial_detectors/num_axial_blocks;
  assert(num_axial_blocks * num_axial_crystals_per_block == num_axial_detectors);
  const int num_transaxial_crystals_per_block = num_transaxial_detectors/num_transaxial_blocks;
  assert(num_transaxial_blocks * num_transaxial_crystals_per_block == num_transaxial_detectors);
  
  block_data.fill(0);
  for (int ra = fan_data.get_min_ra(); ra <= fan_data.get_max_ra(); ++ra)
    for (int a = fan_data.get_min_a(); a <= fan_data.get_max_a(); ++a)
      // loop rb from ra to avoid double counting
      for (int rb = max(ra,fan_data.get_min_rb(ra)); rb <= fan_data.get_max_rb(ra); ++rb)
        for (int b = fan_data.get_min_b(a); b <= fan_data.get_max_b(a); ++b)      
        {
          block_data(ra/num_axial_crystals_per_block,a/num_transaxial_crystals_per_block,
                     rb/num_axial_crystals_per_block,b/num_transaxial_crystals_per_block) +=
	  fan_data(ra,a,rb,b);
        }  
}

void iterate_efficiencies(DetectorEfficiencies& efficiencies,
			  const Array<2,float>& data_fan_sums,
			  const FanProjData& model)
{
  const int num_detectors_per_ring = model.get_num_detectors_per_ring();
  
  assert(model.get_min_ra() == data_fan_sums.get_min_index());
  assert(model.get_max_ra() == data_fan_sums.get_max_index());
  assert(model.get_min_a() == data_fan_sums[data_fan_sums.get_min_index()].get_min_index());
  assert(model.get_max_a() == data_fan_sums[data_fan_sums.get_min_index()].get_max_index());
  for (int ra = model.get_min_ra(); ra <= model.get_max_ra(); ++ra)
    for (int a = model.get_min_a(); a <= model.get_max_a(); ++a)
    {
      if (data_fan_sums[ra][a] == 0)
	efficiencies[ra][a] = 0;
      else
	{
     	  float denominator = 0;
           for (int rb = model.get_min_rb(ra); rb <= model.get_max_rb(ra); ++rb)
             for (int b = model.get_min_b(a); b <= model.get_max_b(a); ++b)
  	       denominator += efficiencies[rb][b%num_detectors_per_ring]*model(ra,a,rb,b);
	  efficiencies[ra][a] = data_fan_sums[ra][a] / denominator;
	}
    }
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

void iterate_block_norm(BlockData3D& norm_block_data,
		      const BlockData3D& measured_block_data,
		      const FanProjData& model)
{
  make_block_data(norm_block_data, model);
  //norm_block_data = measured_block_data / norm_block_data;
  const float threshold = measured_block_data.find_max()/10000.F;
  for (int ra = norm_block_data.get_min_ra(); ra <= norm_block_data.get_max_ra(); ++ra)
    for (int a = norm_block_data.get_min_a(); a <= norm_block_data.get_max_a(); ++a)
      // loop rb from ra to avoid double counting
      for (int rb = max(ra,norm_block_data.get_min_rb(ra)); rb <= norm_block_data.get_max_rb(ra); ++rb)
        for (int b = norm_block_data.get_min_b(a); b <= norm_block_data.get_max_b(a); ++b)      
      {
	norm_block_data(ra,a,rb,b) =
	  (measured_block_data(ra,a,rb,b)>=threshold ||
	   measured_block_data(ra,a,rb,b) < 10000*norm_block_data(ra,a,rb,b))
	  ? measured_block_data(ra,a,rb,b) / norm_block_data(ra,a,rb,b)
	  : 0;
      }
}



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
  const int num_rings = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->
    get_num_rings();
  const int num_detectors_per_ring = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->
    get_num_detectors_per_ring();
  const int num_transaxial_blocks =
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->
    get_num_transaxial_blocks();
  const int num_axial_blocks =
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->
    get_num_axial_blocks();

  CPUTimer timer;
  timer.start();

  FanProjData model_fan_data;
  FanProjData fan_data;
  Array<2,float> data_fan_sums(IndexRange2D(num_rings, num_detectors_per_ring));
  DetectorEfficiencies efficiencies(IndexRange2D(num_rings, num_detectors_per_ring));
  BlockData3D measured_block_data(num_axial_blocks, num_transaxial_blocks,
                                  num_axial_blocks-1, num_transaxial_blocks-1);
  BlockData3D norm_block_data(num_axial_blocks, num_transaxial_blocks,
                              num_axial_blocks-1, num_transaxial_blocks-1);
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
      make_block_data(measured_block_data, measured_fan_data);
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
    
    make_fan_data(model_fan_data, *model_data);
    //std::cerr << "model min " << model_fan_data.find_min() << " ,max " << model_fan_data.find_max() << std::endl; 		   
    if (do_display)
      display(model_fan_data, "model");
#if 0
    {
      shared_ptr<ProjData> out_proj_data_ptr =
        new ProjDataInterfile(model_data->get_proj_data_info_ptr()->clone,
			      output_file_name);      
      
      set_fan_data(*out_proj_data_ptr, model_fan_data);
    }
#endif
    
    for (int iter_num = 1; iter_num<=max(num_iterations, 1); ++iter_num)
    {
      if (iter_num== 1)
      {
        efficiencies.fill(sqrt(data_fan_sums.sum()/model_fan_data.sum()));
        norm_block_data.fill(1);
      }
      // efficiencies
      {
        fan_data = model_fan_data;
        //apply_geo_norm(fan_data, norm_geo_data);
        apply_block_norm(fan_data, norm_block_data);
        if (do_display)
          display(fan_data,  "model*geo*block");
        for (int eff_iter_num = 1; eff_iter_num<=num_eff_iterations; ++eff_iter_num)
        {
          iterate_efficiencies(efficiencies, data_fan_sums, fan_data);
          {
            char *out_filename = new char[out_filename_prefix.size() + 30];
            sprintf(out_filename, "%s_%s_%d_%d.out", 
              out_filename_prefix.c_str(), "eff", iter_num, eff_iter_num);
	    std::ofstream out(out_filename);
            out << efficiencies;
            delete out_filename;
          }
          if (do_KL)
          {
            apply_efficiencies(fan_data, efficiencies);
            //std::cerr << "model*norm min " << fan_data.find_min() << " ,max " << fan_data.find_max() << std::endl; 
            if (do_display)
              display( fan_data, "model_times_norm");
	    std::cerr << "KL " << KL(measured_fan_data, fan_data, threshold_for_KL) << std::endl;
            // now restore for further iterations
            fan_data = model_fan_data;
            //apply_geo_norm(fan_data, norm_geo_data);
            apply_block_norm(fan_data, norm_block_data);
          }
          if (do_display)		 
          {
            fan_data.fill(1);
            apply_efficiencies(fan_data, efficiencies);
            display(fan_data, "eff norm");
            // now restore for further iterations
            fan_data = model_fan_data;
            //apply_geo_norm(fan_data, norm_geo_data);
            apply_block_norm(fan_data, norm_block_data);
          }
          
        }
      } // end efficiencies
      // block norm
      {
        fan_data = model_fan_data;
        apply_efficiencies(fan_data, efficiencies);
        //apply_geo_norm(fan_data, norm_geo_data);
        iterate_block_norm(norm_block_data, measured_block_data, fan_data);
#if 0
        { // check 
          for (int a=0; a<measured_block_data.get_length(); ++a)
            for (int b=0; b<measured_block_data[0].get_length(); ++b)
              if (norm_block_data[a][b]==0 && measured_block_data[a][b]!=0)
                warning("block norm 0 at a=%d b=%d measured value=%g\n",
                a,b,measured_block_data[a][b]);
        }
#endif
        {
          char *out_filename = new char[out_filename_prefix.size() + 30];
          sprintf(out_filename, "%s_%s_%d.out", 
            out_filename_prefix.c_str(), "block", iter_num);
          std::ofstream out(out_filename);
          out << norm_block_data;
          delete out_filename;
        }
        if (do_KL)
        {
          apply_block_norm(fan_data, norm_block_data);
	  std::cerr << "KL " << KL(measured_fan_data, fan_data, threshold_for_KL) << std::endl;
        }
        if (do_display)		 
        {
          fan_data.fill(1);
          apply_block_norm(fan_data, norm_block_data);
          display(norm_block_data, "raw block norm");
          display(fan_data, "block norm");
        }
      } // end block
    
    }
  }    
  timer.stop();
  std::cerr << "CPU time " << timer.value() << " secs" << std::endl;
  return EXIT_SUCCESS;
}
