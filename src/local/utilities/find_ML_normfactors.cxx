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
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#define ARRAY_CONST_IT  

#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjData.h"
#include "stir/Scanner.h"
#include "stir/Bin.h"
#include "stir/stream.h"
#include "stir/Array.h"
#include "stir/Sinogram.h"
#include "stir/IndexRange2D.h"
#include "stir/display.h"
#include "stir/CPUTimer.h"
#include <iostream>
#include <fstream>
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ofstream;
using std::string;
#endif

START_NAMESPACE_STIR

typedef Array<2,float> DetPairData;
typedef Array<2,float> GeoData;

void make_det_pair_data(DetPairData& det_pair_data,
			const ProjData& proj_data,
			const int segment_num,
			const int ax_pos_num)
{
  const ProjDataInfo* proj_data_info_ptr =
    proj_data.get_proj_data_info_ptr();
  const ProjDataInfoCylindricalNoArcCorr& proj_data_info =
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(*proj_data_info_ptr);

  const int num_detectors = 
    proj_data_info.get_scanner_ptr()->get_num_detectors_per_ring();
  det_pair_data.grow(IndexRange2D(0,num_detectors-1, 0,num_detectors-1));
  det_pair_data.fill(0);

  shared_ptr<Sinogram<float> > pos_sino_ptr =
    new Sinogram<float>(proj_data.get_sinogram(ax_pos_num,segment_num));
  shared_ptr<Sinogram<float> > neg_sino_ptr;
  if (segment_num == 0)
    neg_sino_ptr = pos_sino_ptr;
  else
    neg_sino_ptr =
      new Sinogram<float>(proj_data.get_sinogram(ax_pos_num,-segment_num));
  
    
  for (int view_num = 0; view_num < num_detectors/2; view_num++)
    for (int tang_pos_num = proj_data.get_min_tangential_pos_num();
	 tang_pos_num <= proj_data.get_max_tangential_pos_num();
	 ++tang_pos_num)
      {
	int det_num_a = 0;
	int det_num_b = 0;

	proj_data_info.get_det_num_pair_for_view_tangential_pos_num(det_num_a, det_num_b, view_num, tang_pos_num);

	det_pair_data[det_num_a][det_num_b] =
	  (*pos_sino_ptr)[view_num][tang_pos_num];
	det_pair_data[det_num_b][det_num_a] =
	  (*neg_sino_ptr)[view_num][tang_pos_num];
      }
}

void make_geo_data(GeoData& geo_data, const DetPairData& det_pair_data)
{
  const int num_detectors = det_pair_data.get_length();
  const int num_crystals_per_block = geo_data.get_length()*2;
  const int num_blocks = num_detectors / num_crystals_per_block;
  assert(num_blocks * num_crystals_per_block == num_detectors);

  // TODO optimise
  DetPairData work = det_pair_data;
  work.fill(0);

  for (int det_num_a = 0; det_num_a < num_detectors; ++det_num_a)
    for (int det_num_b = 0; det_num_b < num_detectors; ++det_num_b)      
      {
	// mirror symmetry
	work[det_num_a][det_num_b] = 
	  det_pair_data[det_num_a][det_num_b] + 
	  det_pair_data[num_detectors-1-det_num_a][num_detectors-1-det_num_b];
      }

  geo_data.fill(0);

  for (int crystal_num_a = 0; crystal_num_a < num_crystals_per_block/2; ++crystal_num_a)
    for (int det_num_b = 0; det_num_b < num_detectors; ++det_num_b)      
      {
	for (int block_num = 0; block_num<num_blocks; ++block_num)
	  {
	    const int det_inc = block_num * num_crystals_per_block;
	    geo_data[crystal_num_a][det_num_b] +=
	      work[(crystal_num_a+det_inc)%num_detectors][(det_num_b+det_inc)%num_detectors];
	  }
      }
  geo_data /= 2*num_blocks;
}

void apply_geo_norm(DetPairData& det_pair_data, const GeoData& geo_data)
{
  const int num_detectors = det_pair_data.get_length();
  const int num_crystals_per_block = geo_data.get_length()*2;

  for (int a = 0; a < num_detectors; ++a)
    for (int b = 0; b < num_detectors; ++b)      
      {
        int newa = a % num_crystals_per_block;
	int newb = b - (a - newa); 
	if (newa > num_crystals_per_block - 1 - newa)
	  { 
	    newa = num_crystals_per_block - 1 - newa; 
	    newb = - newb + num_crystals_per_block - 1;
	  }
	// note: add 2*num_detectors to newb to avoid using mod with negative numbers
	det_pair_data[a][b] *=
	  geo_data[newa][(2*num_detectors + newb)%num_detectors];
      }
}

void make_fan_sum_data(Array<1,float>& data_fan_sums, const DetPairData& det_pair_data)
{
  const int num_detectors = det_pair_data.get_length();
  for (int det_num_a = 0; det_num_a < num_detectors; ++det_num_a)
    data_fan_sums[det_num_a] = det_pair_data[det_num_a].sum();
}

void apply_efficiencies(DetPairData& det_pair_data, const Array<1,float>& efficiencies)
{
  const int num_detectors = det_pair_data.get_length();
  for (int det_num_a = 0; det_num_a < num_detectors; ++det_num_a)
    for (int det_num_b = 0; det_num_b < num_detectors; ++det_num_b)      
      {
	det_pair_data[det_num_a][det_num_b] *=
	  efficiencies[det_num_a]*efficiencies[det_num_b];
      }
}
  
void iterate_efficiencies(Array<1,float>& efficiencies,
			  const Array<1,float>& data_fan_sums,
			  const DetPairData& model)
{
  const int num_detectors = efficiencies.get_length();

  for (int det_num_a = 0; det_num_a < num_detectors; ++det_num_a)
    {
      if (data_fan_sums[det_num_a] == 0)
	efficiencies[det_num_a] = 0;
      else
	{
	  float denominator = 0;
	  for (int det_num_b = 0; det_num_b < num_detectors; ++det_num_b)      
	    denominator += efficiencies[det_num_b]*model[det_num_a][det_num_b];
	  efficiencies[det_num_a] = data_fan_sums[det_num_a] / denominator;
	}
    }
}

void iterate_geo_norm(GeoData& norm_geo_data,
		      const GeoData& measured_geo_data,
		      const DetPairData& model)
{
  make_geo_data(norm_geo_data, model);
  //norm_geo_data = measured_geo_data / norm_geo_data;
  const int num_detectors = model.get_length();
  const int num_crystals_per_block = measured_geo_data.get_length()*2;
  const float threshold = measured_geo_data.find_max()/100000.F;
  for (int a = 0; a < num_crystals_per_block/2; ++a)
    for (int b = 0; b < num_detectors; ++b)      
      {
	norm_geo_data[a][b] =
	  (measured_geo_data[a][b]>threshold)
	  ? measured_geo_data[a][b] / norm_geo_data[a][b]
	  : 0;
      }
}
  
#if 0
void check_geo_data()
  {
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
      cerr << "(org geo) min max: " << norm_geo_data.find_min() << ',' << norm_geo_data.find_max() << endl;
      cerr << "(-org geo + make geo) min max: " << diff.find_min() << ',' << diff.find_max() << endl;
    }
  }

#endif
  
inline float KL(const float a, const float b)
{
  assert(a>=0);
  assert(b>=0);
  float res = a==0 ? b : (a*(log(a)-log(b)) + b - a);
  assert(res>=0);
  return res;
}

template <int num_dimensions, typename elemT>
float KL(const Array<num_dimensions, elemT>& a, const Array<num_dimensions, elemT>& b)
{
  float sum = 0;
  Array<num_dimensions, elemT>::const_full_iterator iter_a = a.begin_all();
  Array<num_dimensions, elemT>::const_full_iterator iter_b = b.begin_all();
  while (iter_a != a.end_all())
    {
      sum += KL(*iter_a++, *iter_b++);
    }
  return sum;
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc!=6)
    {
      cerr << "Usage: " << argv[0] 
	   << " efficiency_out_filename_prefix measured_data model num_iterations num_eff_iterations\n";
      return EXIT_FAILURE;
    }

  const int num_eff_iterations = atoi(argv[5]);
  const int num_iterations = atoi(argv[4]);
  shared_ptr<ProjData> model_data = ProjData::read_from_file(argv[3]);
  shared_ptr<ProjData> measured_data = ProjData::read_from_file(argv[2]);
  const string out_filename_prefix = argv[1];
  const int num_detectors = 
    measured_data->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_detectors_per_ring();
  const int num_crystals_per_block = 8;

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


  for (int ax_pos_num = measured_data->get_min_axial_pos_num(segment_num);
       ax_pos_num <= measured_data->get_max_axial_pos_num(segment_num);
       ++ax_pos_num)
    {
      // next could be local of KL is not computed below
      DetPairData measured_det_pair_data;
      // compute factors dependent on the data
      {
	make_det_pair_data(measured_det_pair_data, *measured_data, segment_num, ax_pos_num);
#if 1
	// insert known geo factors
	for (int a = 0; a < num_crystals_per_block/2; ++a)
	  for (int b = 0; b < num_detectors; ++b)      
	    {
	      norm_geo_data[a][b] =(a+1)*cos((b-num_detectors/2)*_PI/num_detectors)+.1;
	    }
	apply_geo_norm(measured_det_pair_data, norm_geo_data);
#endif
	cerr << "ax_pos " << ax_pos_num << endl;
	//display(measured_det_pair_data, "measured data");
	
	make_fan_sum_data(data_fan_sums, measured_det_pair_data);
	make_geo_data(measured_geo_data, measured_det_pair_data);
	
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

      make_det_pair_data(model_det_pair_data, *model_data, segment_num, ax_pos_num);
      //display(model_det_pair_data, "model");

      for (int iter_num = 0; iter_num<num_iterations; ++iter_num)
	{
	  if (iter_num== 0)
	    {
	      efficiencies.fill(data_fan_sums.sum()/model_det_pair_data.sum());
	      norm_geo_data.fill(1);
	    }
	  // efficiencies
	  {
	    det_pair_data = model_det_pair_data;
	    apply_geo_norm(det_pair_data, norm_geo_data);
	    //display(det_pair_data,  "model*geo");
	    for (int eff_iter_num = 0; eff_iter_num<num_eff_iterations; ++eff_iter_num)
	      {
		iterate_efficiencies(efficiencies, data_fan_sums, det_pair_data);
		{
		  char *out_filename = new char[out_filename_prefix.size() + 30];
		  sprintf(out_filename, "%s_%s_%d_%d_%d.out", 
			  out_filename_prefix.c_str(), "eff", ax_pos_num, iter_num, eff_iter_num);
		  ofstream out(out_filename);
		  out << efficiencies;
		  delete out_filename;
		}
		{
		  DetPairData model_times_norm = det_pair_data;
		  apply_efficiencies(model_times_norm, efficiencies);
		  //display( model_times_norm, "model_times_norm");
		  //cerr << "model_times_norm min max: " << model_times_norm.find_min() << ',' << model_times_norm.find_max() << endl;

		  cerr << "KL " << KL(measured_det_pair_data, model_times_norm) << endl;
		}
		  
	    }
	  }
	  // geo norm
	  {
	    det_pair_data = model_det_pair_data;
	    apply_efficiencies(det_pair_data, efficiencies);
	    iterate_geo_norm(norm_geo_data, measured_geo_data, det_pair_data);
	    {
	      char *out_filename = new char[out_filename_prefix.size() + 30];
	      sprintf(out_filename, "%s_%s_%d_%d.out", 
		      out_filename_prefix.c_str(), "geo", ax_pos_num, iter_num);
	      ofstream out(out_filename);
	      out << norm_geo_data;
	      delete out_filename;
	    }
	    {
	      apply_geo_norm(det_pair_data, norm_geo_data);
	      cerr << "KL " << KL(measured_det_pair_data, det_pair_data) << endl;
	    }

	  }
	}
    }

  timer.stop();
  cerr << "CPU time " << timer.value() << " secs" << endl;
  return EXIT_SUCCESS;
}
