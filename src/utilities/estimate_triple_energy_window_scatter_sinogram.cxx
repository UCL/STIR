/*
  Copyright (C) 2019, National Physical Laboratory
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup utilities
  \brief Calculate Triple energy window scatter


  \author Daniel Deidda
*/


#include "stir/SegmentByView.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/Succeeded.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ExamInfo.h"
#include "stir/Succeeded.h"
#include "stir/NumericInfo.h"
#include "stir/KeyParser.h"
#include "stir/is_null_ptr.h"
#include "stir/stir_math.h"
#include "stir/ArrayFunction.h"

#include "stir/IndexRange3D.h"
#include "stir/ArrayFilter3DUsingConvolution.h"


#include <fstream> 
#include <iostream> 
#include <functional>
#include <algorithm>
#include <memory>
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::cout;
using std::endl;
using std::fstream;
using std::unary_function;
using std::transform;
using std::max;
using std::min;
using std::string;
using std::vector;
#endif


USING_NAMESPACE_STIR

class estimate_TEW_scatter : public KeyParser, ArrayFilter3DUsingConvolution<float>
{
public:

  estimate_TEW_scatter();

  std::string scatter_filename;
  std::string lower_filename,upper_filename;
  double lower_width, peak_width, upper_width;
  bool do_smooth;

  Succeeded compute();
private:


//  virtual void set_defaults();
  virtual void initialise_keymap();
//  virtual bool post_processing();
};
void
 estimate_TEW_scatter::
initialise_keymap()
{
  add_start_key(" Estimate_TEW_scatter Parameters");
  add_key("output scatter filename",&scatter_filename);
  add_key("lower filename",&lower_filename);
  add_key("upper filename",&upper_filename);
  add_key("lower width",&lower_width);
  add_key("peak width",&peak_width);
  add_key("upper width",&upper_width);
  add_key("do smooth",&do_smooth);
  add_stop_key("END");

}

estimate_TEW_scatter::estimate_TEW_scatter()
{
  initialise_keymap();
}

int 
main(int argc, char **argv)
{
  if(argc<2)
    {
      cerr<< "Usage: " << argv[0] << "\n\t"
      << "[par_file]\n\t"
      << " Should contain the following:\n"
      << "Estimate_TEW_scatter Parameters:= \n"
      << "output scatter filename :=\n"
      << "lower filename :=\n"
      << "upper filename :=\n"
      << "lower width :=\n"
      << "peak width :=\n"
      << "upper width :=\n"
      << "do smooth :=\n"
      << "END :=\n";
      exit(EXIT_FAILURE);
    }

  // first process command line options
 estimate_TEW_scatter estimate;

  if (argc==0)
    { cerr << "No par file on command line\n"; exit(EXIT_FAILURE); }
  else{
      if (argv[1]!=0)
      {
        if (estimate.parse(argv[1]) == false)
            exit(EXIT_FAILURE);
      }
      else
          estimate.ask_parameters();}

  // find output filename
  const string output_file_name = estimate.scatter_filename;

//     start the main processing
//     read windows data
      shared_ptr<ProjData> lower_sptr;
      shared_ptr<ProjData> upper_sptr;
      shared_ptr<ProjData> out_scatter_proj_data_ptr;

      lower_sptr = ProjData::read_from_file(estimate.lower_filename);
      upper_sptr = ProjData::read_from_file(estimate.upper_filename);


	  shared_ptr<ProjDataInfo> 
        output_proj_data_info_sptr((*lower_sptr).get_proj_data_info_sptr()->clone());

      out_scatter_proj_data_ptr.reset(new ProjDataInterfile((*lower_sptr).get_exam_info_sptr(),
							output_proj_data_info_sptr, 
                            output_file_name));

//      if (num_files>1)
//	{
//	  // reset time-frames as we don't really know what's happening with all this
//      ExamInfo new_exam_info(out_scatter_proj_data_ptr->get_exam_info());
//	  new_exam_info.set_time_frame_definitions(TimeFrameDefinitions());
//      out_scatter_proj_data_ptr->set_exam_info(new_exam_info);
//	}


      // do reading/writing in a loop over segments
      for (int segment_num = out_scatter_proj_data_ptr->get_min_segment_num();
       segment_num <= out_scatter_proj_data_ptr->get_max_segment_num();
	   ++segment_num)
	{   

      SegmentByView<float> lower_segment_by_view =
        (*lower_sptr).get_segment_by_view(segment_num);
      SegmentByView<float> upper_segment_by_view =
        (*upper_sptr).get_segment_by_view(segment_num);
      SegmentByView<float> scatter_segment_by_view=
        (*lower_sptr).get_segment_by_view(segment_num);
      SegmentByView<float> filter_lower_segment_by_view =
      (*lower_sptr).get_segment_by_view(segment_num);
      SegmentByView<float> filter_upper_segment_by_view =
      (*upper_sptr).get_segment_by_view(segment_num);

     if(estimate.do_smooth){
         IndexRange3D kernel_size(0,0,-2,2,-2,2);
         Array<3,float> kernel(kernel_size);
         kernel.fill(1/25.F);
         ArrayFilter3DUsingConvolution<float> filter3d(kernel);
         filter3d(filter_lower_segment_by_view,lower_segment_by_view);
         filter3d(filter_upper_segment_by_view,upper_segment_by_view);
     }


      // construct function object that does the manipulations on each data
      float power=1 ;
      float add_scalar=0;
      float  min_threshold = NumericInfo<float>().min_value();
      float  max_threshold = NumericInfo<float>().max_value();

//     The following apply the the TEW method C_{scatter}=(C_{lower}/W_{lower}+C_{upper}/W_{upper})*W_{peak}/2

      pow_times_add divide_by_lower_width(add_scalar, 1/estimate.lower_width,power ,min_threshold ,max_threshold );
      pow_times_add divide_by_upper_width(add_scalar, 1/estimate.upper_width,power ,min_threshold ,max_threshold );
      pow_times_add mult_by_half_peak_width(add_scalar, estimate.peak_width/2,power ,min_threshold ,max_threshold );

      in_place_apply_function(filter_lower_segment_by_view, divide_by_lower_width);
      in_place_apply_function(filter_upper_segment_by_view, divide_by_upper_width);

       scatter_segment_by_view=filter_lower_segment_by_view;
       scatter_segment_by_view+=filter_upper_segment_by_view;


       in_place_apply_function(scatter_segment_by_view, mult_by_half_peak_width);


     if (!(out_scatter_proj_data_ptr->set_segment(scatter_segment_by_view) == Succeeded::yes))
        warning("Error set_segment %d\n", segment_num);
    }

  std::cout<<"TEW scatter estimated "<<std::endl;
  return EXIT_SUCCESS;
}
