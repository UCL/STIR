/*!

  \file
  \ingroup test
  
  \brief Test program for the stir_math  utility
    
  Takes as single command line argument the filename of the stir_math executable 
  (with a relative or absolute path). If no argument is given, the executable im
  the path is used.
  \warning will overwite (and then delete) files called STIR$$*.
  \author Kris Thielemans
*/
/*
    Copyright (C) 2002-2003, Hammersmith Imanet Ltd
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

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfo.h"
#include "stir/Scanner.h"
#include "stir/IndexRange.h"
#include "stir/CartesianCoordinate3D.h"

#include "stir/RunTests.h"
#include "stir/IO/InterfileOutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/ProjDataInterfile.h"
#include "stir/SegmentByView.h"
#include "stir/thresholding.h"
#include "stir/Succeeded.h"
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef STIR_NO_NAMESPACE
using std::generate;
using std::string;
using std::cerr;
#endif

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for the stir_math utility

  \warning will overwite (and then delete) files called STIR$$*.
*/
class stir_mathTests : public RunTests
{
public:
  stir_mathTests(const string& stir_math_executable)
    : stir_math_executable(stir_math_executable)
  {}
  
  void run_tests();
private:
  string stir_math_executable;
  //! returns true if it ran it successfully
  bool run_stir_math(const char * const arguments);

};


bool
stir_mathTests::
run_stir_math(const char * const arguments)
{
  static string cmdline;
  static string error_string;

  // note: add extra quotes around executable name to cope with spaces in the filename
  cmdline = "\"" + stir_math_executable + "\"";
  cmdline += ' ' ;
  cmdline += arguments;
  error_string = "error executing command '";
  error_string += cmdline;
  error_string += "'";
  cerr << "\nExecuting '" << cmdline << "'\n";
  return check(system(cmdline.c_str())==EXIT_SUCCESS,
	       "executing previous command returned non-zero status\n.");
}

void
stir_mathTests::run_tests()

{ 
  cerr << "Tests for the stir_math utility\n";
  cerr << "WARNING: will overwite (and then delete) files called STIRtmp*\n";
  
  // images
  {
    CartesianCoordinate3D<float> origin (0,0,0);  
    CartesianCoordinate3D<float> grid_spacing (3,4,5); 
    IndexRange<3> 
      range(CartesianCoordinate3D<int>(0,-2,-3),
            CartesianCoordinate3D<int>(1,2,3));
    VoxelsOnCartesianGrid<float>  data1(range,origin, grid_spacing);
    VoxelsOnCartesianGrid<float>  data2(range,origin, grid_spacing);
    VoxelsOnCartesianGrid<float>  data3(range,origin, grid_spacing);
    VoxelsOnCartesianGrid<float> calc_data;
    shared_ptr<DiscretisedDensity<3,float> > out_data_ptr;
    
    // fill with some random numbers
    srand(1);
    generate(data1.begin_all(), data1.end_all(), rand);
    generate(data2.begin_all(), data2.end_all(), rand);
    generate(data3.begin_all(), data3.end_all(), rand);
    
    InterfileOutputFileFormat output_file_format;
    check(output_file_format.write_to_file("STIRtmp1.v", data1)== Succeeded::yes,
	  "test writing Interfile image STIRtmp1");
    check(output_file_format.write_to_file("STIRtmp2.v", data2)== Succeeded::yes,
	  "test writing Interfile image STIRtmp2");
    check(output_file_format.write_to_file("STIRtmp3.v", data3)== Succeeded::yes,
	  "test writing Interfile image STIRtmp3");
    
    set_tolerance(data1.find_max()/10000);
    
    // add
    if (run_stir_math("STIRtmpout.v STIRtmp1.hv STIRtmp2.hv STIRtmp3.hv"))
    {
      out_data_ptr = read_from_file<DiscretisedDensity<3,float> >("STIRtmpout.hv");
      calc_data = data1;
      calc_data += data2 + data3;
      
      check_if_equal( calc_data, *out_data_ptr,"test on adding");    
    }
    // mult
    if (run_stir_math("--mult STIRtmpout.v STIRtmp1.hv STIRtmp2.hv STIRtmp3.hv"))
    {
      out_data_ptr = read_from_file<DiscretisedDensity<3,float> >("STIRtmpout.hv");
      calc_data = data1;
      calc_data *= data2 * data3;
      
      check_if_equal( calc_data, *out_data_ptr,"test on multiplying");    
    }
    // add with power etc
    // range for rand() is 0 to RAND_MAX
    const float min_threshold = RAND_MAX/5.F;
    const float max_threshold = RAND_MAX/2.F;
    {
      char cmd_args[1000];
      sprintf(cmd_args, "--add-scalar 3.1 --power 2 --times-scalar 3.2 --min-threshold %g --max-threshold %g "
	      "STIRtmpout.v STIRtmp1.hv STIRtmp2.hv STIRtmp3.hv",
	      min_threshold, max_threshold);
      if (run_stir_math(cmd_args))
	{
	  out_data_ptr = read_from_file<DiscretisedDensity<3,float> >("STIRtmpout.hv");
	  calc_data.fill(0);
	  VoxelsOnCartesianGrid<float>  data2_thresholded(data2);
	  VoxelsOnCartesianGrid<float>  data3_thresholded(data3);
	  threshold_upper_lower( data2_thresholded.begin_all(),  data2_thresholded.end_all(),
				 min_threshold, max_threshold);
	  threshold_upper_lower( data3_thresholded.begin_all(),  data3_thresholded.end_all(),
				 min_threshold, max_threshold);
	  calc_data += data1 + (data2_thresholded*data2_thresholded)*3.2F + 3.1F + (data3_thresholded*data3_thresholded)*3.2F + 3.1F;
	  
	  check_if_equal( calc_data, *out_data_ptr,"test with thresholding, power and scalar multiplication and addition");    
	}
    }
    // add with power etc and including-first
    if (run_stir_math("--power 2 --times-scalar 3.2 --including-first STIRtmpout.v STIRtmp1.hv STIRtmp2.hv STIRtmp3.hv"))
    {
      out_data_ptr = read_from_file<DiscretisedDensity<3,float> >("STIRtmpout.hv");
      calc_data.fill(0);
      calc_data += (data1*data1 + data2*data2 + data3*data3)*3.2F;      
      check_if_equal( calc_data, *out_data_ptr,"test with power and scalar multiplication with --including-first");    
    }
    // add with power etc and accumulate 
    if (run_stir_math("--accumulate --power 2 --times-scalar 3.2  --add-scalar 3.1 --divide-scalar 2.1 --mult STIRtmp1.hv STIRtmp3.hv"))
    {
      calc_data.fill(0);
      calc_data += data1;
      calc_data += (data3*data3)*3.2F/2.1F + 3.1F;      
      out_data_ptr = read_from_file<DiscretisedDensity<3,float> >("STIRtmp1.hv");
      check_if_equal( calc_data, *out_data_ptr,"test with power and scalar multiplication, division and addition with --accumulate");    
    }
    // add with power etc and accumulate and including-first
    if (run_stir_math("--accumulate --power 2 --times-scalar 3.2 --including-first STIRtmp2.hv STIRtmp3.hv"))
    {
      calc_data.fill(0);
      calc_data += data2;
      calc_data *= calc_data * 3.2F;
      calc_data += (data3*data3)*3.2F;      
      out_data_ptr = read_from_file<DiscretisedDensity<3,float> >("STIRtmp2.hv");
      check_if_equal( calc_data, *out_data_ptr,"test with power and scalar multiplication with --including-first and --accumulate");    
    }  

    remove("STIRtmp1.ahv");
    remove("STIRtmp1.hv");
    remove("STIRtmp1.v");
    remove("STIRtmp2.ahv");
    remove("STIRtmp2.hv");
    remove("STIRtmp2.v");
    remove("STIRtmp3.ahv");
    remove("STIRtmp3.hv");
    remove("STIRtmp3.v");
    remove("STIRtmpout.ahv");
    remove("STIRtmpout.hv");
    remove("STIRtmpout.v");
  }

  // projdata
  {
    // to  keep testing code below as close as possible to the image case, we'll just 
    // take a single segment in the data.

    shared_ptr<Scanner> scanner_ptr(new Scanner(Scanner::E953));
    shared_ptr<ProjDataInfo> proj_data_info_ptr(
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr, 
			            /*span=*/1, 
                                    /*max_delta=*/0,
                                    /*num_views=*/8,
                                    /*num_tang_poss=*/16));
    SegmentByView<float> data1 = proj_data_info_ptr->get_empty_segment_by_view(0);
    SegmentByView<float> data2 = proj_data_info_ptr->get_empty_segment_by_view(0);
    SegmentByView<float> data3 = proj_data_info_ptr->get_empty_segment_by_view(0);
    SegmentByView<float> calc_data = proj_data_info_ptr->get_empty_segment_by_view(0);
    shared_ptr<ProjData> out_data_ptr;

    // fill with some random numbers
    srand(1);
    generate(data1.begin_all(), data1.end_all(), rand);
    generate(data2.begin_all(), data2.end_all(), rand);
    generate(data3.begin_all(), data3.end_all(), rand);
    
    // first create files and close them afterwards
    // This is necessary because system() requires all streams to be flushed.
    {
      shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
                                      
      ProjDataInterfile proj_data1(exam_info_sptr, proj_data_info_ptr, "STIRtmp1");
      ProjDataInterfile proj_data2(exam_info_sptr, proj_data_info_ptr, "STIRtmp2");
      ProjDataInterfile proj_data3(exam_info_sptr, proj_data_info_ptr, "STIRtmp3");
      
      proj_data1.set_segment(data1);
      proj_data2.set_segment(data2);
      proj_data3.set_segment(data3);
    
      set_tolerance(data1.find_max()/10000);
    }
    
    // add
    if (run_stir_math("-s STIRtmpout.hs STIRtmp1.hs STIRtmp2.hs STIRtmp3.hs"))
    {
      out_data_ptr = ProjData::read_from_file("STIRtmpout.hs");
      calc_data = data1;
      calc_data += data2 + data3;
      
      check_if_equal( calc_data, out_data_ptr->get_segment_by_view(0),
		      "test on adding");    
    }
    // mult
    if (run_stir_math("-s --mult STIRtmpout.hs STIRtmp1.hs STIRtmp2.hs STIRtmp3.hs"))
    {
      out_data_ptr = ProjData::read_from_file("STIRtmpout.hs");
      calc_data = data1;
      calc_data *= data2 * data3;
      
      check_if_equal( calc_data, out_data_ptr->get_segment_by_view(0),
		      "test on multiplying");    
    }
    // add with power etc
    // range for rand() is 0 to RAND_MAX
    const float min_threshold = RAND_MAX/5.F;
    const float max_threshold = RAND_MAX/2.F;
    {
      char cmd_args[1000];
      sprintf(cmd_args, "-s --add-scalar 3.1 --power 2 --times-scalar 3.2 --min-threshold %g --max-threshold %g "
	      "STIRtmpout.hs STIRtmp1.hs STIRtmp2.hs STIRtmp3.hs",
	      min_threshold, max_threshold);
      if (run_stir_math(cmd_args))
	{
	  out_data_ptr =  ProjData::read_from_file("STIRtmpout.hs");
	  calc_data.fill(0);
	  SegmentByView<float>  data2_thresholded(data2);
	  SegmentByView<float>  data3_thresholded(data3);
	  threshold_upper_lower( data2_thresholded.begin_all(),  data2_thresholded.end_all(),
				 min_threshold, max_threshold);
	  threshold_upper_lower( data3_thresholded.begin_all(),  data3_thresholded.end_all(),
				 min_threshold, max_threshold);
	  calc_data += data1 + (data2_thresholded*data2_thresholded)*3.2F + 3.1F + (data3_thresholded*data3_thresholded)*3.2F + 3.1F;
	  
	  check_if_equal( calc_data, out_data_ptr->get_segment_by_view(0),
			  "test with thresholding, power and scalar multiplication and addition");    
	}
    }

    // add with power etc and including-first
    if (run_stir_math("-s --power 2 --times-scalar 3.2 --including-first STIRtmpout.hs STIRtmp1.hs STIRtmp2.hs STIRtmp3.hs"))
    {
      out_data_ptr = ProjData::read_from_file("STIRtmpout.hs");
      calc_data.fill(0);
      calc_data += (data1*data1 + data2*data2 + data3*data3)*3.2F;      
      check_if_equal( calc_data, out_data_ptr->get_segment_by_view(0),
		      "test with power and scalar multiplication with --including-first");    
    }
    // add with power etc and accumulate 
    if (run_stir_math("-s --accumulate --power 2 --times-scalar 3.2 --divide-scalar 1.9 --add-scalar 3.1 --mult STIRtmp1.hs STIRtmp3.hs"))
    {
      calc_data.fill(0);
      calc_data += data1;
      calc_data += (data3*data3)*3.2F/1.9F+3.1F;      
      out_data_ptr = ProjData::read_from_file("STIRtmp1.hs");
      check_if_equal( calc_data, out_data_ptr->get_segment_by_view(0),
		      "test with power and scalar multiplication, division and addition with --accumulate");    
    }
    // add with power etc and accumulate and including-first
    if (run_stir_math("-s --accumulate --power 2 --times-scalar 3.2 --including-first STIRtmp2.hs STIRtmp3.hs"))
    {
      calc_data.fill(0);
      calc_data += data2;
      calc_data *= calc_data * 3.2F;
      calc_data += (data3*data3)*3.2F;      
      out_data_ptr = ProjData::read_from_file("STIRtmp2.hs");
      check_if_equal( calc_data, out_data_ptr->get_segment_by_view(0),
		      "test with power and scalar multiplication with --including-first and --accumulate");    
    }

    remove("STIRtmp1.hs");
    remove("STIRtmp1.s");
    remove("STIRtmp2.hs");
    remove("STIRtmp2.s");
    remove("STIRtmp3.hs");
    remove("STIRtmp3.s");
    remove("STIRtmpout.hs");
    remove("STIRtmpout.s");
  }

}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR


int main(int argc, char** argv)
{
  stir_mathTests tests(argc==2?argv[1] : "stir_math");
  tests.run_tests();
  return tests.main_return_value();
}
