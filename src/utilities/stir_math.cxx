//
// $Id$
//
/*
  Copyright (C) 2001- $Date$, Hammersmith Imanet Ltd
  This file is part of STIR.

  This file is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2.0 of the License, or
  (at your option) any later version.

  This file is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup utilities
  \brief add or multiply data, with some other basic math manipulations 

  This is a command line utility for adding, multiplying, thresholding ... data, 
  with a somewhat awkward syntax. 
  The command line arguments are as follows (but everything has to fit on 1 line):
  \code
  [--output-format parameter-filename ]
  [-s [--max_segment_num_to_process number] ] 
  [--add | --mult] 
  [--power power_float] 
  [--times-scalar mult_scalar_float] 
  [--divide-scalar divide_scalar_float] 
  [--add-scalar add_scalar_float] 
  [--min-threshold min_threshold]
  [--max-threshold max_threshold]
  [--including-first] 
  [--verbose]
  output_filename_with_extension in_data1 [in_data2 [in_data3...]]
  \endcode
  or
  \code
  [--output-format parameter-filename ]
  --accumulate
  [-s] 
  [--add | --mult] 
  [--power power_float] 
  [--times-scalar mult_scalar_float] 
  [--divide-scalar divide_scalar_float] 
  [--add-scalar add_scalar_float] 
  [--min-threshold min_threshold]
  [--max-threshold max_threshold]
  [--including-first] 
  [--verbose]
  out_and_input_filename in_data2 [in_data3 [in_data4...]]
  \endcode

  '--add' is default, and outputs the sum of the result of processed data.
  '--mult' outputs the multiplication of the result of processed data.<br>
  The '--include-first' option can be used such that power and scalar 
  multiplication are done on the first input argument as well. 
  Otherwise these manipulations are done only on the 2nd, 3rd,.. argument.<br>
  The '--accumulate' option can be used to say that the first filename given will be
  used for input AND output. Note that when using this option together with 
  '--including-first', the data in the
  first filename will first be manipulated according to '--power' and '--times-scalar'.<br>
  The '-s' option is necessary if the arguments are projection data.
  Otherwise, it is assumed the data are images.<br>
  For projection data, you can restrict the number of segments read/written
  using the --max_segment_num_to_process option (unless --accumulate is used).
  For example, using 2 as an argument of this option, will read/write
  segments -2,-1,0,1,2.<br>
  Multiple occurences of '--times-scalar' and '--divide-scalar' are
  allowed and will just result in accumulation of the factors.<P>
  The order of the manipulations is as follows:<br>
  (1) thresholding (2) power (3) scalar multiplication (4) scalar addition.

  The '--output-format' option can be used to write the output in 
  a different file format then the default (although this currently only
  works for images). The parameter file should have the following format:
  \verbatim
  output file format parameters:=
  output file format type:= <sometype>
  END:=
  \endverbatim
  See the stir::OutputFileFormat hierarchy for possible values.

  \par Examples
  <ul>
  <li> subtracting 2 files<br>
  \code stir_math --times-scalar -1 output in1 in2 \endcode
  <li> sum the square of each file<br>
  \code stir_math --power 2 --including-first output in1 in2 \endcode
  <li> Dividing 2 files after thresholding the 2nd file<br>
  \code stir_math --mult --power -1 --min-threshold .1 output in1 in2 \endcode
  <li> Dividing 2 files, with first file set to the quotient<br>
  \code stir_math --accumulate --mult --power -1 in1 in2 \endcode

  </ul>
  \warning There is no check that the data sizes and other info are compatible 
  and the output will have the largest data size in the input, 
  and the characteristics (like voxel-size or so) are taken from the first input data. 
  Hence, lots of funny effects can happen if data are not compatible.

  \warning When '--accumulate' is not used, the output file HAS to be different from all
  the input files.
  \warning The result of using non-integral powers on  negative numbers is probably 
  system-dependent.
  \warning For future compatibility, it is recommended to put the command line arguments
  in the order that they will be executed (i.e. as listed above). It might be 
  that we take the order into account in a future release.
  \author Kris Thielemans 

  $Date$
  $Revision$ 
*/

#include "stir/ArrayFunction.h"
#include "stir/DiscretisedDensity.h"
#include "stir/SegmentByView.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/Succeeded.h"
#include "stir/ProjDataInterfile.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"
#include "stir/NumericInfo.h"
#include "stir/KeyParser.h"
#include "stir/is_null_ptr.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/DynamicDiscretisedDensity.h"

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
#endif


USING_NAMESPACE_STIR

// a function object that takes a power of a float, and then multiplies with a float, and finally adds a float
class pow_times_add: public unary_function<float,float>
{
public:
  pow_times_add(const float add_scalar, const float mult_scalar, const float power, 
		const float min_threshold, const float max_threshold)
    : add(add_scalar), mult(mult_scalar), power(power), 
      min_threshold(min_threshold), max_threshold(max_threshold)
  {}

  float operator()(float const arg) const
  {
    const float value = min(max(arg, min_threshold), max_threshold);
    return add+mult*(power==1?value : pow(value,power));
  }
private:
  const float add;
  const float mult;
  const float power;
  const float min_threshold;
  const float max_threshold;
};

template <class DataT, class FunctionObjectT>
void process_data(const string& output_file_name,
		  const int num_files, char **argv, 
		  const bool no_math_on_data,
		  const bool except_first,
		  const bool verbose,
		  const bool do_add,
		  const FunctionObjectT& pow_times_add_object,
		  const OutputFileFormat<DataT>& output_format)
{
  std::auto_ptr< DataT >  image_ptr = 
    read_from_file<DataT>(*argv);
  if (!no_math_on_data && !except_first )
    in_place_apply_function(*image_ptr, pow_times_add_object);

  shared_ptr< DataT >  current_image_ptr;

  for (int i=1; i<num_files; ++i)
    {
      if (verbose)
	cout << "Reading image " << argv[i] << endl;
      current_image_ptr.reset(DataT::read_from_file(argv[i]));
      if (!no_math_on_data)
	in_place_apply_function(*current_image_ptr, pow_times_add_object);
      if (do_add)
	{
	  // TODO the next line doesn't work with some DataT, but its replacement is ugly!
	  // also, it would be better to be able to call += on each element
	  //*image_ptr += *current_image_ptr;
	  std::transform(image_ptr->begin_all(), image_ptr->end_all(),
			 current_image_ptr->begin_all(), 
			 image_ptr->begin_all(),
			 std::plus<float>());
	}
      else
	{
	  // *image_ptr *= *current_image_ptr;
	  std::transform(image_ptr->begin_all(), image_ptr->end_all(),
			 current_image_ptr->begin_all(), 
			 image_ptr->begin_all(),
			 std::multiplies<float>());
	}
    }

  if (verbose)
    cout << "Writing output image " << output_file_name << endl;
  output_format.write_to_file(output_file_name, *image_ptr);
}

template <class FunctionObjectT> //class DataT, for DynProjectionData ?
void process_data(const string& output_file_name,
		  const int num_files, char **argv, 
		  const bool no_math_on_data,
		  const bool except_first,
		  const bool verbose,
		  const bool do_add,
		  const FunctionObjectT& pow_times_add_object)
{
  shared_ptr<DynamicDiscretisedDensity> 
    dyn_image_sptr(DynamicDiscretisedDensity::read_from_file(*argv));
  DynamicDiscretisedDensity & dyn_image = *dyn_image_sptr;
  for(unsigned int frame_num=1;frame_num<=(dyn_image_sptr->get_time_frame_definitions()).get_num_frames();++frame_num)
    {
      if (!no_math_on_data && !except_first )
	in_place_apply_function(dyn_image[frame_num], pow_times_add_object);
    }
  shared_ptr<DynamicDiscretisedDensity> dyn_current_image_sptr;

  for (int i=1; i<num_files; ++i)
    {
      if (verbose)
	cout << "Reading image " << argv[i] << endl;
      dyn_current_image_sptr =
	read_from_file<DynamicDiscretisedDensity>(argv[i]);
      DynamicDiscretisedDensity & dyn_current_image = *dyn_current_image_sptr;
      for(unsigned int frame_num=1;frame_num<=(dyn_image_sptr->get_time_frame_definitions()).get_num_frames();++frame_num)
	{
	  if (!no_math_on_data)
	    in_place_apply_function(dyn_current_image[frame_num], pow_times_add_object);
	  if (do_add)
	    {
	      // TODO the next line doesn't work with some DataT, but its replacement is ugly!
	      // also, it would be better to be able to call += on each element
	      //*image_ptr += *current_image_ptr;
	      std::transform(dyn_image[frame_num].begin_all(), dyn_image[frame_num].end_all(),
			     dyn_current_image[frame_num].begin_all(), 
			     dyn_image[frame_num].begin_all(),
			     std::plus<float>());
	    }
	  else
	    {
	      // *image_ptr *= *current_image_ptr;
	      std::transform(dyn_image[frame_num].begin_all(), dyn_image[frame_num].end_all(),
			     dyn_current_image[frame_num].begin_all(), 
			     dyn_image[frame_num].begin_all(),
			     std::multiplies<float>());
	    }
	}
    }

  if (verbose)
    cout << "Writing output image " << output_file_name << endl;
  dyn_image_sptr->write_to_ecat7(output_file_name);
}

int 
main(int argc, char **argv)
{
  if(argc<3)
    {
      cerr<< "Usage: " << argv[0] << "\n\t"
	  << "[--output-format parameter-filename ]\n\t"
	  << "[--parametric || --dynamic]\n\t"
	  << "[-s [--max_segment_num_to_process number] ]\n\t"
	  << "[--accumulate] [--add | --mult]\n\t"
	  << "[--power power_float]\n\t"
	  << "[--times-scalar mult_scalar_float]\n\t"
	  << "[--divide-scalar div_scalar_float]\n\t"
	  << "[--add-scalar add_scalar_float]\n\t"
	  << "[--min-threshold min_threshold]\n\t"
	  << "[--max-threshold max_threshold]\n\t"
	  << "[--including-first] [--verbose]\n\t"
	  << "output_filename_with_extension in_data1 [in_data2 [in_data3...]]\n\n"
	  << "(but everything on 1 line).\n"
	  << "'--add' is default, and outputs the sum of the result of processed data.\n"
	  << "'--mult' outputs the multiplication of the result of processed data.\n"
	  << "The '--include-first' option can be used such that power and\n\t"
	  << "scalar multiplication are done on the first input argument\n\t"
	  << "as well. "
	  << "Otherwise these manipulations are done only on the 2nd, 3rd,.. argument.\n"
	  << "The '--accumulate' option can be used to say that the first filename given will be "
	"used for input AND output. Note that when using this option together with "
	"'--including-first', the data in the first filename will first be manipulated "
	"according to '--power', '--times-scalar' and '--divide-scalar'.\n\n"
	  << "Multiple occurences of '--times-scalar' and '--divide-scalar' are\n"
	  << "allowed and will just result in accumulation of the factors.\n\n"
	  << "The order of the manipulations is as follows:\n"
	  << "(1) thresholding (2) power (3) scalar multiplication (4) scalar addition.\n\n"
	  << "The '-s' option is necessary if the arguments are projection data."
	  << " Otherwise, it is assumed the data are images.\n\n"
	  << "For projection data, you can restrict the number of segments read/written\n"
	  << "using the --max_segment_num_to_process option (unless --accumulate is used).\n"
	  << "For example, using 2 as an argument of this option, will read/write"
	  << "segments -2,-1,0,1,2.\n\n"
	  << "WARNING: there is no check that the data sizes and other info are compatible "
	  << "and the output will have the largest data size in the input, "
	  << "and the characteristics (like voxel-size or so) are taken from the first input data. "
	  << "Hence, lots of funny effects can happen if data are not compatible.\n\n"
	  << "WARNING: For future compatibility, it is recommended to put \n"
	  << "the command line arguments in the order that they will be\n"
	  << "executed (i.e. as listed above). It might be that\n"
	  << "we take the order into account in a future release.\n";
      exit(EXIT_FAILURE);
    }
  const char * const program_name = argv[0];
  // skip program name
  --argc;
  ++argv;

  bool do_add = true;
  bool accumulate=false;
  float add_scalar = 0.F;
  float mult_scalar = 1.F;
  float min_threshold = NumericInfo<float>().min_value();
  float max_threshold = NumericInfo<float>().max_value();
  float power = 1;
  bool except_first = true;
  bool verbose = false;
  bool do_projdata = false;
  bool parametric = false;
  bool dynamic = false;

  int max_segment_num_to_process = -1;
  shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > >
    output_format_sptr =
    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr();


  // first process command line options

  while (argc>0 && argv[0][0]=='-')
    {
      if (strcmp(argv[0], "--max_segment_num_to_process")==0)
	{
	  if (argc<2)
	    { cerr << "Option '--max_segment_num_to_process' expects a (int) argument\n"; exit(EXIT_FAILURE); }
	  max_segment_num_to_process = atoi(argv[1]);
	  argc-=2; argv+=2;
	} 
      else if (strcmp(argv[0], "--output-format")==0)
	{
	  if (argc<2)
	    { 
	      cerr << "Option '--output-format' expects a (filename) argument\n"; 
	      exit(EXIT_FAILURE); 
	    }
	  KeyParser parser;
	  parser.add_start_key("output file format parameters");
	  parser.add_parsing_key("output file format type", &output_format_sptr);
	  parser.add_stop_key("END"); 
	  if (parser.parse(argv[1]) == false || is_null_ptr(output_format_sptr))
	    {
	      cerr << "Error parsing output file format from " << argv[1]<<endl;
	      exit(EXIT_FAILURE); 
	    }
	  argc-=2; argv+=2;
	} 
	
      else if (strcmp(argv[0], "--add-scalar")==0)
	{
	  if (argc<2)
	    { cerr << "Option '--add-scalar' expects a (float) argument\n"; exit(EXIT_FAILURE); }
	  add_scalar += static_cast<float>(atof(argv[1]));
	  argc-=2; argv+=2;
	} 
      else if (strcmp(argv[0], "--times-scalar")==0)
	{
	  if (argc<2)
	    { cerr << "Option '--times-scalar' expects a (float) argument\n"; exit(EXIT_FAILURE); }
	  mult_scalar *= static_cast<float>(atof(argv[1]));
	  argc-=2; argv+=2;
	} 
      else if (strcmp(argv[0], "--divide-scalar")==0)
	{
	  if (argc<2)
	    { cerr << "Option '--divide-scalar' expects a (float) argument\n"; exit(EXIT_FAILURE); }
	  mult_scalar /= static_cast<float>(atof(argv[1]));
	  argc-=2; argv+=2;
	} 
      else if (strcmp(argv[0], "--max-threshold")==0)
	{
	  if (argc<2)
	    { cerr << "Option '--max-threshold' expects a (float) argument\n"; exit(EXIT_FAILURE); }
	  max_threshold = static_cast<float>(atof(argv[1]));
	  argc-=2; argv+=2;
	} 
      else if (strcmp(argv[0], "--min-threshold")==0)
	{
	  if (argc<2)
	    { cerr << "Option '--min-threshold' expects a (float) argument\n"; exit(EXIT_FAILURE); }
	  min_threshold = static_cast<float>(atof(argv[1]));
	  argc-=2; argv+=2;
	} 
      else if (strcmp(argv[0], "--power")==0)
	{
	  if (argc<2)
	    { cerr << "Option '--power' expects an argument\n"; exit(EXIT_FAILURE); }
	  power = static_cast<float>(atof(argv[1]));
	  argc-=2; argv+=2;
	} 
      else  if (strcmp(argv[0], "--including-first")==0)
	{
	  except_first = false;
	  argc-=1; argv+=1;
	}
      else  if (strcmp(argv[0], "--verbose")==0)
	{
	  verbose = true;
	  argc-=1; argv+=1;
	}
      else  if (strcmp(argv[0], "-s")==0)
	{
	  do_projdata = true;
	  argc-=1; argv+=1;
	}
      else  if (strcmp(argv[0], "--parametric")==0)
	{
	  parametric = true;
	  argc-=1; argv+=1;
	}
      else  if (strcmp(argv[0], "--dynamic")==0)
	{
	  dynamic = true;
	  argc-=1; argv+=1;
	}
      else  if (strcmp(argv[0], "--add")==0)
	{
	  do_add = true;
	  argc-=1; argv+=1;
	}
      else  if (strcmp(argv[0], "--mult")==0)
	{
	  do_add = false;
	  argc-=1; argv+=1;
	}
      else  if (strcmp(argv[0], "--accumulate")==0)
	{
	  accumulate = true;
	  argc-=1; argv+=1;
	}
      else
	{ cerr << "Unknown option '" << argv[0] <<"'\n"; exit(EXIT_FAILURE); }
    }

  if (argc==0)
    { cerr << "No output file (nor input files) on command line\n"; exit(EXIT_FAILURE); }

  // find output filename
  const string output_file_name = *argv;
  if (!accumulate)
    {
      --argc; ++argv;
    }

  // some basic error checking and output
  const int num_files = argc;

  if (num_files==0)
    { cerr << "No input files on command line\n"; exit(EXIT_FAILURE); }

  const bool no_math_on_data = power==1 && mult_scalar==1 && add_scalar==0 &&
    min_threshold == NumericInfo<float>().min_value() &&
    max_threshold == NumericInfo<float>().max_value();


  if (verbose)
    {
      cout << program_name << ": "
	   << (do_add ? "adding " : "multiplying ")
	   << num_files;
      if (!no_math_on_data)
	cout <<" files after thresholding to a min value of "
	     << min_threshold << "\n and a max value of "
	     << max_threshold << "\n and then taking a power of "
	     << power << "\n and then multiplying with "
	     << mult_scalar << "\n and then adding  "
	     << add_scalar
	     << (except_first?"\n except for" : " including")
	     <<" the first file";
      cout << endl;
    }

  // construct function object that does the manipulations on each data
  pow_times_add pow_times_add_object(add_scalar, mult_scalar, power, min_threshold, max_threshold);

  // start the main processing
  if (!do_projdata)
    {
 
      if (!parametric && !dynamic)
	{
	  process_data(output_file_name,
		       num_files, argv, 
		       no_math_on_data,
		       except_first,
		       verbose,
		       do_add,
		       pow_times_add_object,
		       *output_format_sptr);
	}
      else if (parametric)
	{
	  shared_ptr<OutputFileFormat<ParametricVoxelsOnCartesianGrid > >
	    output_format_sptr =
	    OutputFileFormat<ParametricVoxelsOnCartesianGrid >::default_sptr();	  

	  process_data(output_file_name,
		       num_files, argv, 
		       no_math_on_data,
		       except_first,
		       verbose,
		       do_add,
		       pow_times_add_object,
		       *output_format_sptr);
	}
      else if (dynamic)
	{	  
	  process_data(output_file_name,
		       num_files, argv, 
		       no_math_on_data,
		       except_first,
		       verbose,
		       do_add,
		       pow_times_add_object);
	}
    }
  else // do_projdata
    {
      vector< shared_ptr<ProjData> > all_proj_data(num_files);
      shared_ptr<ProjData> out_proj_data_ptr;
      if (accumulate)
	{
	  all_proj_data[0] = ProjData::read_from_file(argv[0], ios::in | ios::out);
	  out_proj_data_ptr = all_proj_data[0];

	  if (max_segment_num_to_process>=0)
	    warning("Parameter max_segment_num_to_process will be ignored.");
	}
      else
	{
	  all_proj_data[0] = ProjData::read_from_file(argv[0]);
	  shared_ptr<ProjDataInfo> 
	    output_proj_data_info_sptr((*all_proj_data[0]).get_proj_data_info_ptr()->clone());
	  if (max_segment_num_to_process>=0)
	    {
	      output_proj_data_info_sptr->
		reduce_segment_range(-max_segment_num_to_process,
				     max_segment_num_to_process);
	    }
	  out_proj_data_ptr.reset(new ProjDataInterfile(output_proj_data_info_sptr, 
							output_file_name));        
	}

      // read rest of projection data headers
      for (int i=1; i<num_files; ++i)
	all_proj_data[i] =  ProjData::read_from_file(argv[i]); 

      // do reading/writing in a loop over segments
      for (int segment_num = out_proj_data_ptr->get_min_segment_num();
	   segment_num <= out_proj_data_ptr->get_max_segment_num();
	   ++segment_num)
	{   
	  if (verbose)
	    cout << "Processing segment num " << segment_num << " for all files" << endl;
	  SegmentByView<float> segment_by_view = 
	    (*all_proj_data[0]).get_segment_by_view(segment_num);
	  if (!no_math_on_data && !except_first )
	    in_place_apply_function(segment_by_view, pow_times_add_object);
	  for (int i=1; i<num_files; ++i)
	    {
	      SegmentByView<float> current_segment_by_view = 
		(*all_proj_data[i]).get_segment_by_view(segment_num);
	      if (!no_math_on_data)
		in_place_apply_function(current_segment_by_view, pow_times_add_object);
	      if(do_add)
		segment_by_view += current_segment_by_view;
	      else
		segment_by_view *= current_segment_by_view;
	    }
    
	  if (!(out_proj_data_ptr->set_segment(segment_by_view) == Succeeded::yes))
	    warning("Error set_segment %d\n", segment_num);   
	}
    } 
  return EXIT_SUCCESS;
}
