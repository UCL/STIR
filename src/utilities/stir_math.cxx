//
// $Id$
//

/*!
\file
\ingroup utilities
\brief add or multiply data, with some other basic math manipulations 

This is a command line utility for adding or multiplying data, with a
somewhat awkward syntax. 
The command line arguments are as follows (but everything has to fit on 1 line):
\code
  [-s] 
  [--add | --mult] 
  [--power power_float] 
  [--times-scalar mult_scalar_float] 
  [--divide-scalar divide_scalar_float] 
  [--add-scalar add_scalar_float] 
  [--including-first] 
  [--verbose]
  output_filename_with_extension in_data1 [in_data2 [in_data3...]]
\endcode
or
\code
  --accumulate
  [-s] 
  [--add | --mult] 
  [--power power_float] 
  [--times-scalar mult_scalar_float] 
  [--divide-scalar divide_scalar_float] 
  [--add-scalar add_scalar_float] 
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
Multiple occurences of '--times-scalar' and '--divide-scalar' are
allowed and will just result in accumulation of the factors.<br>
Power is taken before multiplication with the scalar, which is done before addition of the scalar.<br>
\par Examples
<ul>
<li> subtracting 2 files<br>
\code stir_math --times-scalar -1 output in1 in2 \endcode
<li> sum the square of each file<br>
\code stir_math --power 2 --including-first output in1 in2 \endcode
<li> Dividing 2 files<br>
\code stir_math --mult --power -1 output in1 in2 \endcode
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
\todo allow different output file formats, currently uses DefaultOutputFileFormat
\author Kris Thielemans 

$Date$
$Revision$ 
*/
/*
    Copyright (C) 2001- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataFromStream.h"
#include "stir/DiscretisedDensity.h"
#include "stir/SegmentByView.h"
#include "stir/IO/DefaultOutputFileFormat.h"
#include "stir/Succeeded.h"
#include "stir/ProjDataInterfile.h"
#include "stir/utilities.h"
#include "stir/ArrayFunction.h"
#include "stir/Succeeded.h"

#include <fstream> 
#include <iostream> 
#include <functional>
#include <algorithm>
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::cout;
using std::endl;
using std::fstream;
using std::unary_function;
using std::transform;
#endif

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
#define	in_place_apply_function(A, B) \
        transform((A).begin_all(), (A).end_all(), \
                  (A).begin_all(), pow_times_object)
#endif


USING_NAMESPACE_STIR

// a function object that takes a power of a float, and then multiplies with a float, and finally adds a float
class pow_times_add: public unary_function<float,float>
{
public:
  pow_times_add(const float add_scalar, const float mult_scalar, const float power)
    : add(add_scalar), mult(mult_scalar), power(power)
  {}

  float operator()(float const arg) const
  {
    return add+mult*(power==1?arg : pow(arg,power));
  }
private:
  const float add;
  const float mult;
  const float power;
};

int 
main(int argc, char **argv)
{
  if(argc<4)
  {
    cerr<< "Usage: " << argv[0] << "\n\t"
	<< "[-s] [--accumulate] [--add | --mult]\n\t"
	<< "[--power power_float]\n\t"
	<< "[--times-scalar mult_scalar_float]\n\t"
	<< "[--divide-scalar div_scalar_float]\n\t"
	<< "[--add-scalar add_scalar_float]\n\t"
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
           "according to '--power', '--times-scalar' and '--divide-scalar'.\n"
	<< "Multiple occurences of '--times-scalar' and '--divide-scalar' are\n"
	<< "allowed and will just result in accumulation of the factors.\n"
        << "Power is taken before multiplication with the scalar, which is\n"
	<< "done before addition of the scalar.\n\n"
	<< "The '-s' option is necessary if the arguments are projection data."
	<< " Otherwise, it is assumed the data are images.\n\n"
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
  float power = 1;
  bool except_first = true;
  bool verbose = false;
  bool do_projdata = false;


  // first process command line options

  while (argc>0 && argv[0][0]=='-')
  {
    if (strcmp(argv[0], "--add-scalar")==0)
    {
      if (argc<2)
      { cerr << "Option '--add-scalar' expects a (float) argument\n"; exit(EXIT_FAILURE); }
      add_scalar += atof(argv[1]);
      argc-=2; argv+=2;
    } else
    if (strcmp(argv[0], "--times-scalar")==0)
    {
      if (argc<2)
      { cerr << "Option '--times-scalar' expects a (float) argument\n"; exit(EXIT_FAILURE); }
      mult_scalar *= atof(argv[1]);
      argc-=2; argv+=2;
    } else
    if (strcmp(argv[0], "--divide-scalar")==0)
    {
      if (argc<2)
      { cerr << "Option '--divide-scalar' expects a (float) argument\n"; exit(EXIT_FAILURE); }
      mult_scalar /= atof(argv[1]);
      argc-=2; argv+=2;
    } 
    else if (strcmp(argv[0], "--power")==0)
    {
      if (argc<2)
      { cerr << "Option '--power' expects an argument\n"; exit(EXIT_FAILURE); }
      power = atof(argv[1]);
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

  const bool no_math_on_data = power==1 && mult_scalar==1 && add_scalar==0;

  if (verbose)
    {
      cout << program_name << ": "
	   << (do_add ? "adding " : "multiplying ")
	   << num_files;
      if (!no_math_on_data)
	cout <<" files after taking a power of "
	     << power << "\n and then multiplying with "
	     << mult_scalar << "\n and then adding  "
	     << add_scalar
	     << (except_first?"\n except for" : " including")
	     <<" the first file";
      cout << endl;
    }

  // construct function object that does the manipulations on each data
  pow_times_add pow_times_add_object(add_scalar, mult_scalar, power);

  // start the main processing
  if (!do_projdata)
    {
 
      shared_ptr< DiscretisedDensity<3,float> >  image_ptr = 
	DiscretisedDensity<3,float>::read_from_file(*argv);
      if (!no_math_on_data && !except_first )
	in_place_apply_function(*image_ptr, pow_times_add_object);

      shared_ptr< DiscretisedDensity<3,float> >  current_image_ptr;

      for (int i=1; i<num_files; ++i)
	{
	  if (verbose)
	    cout << "Reading image " << argv[i] << endl;
	  current_image_ptr = 
	    DiscretisedDensity<3,float>::read_from_file(argv[i]);
	  if (!no_math_on_data)
	    in_place_apply_function(*current_image_ptr, pow_times_add_object);
	  if (do_add)
	    *image_ptr += *current_image_ptr;
	  else
	    *image_ptr *= *current_image_ptr;
	}

      if (verbose)
	cout << "Writing output image " << output_file_name << endl;
      DefaultOutputFileFormat output_format;
      output_format.write_to_file(output_file_name, *image_ptr);
    }
  else // do_projdata
    {
      vector< shared_ptr<ProjData> > all_proj_data(num_files);
      shared_ptr<ProjData> out_proj_data_ptr;
      if (accumulate)
      {
        all_proj_data[0] = ProjData::read_from_file(argv[0], ios::in | ios::out);
        out_proj_data_ptr = all_proj_data[0];
      }
      else
      {
        all_proj_data[0] = ProjData::read_from_file(argv[0]);

        out_proj_data_ptr =
          new ProjDataInterfile((*all_proj_data[0]).get_proj_data_info_ptr()->clone(), 
                                output_file_name);        
      }

      // read rest of projection data headers
      for (int i=1; i<num_files; ++i)
	all_proj_data[i] =  ProjData::read_from_file(argv[i]); 

      // do reading/writing in a loop over segments
      for (int segment_num = (*all_proj_data[0]).get_min_segment_num();
	   segment_num <= (*all_proj_data[0]).get_max_segment_num();
	   ++segment_num)
	{   
	  if (verbose)
	    cout << "Processing segment num " << segment_num << "for all files" << endl;
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
