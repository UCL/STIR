//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2018, University College London
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

  \brief  This program performs filtering on image data
 
  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Matthew Jacobson
  \author PARAPET project
  \author Richard Brown
  

  This program enables calling any ImageProcessor object on input data, 
  and writing it to file. It can take the following command line:
  \verbatim
  spostfilter [--verbose] [--dynamic/--parametric] <output filename> <input header filename> <filter .par filename> [output_format_par_file]
  \endverbatim
  This is done to make it easy to process a lot of files with the same 
  ImageProcessor. However, if the number of command line arguments is not 
  correct, appropriate questions will be asked interactively.

  If the <tt>--verbose</tt> option is used, the filter-parameters that are going to be
  used will be written to stdout. This is useful for checking/debugging.

  \par Example .par file
  (but see stir::MedianImageFilter3D to see if the following example is still correct)
  \verbatim
  PostFilteringParameters :=
  Postfilter type :=Median   
  Median Filter Parameters :=
  mask radius x := 1   
  mask radius y := 2
  mask radius z := 3
  End Median Filter Parameters:=
  End PostFiltering Parameters:=
  \endverbatim

  An optional output file format parameter file can also be given. An example for this might be:
    output file format parameters :=
    output file format type := Interfile
    interfile Output File Format Parameters:=
    number format := float
    number_of_bytes_per_pixel:=4
    End Interfile Output File Format Parameters:=
    end :=

*/

#include "stir/PostFiltering.h"
#include "stir/utilities.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/Succeeded.h"
#include <iostream> 

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_STIR

template<typename STIRImageType>
STIRImageType * ask_image(const char *const input_query)
{
  char filename[max_filename_length];
  ask_filename_with_extension(filename, 
				input_query,
				"");
  
  return STIRImageType::read_from_file(filename);
}
  
template<typename STIRImageType>
shared_ptr<OutputFileFormat<STIRImageType> > set_up_output_format(const std::string &filename)
{
    shared_ptr<OutputFileFormat<STIRImageType> > output =
            OutputFileFormat<STIRImageType>::default_sptr();
     if (filename.size() != 0) {
         KeyParser parser;
         parser.add_start_key("output file format parameters");
         parser.add_parsing_key("output file format type", &output);
         parser.add_stop_key("END");
         if (parser.parse(filename.c_str()) == false || is_null_ptr(output)) {
            warning("Error parsing output file format. Using default format.");
            output = OutputFileFormat<STIRImageType>::default_sptr();
        }
    }
    return output;
}

template<typename STIRImageType>
static
shared_ptr<STIRImageType> read_image(const std::string &filename)
{
    shared_ptr<STIRImageType> output;
    if (!filename.empty())
        output = read_from_file<STIRImageType>(filename);
    else
        output.reset(ask_image<STIRImageType>("Image to process?"));
    if (is_null_ptr(output))
        error("postfilter: No input image. Not writing any output.\n");

    return output;
}

template<typename STIRImageType>
static
void save_image(shared_ptr<STIRImageType> image, const std::string &filename, const std::string &par)
{
    shared_ptr<OutputFileFormat<STIRImageType> > output_file_format =
            set_up_output_format<STIRImageType>(par);

    if (output_file_format->write_to_file(filename,*image) == Succeeded::no)
        error("postfilter: Saving image failed.\n");
}


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

static void
print_usage()
{
  cerr<<"\nUsage: postfilter [--verbose] [--dynamic/--parametric] <output filename> <input header filename> <filter .par filename> [output_format_par_file]\n"<<endl;
}

int
main(int argc, char *argv[])
{
  enum ImageType { normal, dynamic, parametric };
  ImageType image_type = normal;

  shared_ptr<DiscretisedDensity<3,float> > input_image_single_ptr;
  shared_ptr<DynamicDiscretisedDensity>    input_image_dynamic_ptr;
  shared_ptr<ParametricVoxelsOnCartesianGrid> input_image_parametric_ptr;
  PostFiltering<DiscretisedDensity<3,float> > post_filtering;
  std::string out_filename, output_file_format_par = "", input_filename = "";
  bool verbose = false;

  // option processing
  while (argc>1 && argv[1][0]=='-')
    {
      if (strcmp(argv[1], "--verbose") == 0)
	{
	  verbose = true;
	  --argc; ++argv;
	}
      else if (strcmp(argv[1], "--dynamic") == 0)
    {
      image_type = dynamic;
      --argc; ++argv;
    }
      else if (strcmp(argv[1], "--parametric") == 0)
    {
      image_type = parametric;
      --argc; ++argv;
    }
      else
	{
	  print_usage();
	  return EXIT_FAILURE;
	}
    }
  if (argc<5 || argc>6)
    {
      print_usage();
    }
  if (argc>1)
    {
      out_filename = argv[1];
    }
  else
    {
      char outfile[max_filename_length];
      ask_filename_with_extension(outfile,
				  "Output to which file: ", "");
      out_filename = outfile;
    }
  if (argc>2)
    {
      input_filename = argv[2];
    }

  if (argc>3)
    {
      if (post_filtering.parse(argv[3]) == false)
	{
	  warning("postfilter aborting because error in parsing. Not writing any output");
	  return EXIT_FAILURE;
	}
    }
  else
    {     
      cerr << "\nI'm going to ask you for the type of filter (or image processor)\n"
	"Possible values:\n";
      DataProcessor<DiscretisedDensity<3,float> >::list_registered_names(cerr);
      
      post_filtering.ask_parameters();
    }
  if (argc>4)
    {
      output_file_format_par = argv[4];
    }

  if (post_filtering.is_filter_null())
    {
      warning("postfilter: No filter set. Not writing any output.\n");
      return EXIT_FAILURE;
    }

  // Read image
  if (image_type == normal)
    input_image_single_ptr =
            read_image<DiscretisedDensity<3,float> >(input_filename);
  else if (image_type == dynamic)
      input_image_dynamic_ptr =
              read_image<DynamicDiscretisedDensity>(input_filename);
  else /*if (image_type == parametric)*/
      input_image_parametric_ptr =
              read_image<ParametricVoxelsOnCartesianGrid>(input_filename);

  if (verbose)
    {
      cerr << "PostFilteringParameters:\n" << post_filtering.parameter_info();
    }

  // Post filter!
  stir::Succeeded success(Succeeded::yes);
  if (image_type == normal)
      success = post_filtering.process_data(*input_image_single_ptr);
  else if (image_type == dynamic) {
      for (unsigned i=1; i<=input_image_dynamic_ptr->get_num_time_frames(); i++) {
          if (post_filtering.process_data(input_image_dynamic_ptr->get_density(i)) == Succeeded::no) {
              success = Succeeded::no;
              break;
          }
      }
  }
  else /*if (image_type == parametric)*/ {
      for (unsigned i=1; i<=input_image_parametric_ptr->get_num_params(); i++) {
          VoxelsOnCartesianGrid<float> single_parametric_param =
                  input_image_parametric_ptr->construct_single_density(int(i));
          if (post_filtering.process_data(single_parametric_param) == Succeeded::no) {
              success = Succeeded::no;
              break;
          }
          input_image_parametric_ptr->update_parametric_image(single_parametric_param,i);
      }
  }
  if (success == Succeeded::no)
      error("Postfiltering failed.");

  // Save to file
  if (image_type == normal)
    save_image(input_image_single_ptr, out_filename, output_file_format_par);
  else if (image_type == dynamic)
    save_image(input_image_dynamic_ptr, out_filename, output_file_format_par);
  else /*if (image_type == parametric)*/
    save_image(input_image_parametric_ptr, out_filename, output_file_format_par);

  return EXIT_SUCCESS;
}


