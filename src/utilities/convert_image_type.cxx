//
//
/*
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
 
  \brief This program converts an image into another file type.

  \author Richard Brown

*/
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/read_from_file.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/DiscretisedDensity.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"

USING_NAMESPACE_STIR

template<typename STIRImageType>
static
shared_ptr<STIRImageType> read_image(const std::string &filename)
{
    if (filename.empty())
        error("Input filename is blank.");
    shared_ptr<STIRImageType> output(read_from_file<STIRImageType>(filename));
    if (is_null_ptr(output))
        error("convert_image_type: Failed reading input image.");
    return output;
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
void write_image(shared_ptr<STIRImageType> image, const std::string &filename, const std::string &par)
{
    shared_ptr<OutputFileFormat<STIRImageType> > output_file_format =
            set_up_output_format<STIRImageType>(par);

    if (output_file_format->write_to_file(filename,*image) == Succeeded::no)
        error("postfilter: Saving image failed.");
}

int main(int argc, char **argv)
{
    if(argc<4 || argc>5) {
        std::cerr<<"Usage: " << argv[0] << " <output filename> <input filename> <output file format parameters> [--parametric | --dynamic]\n";
        exit(EXIT_FAILURE);
    }

    // Type of image
    enum ImageType { normal, dynamic, parametric };
    ImageType image_type = normal;
    if (argc == 5) {
        if      (strcmp(argv[4], "--dynamic") == 0)    image_type = dynamic;
        else if (strcmp(argv[4], "--parametric") == 0) image_type = parametric;
    }

    const std::string output_filename = argv[1];
    const std::string input_filename  = argv[2];
    const std::string output_params   = argv[3];

    Succeeded success = Succeeded::no;

    // Normal
    if      (image_type == normal) {
      shared_ptr<DiscretisedDensity<3,float> > im = read_image<DiscretisedDensity<3,float> >(input_filename);
      write_image(im, output_filename, output_params);
    }
    else if (image_type == dynamic) {
        shared_ptr<DynamicDiscretisedDensity> im = read_image<DynamicDiscretisedDensity>(input_filename);
        write_image(im, output_filename, output_params);
    }
    else if (image_type == parametric) {
        shared_ptr<ParametricVoxelsOnCartesianGrid> im = read_image<ParametricVoxelsOnCartesianGrid>(input_filename);
        write_image(im, output_filename, output_params);
    }
}
