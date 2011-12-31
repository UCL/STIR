//
// $Id$
//
/*!

  \file
  \ingroup test

  \brief A simple program to test the stir::OutputFileFormat function.

  \author Kris Thielemans

  $Date$
  $Revision$

  
  To run the test, you should use a command line argument with the name of a file.
  This should contain a test par file.
  See stir::OutputFileFormatTests class documentation for file contents.

  \warning Overwrites files STIRtmp.* in the current directory

  \todo The current implementation requires that the output file format as also
  readable by stir::read_from_file. At least we should provide a
  run-time switch to not run that part of the tests.
*/
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd

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
  
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#ifdef HAVE_LLN_MATRIX
#include "stir/IO/ECAT6OutputFileFormat.h" // need this for test on pixel_size
#endif
#include "stir/RunTests.h"
#include "stir/KeyParser.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/IndexRange3D.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <math.h>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ifstream;
using std::istream;
#endif

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief A simple class to test the OutputFileFormat function.

  The class reads input from a stream, whose contents should be as
  follows:

  \verbatim
  Test OutputFileFormat Parameters:=
  output file format type := 
  ; here are parameters specific for the file format
  End:=
  \endverbatim

  \warning Overwrites files STIRtmp.* in the current directory
  \todo Delete STIRtmp.* files, but that's a bit difficult as we don't know which ones 
  are written.
*/
class OutputFileFormatTests : public RunTests
{
public:
  OutputFileFormatTests(istream& in) ;

  void run_tests();
private:
  istream& in;
  shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > > output_file_format_ptr;
  KeyParser parser;
};

OutputFileFormatTests::
OutputFileFormatTests(istream& in) :
  in(in)
{
  output_file_format_ptr.reset();
  parser.add_start_key("Test OutputFileFormat Parameters");
  parser.add_parsing_key("output file format type", &output_file_format_ptr);
  parser.add_stop_key("END");
}

void OutputFileFormatTests::run_tests()
{  
  cerr << "Testing OutputFileFormat parsing function..." << endl;
  cerr << "WARNING: will overwite files called STIRtmp*\n";

  if (!check(parser.parse(in), "parsing failed"))
    return;

  if (!check(!is_null_ptr(output_file_format_ptr), 
        "parsing failed to set output_file_format_ptr"))
    return;
#if 0 
  cerr << "Output parameters after reading from input file:\n"
       << "-------------------------------------------\n";
  cerr << static_cast<ParsingObject&>(*output_file_format_ptr).parameter_info();

  cerr << "-------------------------------------------\n\n";
#endif
  cerr << "Now writing to file and reading it back." << endl; 
  // construct density and write to file
  {
#ifdef HAVE_LLN_MATRIX
    USING_NAMESPACE_ECAT
    USING_NAMESPACE_ECAT6
    // TODO get next info from OutputFileFormat class instead of hard-wiring
    // this in here
    const bool supports_different_xy_pixel_sizes =
      dynamic_cast<ECAT6OutputFileFormat const * const>(output_file_format_ptr.get()) == 0
      ? true : false;
    const bool supports_origin_z_shift =
      dynamic_cast<ECAT6OutputFileFormat const * const>(output_file_format_ptr.get()) == 0
      ? true : false;
    const bool supports_origin_xy_shift =
      true;
#else
    const bool supports_different_xy_pixel_sizes = true;
    const bool supports_origin_z_shift = true;
    const bool supports_origin_xy_shift =
      true;
#endif

    CartesianCoordinate3D<float> origin (0.F,0.F,0.F);
    if (supports_origin_xy_shift)
      {  origin.x()=2.4F; origin.y() = -3.5F; }
    if (supports_origin_z_shift)
      {  origin.z()=6.4F; }

    CartesianCoordinate3D<float> grid_spacing (3.F,4.F,supports_different_xy_pixel_sizes?5.F:4.F); 
  
    IndexRange<3> 
      range(CartesianCoordinate3D<int>(0,-15,-14),
	    CartesianCoordinate3D<int>(4,14,14));
	    
    VoxelsOnCartesianGrid<float>  image(range,origin, grid_spacing);
    {
      // fill with some data
      for (int z=image.get_min_z(); z<=image.get_max_z(); ++z)
	for (int y=image.get_min_y(); y<=image.get_max_y(); ++y)
	  for (int x=image.get_min_x(); x<=image.get_max_x(); ++x)
	    image[z][y][x]=
	      300*sin(static_cast<float>(x*_PI)/image.get_max_x())
	      *sin(static_cast<float>(y+10*_PI)/image.get_max_y())
	      *cos(static_cast<float>(z*_PI/3)/image.get_max_z());
    }

    // write to file

    string filename = "STIRtmp";
    const Succeeded success =
      output_file_format_ptr->write_to_file(filename,image);
    
    if (check( success==Succeeded::yes, "test writing to file"))
      {

	// now read it back
	
	std::auto_ptr<DiscretisedDensity<3,float> >
	  density_ptr = read_from_file<DiscretisedDensity<3,float> >(filename);
	
	const  VoxelsOnCartesianGrid<float> * image_as_read_ptr =
	  dynamic_cast< VoxelsOnCartesianGrid<float> const *>
	  (density_ptr.get());

	set_tolerance(.00001);
	if (check(!is_null_ptr(image_as_read_ptr), "test on image type read back from file"))
	  {
	    check_if_equal(image_as_read_ptr->get_grid_spacing(), grid_spacing, "test on grid spacing read back from file");
	    

	    if (output_file_format_ptr->get_type_of_numbers().integer_type())
	      {
		set_tolerance(image.find_max()/
			      pow(2.,static_cast<double>(output_file_format_ptr->get_type_of_numbers().size_in_bits())));
	      }

	    check_if_equal(image, *density_ptr, "test on data read back from file");
	    set_tolerance(.00001);
	    check_if_equal(density_ptr->get_origin(), origin, "test on origin read back from file");
	  }
      }
    if (is_everything_ok())
      {
	
      }
    else
	cerr << "You can check what was written in STIRtmp.*\n";

  }
    


}


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    cerr << "Usage : " << argv[0] << " filename\n"
         << "See source file for the format of this file.\n\n";
    return EXIT_FAILURE;
  }


  ifstream in(argv[1]);
  if (!in)
  {
    cerr << argv[0] 
         << ": Error opening input file " << argv[1] << "\nExiting.\n";

    return EXIT_FAILURE;
  }

  OutputFileFormatTests tests(in);
  tests.run_tests();
  return tests.main_return_value();
}
