//
//
/*!

  \file
  \ingroup IO

  \brief A simple program to test the stir::OutputFileFormat function.

  \author Kris Thielemans
  \author Richard Brown



  To run the test, you should use a command line argument with the name of a file.
  This should contain a test par file.
  See stir::OutputFileFormatTests class documentation for file contents.

  \warning Overwrites files STIRtmp.* in the current directory

  \todo The current implementation requires that the output file format as also
  readable by stir::read_from_file. At least we should provide a
  run-time switch to not run that part of the tests.
*/
/*
    Copyright (C) 2002- 2011, Hammersmith Imanet Ltd

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

#include "stir/IO/test/test_IO.h"
#include "stir/DynamicDiscretisedDensity.h"

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
class IOTests_DynamicDiscretisedDensity : public IOTests<DynamicDiscretisedDensity>
{
public:
    explicit IOTests_DynamicDiscretisedDensity(istream& in) : IOTests(in) {}

protected:

    void create_image();
    void check_result();
};

void IOTests_DynamicDiscretisedDensity::create_image()
{
    // Create time definitions
    std::vector<double> starts, durations;
    starts.push_back(1);
    starts.push_back(2);
    durations.push_back(1);
    durations.push_back(1);
    TimeFrameDefinitions time_defs(starts,durations);

     // Scan time somewhere in June 2010...
    const double scan_start = double(1277478034);

    // Create a scanner (any will do)
    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::Advance));

    _image_to_write_sptr.reset(new DynamicDiscretisedDensity(time_defs,scan_start,scanner_sptr,_single_image_sptr));
    _image_to_write_sptr->set_density_sptr(_single_image_sptr,1);
    _image_to_write_sptr->set_density_sptr(_single_image_sptr,2);
}

void IOTests_DynamicDiscretisedDensity::check_result()
{
    set_tolerance(.00001);

    check_if_equal(_image_to_read_sptr->get_densities().size(),_image_to_write_sptr->get_densities().size(), "test number of dynamic images");

    for (int i=1; i<=_image_to_read_sptr->get_densities().size(); i++) {

        // Cast the discretised density to voxels on cartesian grids to check grid spacing
        VoxelsOnCartesianGrid<float> *image_to_write_ptr = dynamic_cast<VoxelsOnCartesianGrid<float> *>(&_image_to_write_sptr->get_density(i));
        VoxelsOnCartesianGrid<float> *image_to_read_ptr  = dynamic_cast<VoxelsOnCartesianGrid<float> *>(&_image_to_read_sptr->get_density(i));

        if (image_to_write_ptr==0 || image_to_read_ptr==0) {
            everything_ok = false;
            return;
        }

        compare_images(*image_to_write_ptr, *image_to_read_ptr);
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

  IOTests_DynamicDiscretisedDensity tests(in);
  tests.run_tests();
  return tests.main_return_value();
}
