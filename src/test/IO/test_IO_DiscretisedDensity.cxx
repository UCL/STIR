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
class IOTests_DiscretisedDensity : public IOTests<DiscretisedDensity<3,float> >
{
public:
    explicit IOTests_DiscretisedDensity(istream& in) : IOTests(in) {}

protected:

    void create_image();
    void read_image();
    void check_result();
};

void IOTests_DiscretisedDensity::create_image()
{
    _image_to_write_sptr = create_single_image();
}

void IOTests_DiscretisedDensity::read_image()
{
    // now read it back
    unique_ptr<DiscretisedDensity<3,float> >
        density_ptr = read_from_file<DiscretisedDensity<3,float> >(_filename);

    if(!check(!is_null_ptr(density_ptr), "failed reading"))
        return;

    _image_to_read_sptr.reset(
                new VoxelsOnCartesianGrid<float>(
                    *dynamic_cast< VoxelsOnCartesianGrid<float> *>(density_ptr.get())));

    if(!check(!is_null_ptr(_image_to_read_sptr), "failed reading"))
        return;
}

void IOTests_DiscretisedDensity::check_result()
{
    // Cast the discretised density to voxels on cartesian grids to check grid spacing
    VoxelsOnCartesianGrid<float> *image_to_write_ptr = dynamic_cast<VoxelsOnCartesianGrid<float> *>(_image_to_write_sptr.get());
    VoxelsOnCartesianGrid<float> *image_to_read_ptr  = dynamic_cast<VoxelsOnCartesianGrid<float> *>(_image_to_read_sptr.get());

    compare_images(*image_to_write_ptr, *image_to_read_ptr);

    // Check TimeFrameDefinitions in ExamInfo. Not all formats support this. Skip if ITK
    if (_output_file_format_sptr->get_registered_name() != "ITK")
        check_exam_info(image_to_write_ptr->get_exam_info(), image_to_read_ptr->get_exam_info());
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

  IOTests_DiscretisedDensity tests(in);
  tests.run_tests();
  return tests.main_return_value();
}
