//
//
/*!

  \file
  \ingroup test

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
    Copyright (C) 2018-, University College London

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
using std::string;
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
class IOTests_ITKMulticomponent : public RunTests
{
public:
    IOTests_ITKMulticomponent(string multi)
    { _multi = multi; }

    void run_tests();

protected:

    string _multi;
};
void IOTests_ITKMulticomponent::run_tests()
{
    typedef VoxelsOnCartesianGrid<CartesianCoordinate3D<float> > VoxelsCoords;

    // Read the files
    std::cerr << "\nReading: " << _multi << "\n";

    try {
        shared_ptr<VoxelsCoords> voxels_coords(read_from_file<VoxelsCoords>(_multi));

        check(!is_null_ptr(voxels_coords), "failed reading %s");

        cerr << "\nMinimal checks currently implemented. However, if you're seeing this "
                "then it seems that the multi-component ITK file was opened without any errors.\n";

    } catch(...) {
        everything_ok = false;
    }
}

END_NAMESPACE_STD

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
    cerr << "\nTesting multi-component ITK reading is currently impossible since "
            "STIR can only read multicomponent ITK files, it cannot write them.\n"
         << "This could be resolved by adding a multi-component .nii image to STIR's"
            "sample data.\nUntil then, skip this ctest and return success.\n";
    return EXIT_SUCCESS;
    
    if (argc != 2) {
        cerr << "Usage : " << argv[0] << " filename\n";
        return EXIT_FAILURE;
    }

    IOTests_ITKMulticomponent tests(argv[1]);

    tests.run_tests();

    return tests.main_return_value();
}
