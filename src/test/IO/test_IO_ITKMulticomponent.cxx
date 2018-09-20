//
//
/*!

  \file
  \ingroup test

  \brief A simple program to test the reading of multicomponent images with ITK.

  \author Kris Thielemans
  \author Richard Brown



  To run the test, you should use a command line argument with the name of a file.
*/
/*
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

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief A simple class to test the reading of multicomponent ITK images.
*/
class IOTests_ITKMulticomponent : public RunTests
{
public:
    explicit IOTests_ITKMulticomponent(const std::string &multi) :
        _multi(multi) {}

    void run_tests();

protected:

    std::string _multi;
};
void IOTests_ITKMulticomponent::run_tests()
{
    typedef VoxelsOnCartesianGrid<CartesianCoordinate3D<float> > VoxelsCoords;

    // Read the files
    std::cerr << "\nReading: " << _multi << "\n";

    try {
        shared_ptr<VoxelsCoords> voxels_coords(read_from_file<VoxelsCoords>(_multi));

        check(!is_null_ptr(voxels_coords), "failed reading %s");

        // Check sizes
        check_if_equal<size_t>(voxels_coords->size(),                  17);
        check_if_equal<size_t>(voxels_coords->at(0).size(),             9);
        check_if_equal<size_t>(voxels_coords->at(0).at(0).size(),      27);
        check_if_equal<size_t>(voxels_coords->at(0).at(0).at(0).size(), 3);

        // Check voxel sizes
        check_if_equal<float>(voxels_coords->get_voxel_size()[1], 2.03125F);
        check_if_equal<float>(voxels_coords->get_voxel_size()[2], 2.08626F);
        check_if_equal<float>(voxels_coords->get_voxel_size()[3], 2.08626F);

    } catch(...) {
        everything_ok = false;
    }
}

END_NAMESPACE_STD

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage : " << argv[0] << " filename\n";
        return EXIT_FAILURE;
    }

    IOTests_ITKMulticomponent tests(argv[1]);

    tests.run_tests();

    return tests.main_return_value();
}
