/*
    Copyright (C) 2019, University College London - Richard Brown
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
  \ingroup stir::projector_test

  \brief Test program for forward and backwards projectors using wrappers around NiftyPET's GPU projectors (stir::ForwardProjectorByBinNiftyPET and stir::BackProjectorByBinNiftyPET)

  \author Richard Brown
*/

#include "stir/gpu/ForwardProjectorByBinNiftyPET.h"
#include "stir/RunTests.h"
#include "stir/num_threads.h"

START_NAMESPACE_STIR

using namespace std;

/*!
  \ingroup test
  \brief Test class for GPU projectors

*/
class TestGPUProjectors : public RunTests
{
public:
  //! Constructor that can take some input data to run the test with
  TestGPUProjectors(std::string image_filename, std::string sinogram_filename);
  virtual ~TestGPUProjectors() {}

  void run_tests();
protected:
  void run_projections();
  std::string _image_filename;
  std::string _sinogram_filename;
};

TestGPUProjectors::
TestGPUProjectors(std::string image_filename, std::string sinogram_filename) :
    _image_filename(image_filename),
    _sinogram_filename(sinogram_filename)
{
}

void
TestGPUProjectors::
run_projections()
{
    // Open image
    std::cerr << "\nReading the image to project...\n";
    shared_ptr<DiscretisedDensity<3,float> > input(
                DiscretisedDensity<3,float>::read_from_file(_image_filename));
    if(is_null_ptr(input)) {
        std::cerr << "\nError reading image to project.\n";
        everything_ok = false;
        return;
    }
    std::cerr << "\tDone!\n";

    // Open sinogram
    std::cerr << "\nReading the sinogram to project...\n";
    shared_ptr<ProjData> proj_data =
      ProjData::read_from_file(_sinogram_filename);
    std::cerr << "\tDone!\n";

    // Forward project
    std::cerr << "\nDoing forward projection...\n";
    ForwardProjectorByBinNiftyPET fwrd_projector;
    fwrd_projector.set_up(proj_data->get_proj_data_info_sptr(), input);
    fwrd_projector.set_input(input);
    proj_data->fill(0.F);
    fwrd_projector.forward_project(*proj_data);
    std::cerr << "\tDone!\n";
}

void
TestGPUProjectors::
run_tests()
{
    try {
        cerr << "Tests for GPU-accelerated projectors\n";
        this->run_projections();
    }
    catch(const std::exception &error) {
        std::cerr << "\nHere's the error:\n\t" << error.what() << "\n\n";
        everything_ok = false;
    }
    catch(...) {
        everything_ok = false;
    }
}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
    if (argc != 3) {
        cerr << "\n\tUsage: " << argv[0] << " image sinogram\n";
        return EXIT_FAILURE;
    }

    set_default_num_threads();

    TestGPUProjectors test(argv[1], argv[2]);

    if (test.is_everything_ok())
        test.run_tests();

    return test.main_return_value();
}
