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
#include "stir/gpu/BackProjectorByBinNiftyPET.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/RunTests.h"
#include "stir/num_threads.h"
#include "stir/CPUTimer.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/ITKOutputFileFormat.h"

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

template<class fwrd, class back>
void project(const shared_ptr<const ProjDataInMemory> input_sino_sptr, const shared_ptr<const DiscretisedDensity<3,float> > input_image_sptr)
{
    CPUTimer timer;
    timer.start();

    // Copy image and sinogram
    shared_ptr<DiscretisedDensity<3,float> > image_sptr(input_image_sptr->clone());
    shared_ptr<ProjDataInMemory> sino_sptr = MAKE_SHARED<ProjDataInMemory>(*input_sino_sptr);

    // Set the sinogram to zero
    sino_sptr->fill(0.F);

    // Do the forward projection
    std::cerr << "\nDoing forward projection using " << fwrd::registered_name << "...\n";
    fwrd fwrd_projector;
    fwrd_projector.set_up(sino_sptr->get_proj_data_info_sptr(), image_sptr);
    fwrd_projector.set_input(image_sptr);
    fwrd_projector.forward_project(*sino_sptr);
    timer.stop();
    double time_fwd(timer.value());
    std::cerr << "\tDone! (" << time_fwd << " secs)\n";

    timer.reset();
    timer.start();

    // Set the image to zero
    image_sptr->fill(0.F);

    // Back project
    std::cerr << "\nDoing back projection using " << back::registered_name << "...\n";
    back back_projector;
    back_projector.set_up(sino_sptr->get_proj_data_info_sptr(),image_sptr);
    back_projector.start_accumulating_in_new_image();
    back_projector.back_project(*sino_sptr);
    back_projector.get_output(*image_sptr);
    timer.stop();
    double time_bck(timer.value());
    std::cerr << "\tDone! (" << time_bck << " secs)\n";

    std::cerr << "\nTotal time for projection with " << fwrd::registered_name << ": " << time_fwd+time_bck << " secs.\n";

    sino_sptr->write_to_file("/home/rich/Documents/Data/forward_projected.hs");
    shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > > output_file_format_sptr =
            OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr();
    output_file_format_sptr->write_to_file("/home/rich/Documents/Data/forward_then_back_projected",*image_sptr);
    ITKOutputFileFormat itk_writer;
    itk_writer.default_extension = ".nii";
    itk_writer.write_to_file("/home/rich/Documents/Data/forward_then_back_projected",*image_sptr);
}

void
TestGPUProjectors::
run_projections()
{
    // Open image
    const shared_ptr<const DiscretisedDensity<3,float> > image_sptr(DiscretisedDensity<3,float>::read_from_file(_image_filename));
    const shared_ptr<const ProjDataInMemory> sino_sptr = MAKE_SHARED<ProjDataInMemory>(*ProjData::read_from_file(_sinogram_filename));

    // Forward and back project
    project<ForwardProjectorByBinNiftyPET,BackProjectorByBinNiftyPET>(sino_sptr,image_sptr);
    //    project<ForwardProjectorByBinUsingProjMatrixByBin,BackProjectorByBinUsingProjMatrixByBin>(proj_data,input);

    // comparison
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
