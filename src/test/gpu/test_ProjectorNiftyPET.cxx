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
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/RunTests.h"
#include "stir/num_threads.h"
#include "stir/CPUTimer.h"
#include "stir/IO/OutputFileFormat.h"

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
  TestGPUProjectors(const std::string &image_filename, const std::string &sinogram_filename);
  virtual ~TestGPUProjectors() {}

  void run_tests();
protected:
  void run_projections();
  std::string _image_filename;
  std::string _sinogram_filename;
};

TestGPUProjectors::
TestGPUProjectors(const string &image_filename, const string &sinogram_filename) :
    _image_filename(image_filename),
    _sinogram_filename(sinogram_filename)
{
}

void project(double &time, shared_ptr<ProjDataInMemory> &sino_sptr, shared_ptr<DiscretisedDensity<3,float> > &image_sptr, const shared_ptr<const ProjDataInMemory> input_sino_sptr, const shared_ptr<const DiscretisedDensity<3,float> > input_image_sptr, ForwardProjectorByBin &fwrd_projector, BackProjectorByBin &back_projector, const std::string &name)
{
    image_sptr.reset(input_image_sptr->clone());
    sino_sptr = MAKE_SHARED<ProjDataInMemory>(*input_sino_sptr);

    // set the output image and singorams to zero
    sino_sptr->fill(0.F);
    image_sptr->fill(0.F);

    CPUTimer timer;
    timer.start();

    // Do the forward projection
    std::cerr << "\nDoing forward projection using " << fwrd_projector.get_registered_name() << "...\n";
    fwrd_projector.set_up(sino_sptr->get_proj_data_info_sptr(), image_sptr);
    fwrd_projector.set_input(*input_image_sptr);
    fwrd_projector.forward_project(*sino_sptr);
    timer.stop();
    double time_fwd(timer.value());
    std::cerr << "\tDone! (" << time_fwd << " secs)\n";

    timer.reset();
    timer.start();

    // Set the image to zero
    image_sptr->fill(0.F);

    // Back project
    std::cerr << "\nDoing back projection using " << back_projector.get_registered_name() << "...\n";
    back_projector.set_up(input_sino_sptr->get_proj_data_info_sptr(),image_sptr);
    back_projector.start_accumulating_in_new_target();
    back_projector.back_project(*input_sino_sptr);
    back_projector.get_output(*image_sptr);
    timer.stop();
    double time_bck(timer.value());
    std::cerr << "\tDone! (" << time_bck << " secs)\n";

    time = time_fwd+time_bck;

    std::cerr << "\nTotal time for projection with " << fwrd_projector.get_registered_name() << ": " << time << " secs.\n";

    sino_sptr->write_to_file(name + "_forward_projected.hs");
    shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > > output_file_format_sptr =
            OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr();
    output_file_format_sptr->write_to_file(name + "_back_projected",*image_sptr);
}

void get_min_max_in_proj_data(float &min, float &max, const shared_ptr<const ProjDataInMemory> sino_sptr)
{
    const int min_segment_num = sino_sptr->get_min_segment_num();
    const int max_segment_num = sino_sptr->get_max_segment_num();
    bool accumulators_initialized = false;
    float accum_min=std::numeric_limits<float>::max(); // initialize to very large in case projdata is empty (although that's unlikely)
    float accum_max=std::numeric_limits<float>::min();
    double sum=0.;
    for (int segment_num = min_segment_num; segment_num<= max_segment_num; ++segment_num) {
        const SegmentByView<float> seg(sino_sptr->get_segment_by_view(segment_num));
        const float this_max=seg.find_max();
        const float this_min=seg.find_min();
        sum+=static_cast<double>(seg.sum());
        if(!accumulators_initialized) {
            accum_max=this_max;
            accum_min=this_min;
            accumulators_initialized=true;
        }
        else {
            if (accum_max<this_max) max=this_max;
            if (accum_min>this_min) min=this_min;
        }
    }
}

void
TestGPUProjectors::
run_projections()
{
    // Open image
    const shared_ptr<const DiscretisedDensity<3,float> > image_sptr(DiscretisedDensity<3,float>::read_from_file(_image_filename));
    const shared_ptr<const ProjDataInMemory> sino_sptr = MAKE_SHARED<ProjDataInMemory>(*ProjData::read_from_file(_sinogram_filename));

    // Create output images and sinograms
    shared_ptr<DiscretisedDensity<3,float> > gpu_image_sptr, cpu_image_sptr;
    shared_ptr<ProjDataInMemory> gpu_sino_sptr, cpu_sino_sptr;
    double time_gpu, time_cpu;

    // Forward and back project - gpu
    ForwardProjectorByBinNiftyPET gpu_fwrd;
    BackProjectorByBinNiftyPET    gpu_back;
    project(time_gpu, gpu_sino_sptr, gpu_image_sptr, sino_sptr, image_sptr, gpu_fwrd, gpu_back, "gpu");

    // Forward and back project - cpu
    shared_ptr<ProjMatrixByBin> PM_sptr(new  ProjMatrixByBinUsingRayTracing());
    ForwardProjectorByBinUsingProjMatrixByBin cpu_fwrd(PM_sptr);
    BackProjectorByBinUsingProjMatrixByBin    cpu_back(PM_sptr);
    project(time_cpu, cpu_sino_sptr, cpu_image_sptr, sino_sptr, image_sptr, cpu_fwrd, cpu_back, "cpu");

    // comparison
    std::cout << "\nTime for forward and back projection\n";
    std::cout << "\tGPU: " << time_gpu << "\n";
    std::cout << "\tCPU: " << time_cpu << "\n";

    // Min and max in images
    const float min_image_gpu = gpu_image_sptr->find_min();
    const float min_image_cpu = cpu_image_sptr->find_min();
    const float max_image_gpu = gpu_image_sptr->find_max();
    const float max_image_cpu = cpu_image_sptr->find_max();
    const float percent_diff_max_image = 100.f*(max_image_gpu-max_image_cpu)/max_image_cpu;
    std::cout << "\nMin/max in back projected images\n";
    std::cout << "\tGPU: " << min_image_gpu << " / " << max_image_gpu << "\n";
    std::cout << "\tCPU: " << min_image_cpu << " / " << max_image_cpu << "\n";
    std::cout << "\tDiff in max (%): " << percent_diff_max_image << "\n";

    // Min and max in sinograms
    float min_sino_gpu, min_sino_cpu, max_sino_gpu, max_sino_cpu;
    get_min_max_in_proj_data(min_sino_gpu,max_sino_gpu,gpu_sino_sptr);
    get_min_max_in_proj_data(min_sino_cpu,max_sino_cpu,cpu_sino_sptr);
    const float percent_diff_max_sino = 100.f*(max_sino_gpu-max_sino_cpu)/max_sino_cpu;
    std::cout << "\nMin/max in forward projected sinograms\n";
    std::cout << "\tGPU: " << min_sino_gpu << " / " << max_sino_gpu << "\n";
    std::cout << "\tCPU: " << min_sino_cpu << " / " << max_sino_cpu << "\n";
    std::cout << "\tDiff in max (%): " << percent_diff_max_sino << "\n";

    if (std::abs(percent_diff_max_image) > 1)
        throw std::runtime_error("Images don't agree!");
    if (std::abs(percent_diff_max_sino) > 1)
        throw std::runtime_error("Sinograms don't agree!");
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
