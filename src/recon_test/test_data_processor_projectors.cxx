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
  \brief Test program for forward and backwards projectors with pre- and post- data processors (smoothing in this case).
  \author Richard Brown
*/

#include "stir/RunTests.h"
#include "stir/num_threads.h"
#include "stir/CPUTimer.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/ProjDataInMemory.h"
#include "stir/SeparableCartesianMetzImageFilter.h"

START_NAMESPACE_STIR

using namespace std;

/*!
  \ingroup test
  \brief Test class for GPU projectors
*/
class TestDataProcessorProjectors : public RunTests
{
public:
    //! Constructor that can take some input data to run the test with
    TestDataProcessorProjectors(const std::string &image_filename, const std::string &sinogram_filename, const float fwhm);

    virtual ~TestDataProcessorProjectors() {}

    void run_tests();
protected:
    std::string _image_filename;
    std::string _sinogram_filename;
    float _fwhm;
    shared_ptr<DiscretisedDensity<3,float> > _input_image_sptr;
    shared_ptr<ProjDataInMemory> _input_sino_sptr;
    Succeeded pre_data_processor_fwd_proj() const;
    Succeeded post_data_processor_bck_proj() const;
};

TestDataProcessorProjectors::TestDataProcessorProjectors(const string &image_filename, const string &sinogram_filename, const float fwhm) :
    _image_filename(image_filename),
    _sinogram_filename(sinogram_filename),
    _fwhm(fwhm)
{
}

void
TestDataProcessorProjectors::
run_tests()
{
    try {

        // Open image
        _input_image_sptr.reset(DiscretisedDensity<3,float>::read_from_file(_image_filename));
        _input_sino_sptr = MAKE_SHARED<ProjDataInMemory>(*ProjData::read_from_file(_sinogram_filename));

        cerr << "Tests for pre-data-processor forward projection\n";
        Succeeded success_fwd = this->pre_data_processor_fwd_proj();
        cerr << "Tests for post-data-processor back projection\n";
        Succeeded success_bck = this->post_data_processor_bck_proj();
        if (success_fwd == Succeeded::no || success_bck == Succeeded::no)
            everything_ok = false;
    }
    catch(const std::exception &error) {
        std::cerr << "\nHere's the error:\n\t" << error.what() << "\n\n";
        everything_ok = false;
    }
    catch(...) {
        everything_ok = false;
    }
}

void
get_data_processor(shared_ptr<SeparableCartesianMetzImageFilter<float> > &data_processor_sptr, const float fwhm)
{
    data_processor_sptr.reset(new SeparableCartesianMetzImageFilter<float>);
    std::string buffer;
    std::stringstream parameterstream(buffer);

    parameterstream << "Separable Cartesian Metz Filter Parameters :=\n"
                    << "x-dir filter FWHM (in mm):= " << fwhm << "\n"
                    << "y-dir filter FWHM (in mm):= " << fwhm << "\n"
                    << "z-dir filter FWHM (in mm):= " << fwhm << "\n"
                    << "x-dir filter Metz power:= .0\n"
                    << "y-dir filter Metz power:= .0\n"
                    << "z-dir filter Metz power:=.0\n"
                    << "END Separable Cartesian Metz Filter Parameters :=\n";
    data_processor_sptr->parse(parameterstream);
}

Succeeded compare_images(const DiscretisedDensity<3,float> &im_1, const DiscretisedDensity<3,float> &im_2)
{
    if (im_1 == im_2) {
        std::cout << "\nImages match!\n";
        return Succeeded::yes;
    }

    std::cout << "\nImages don't match\n";
    std::cout << "Min im1 / im2 = " << im_1.find_min() << " / " << im_2.find_min() << "\n";
    std::cout << "Max im1 / im2 = " << im_1.find_max() << " / " << im_2.find_max() << "\n";
    std::cout << "Sum im1 / im2 = " << im_1.sum()      << " / " << im_2.sum()      << "\n";

    return Succeeded::no;
}

Succeeded
TestDataProcessorProjectors::
pre_data_processor_fwd_proj() const
{
    // Create two sinograms, images and two forward projectors.
    //    One for pre-data processor forward projection,
    //    the other for data processor then forward projection
    shared_ptr<ProjMatrixByBin> PM_sptr(new  ProjMatrixByBinUsingRayTracing());
    std::vector<shared_ptr<ProjDataInMemory> > sinos(2, MAKE_SHARED<ProjDataInMemory>(*_input_sino_sptr));
    std::vector<shared_ptr<DiscretisedDensity<3,float> > > images(2);
    images[0].reset(_input_image_sptr->clone());
    images[1].reset(_input_image_sptr->clone());
    std::vector<ForwardProjectorByBinUsingProjMatrixByBin> projectors(2, ForwardProjectorByBinUsingProjMatrixByBin(PM_sptr));

    // Set up the data processor
    shared_ptr<SeparableCartesianMetzImageFilter<float> > data_processor_sptr;
    get_data_processor(data_processor_sptr, _fwhm);
    data_processor_sptr->set_up(*_input_image_sptr);

    // Loop over twice!
    for (unsigned i=0; i<sinos.size(); ++i) {

        // Fill sinogram with zeros
        sinos[i]->fill(0.f);

        // The first time, use the data processor during the forward projection
        if (i==0)
            projectors[i].ForwardProjectorByBin::set_up(_input_sino_sptr->get_proj_data_info_sptr(),images[i],data_processor_sptr);
        // The second time, use the data processor outside of the projection process
        else {
            projectors[i].set_up(_input_sino_sptr->get_proj_data_info_sptr(),images[i]);
            data_processor_sptr->apply(*images[i]);
        }

        // Forward project
        projectors[i].set_input(*images[i]);
        projectors[i].forward_project(*sinos[i]);
    }

    return compare_images(*images[0],*images[1]);
}

Succeeded
TestDataProcessorProjectors::
post_data_processor_bck_proj() const
{
    // Create two images and two back projectors.
    //    One for pre-data processor back projection,
    //    the other for data processor then back projection
    shared_ptr<ProjMatrixByBin> PM_sptr(new  ProjMatrixByBinUsingRayTracing());
    std::vector<shared_ptr<DiscretisedDensity<3,float> > > images(2);
    images[0].reset(_input_image_sptr->clone());
    images[1].reset(_input_image_sptr->clone());
    std::vector<BackProjectorByBinUsingProjMatrixByBin> projectors(2, BackProjectorByBinUsingProjMatrixByBin(PM_sptr));

    // Set up the data processor
    shared_ptr<SeparableCartesianMetzImageFilter<float> > data_processor_sptr;
    get_data_processor(data_processor_sptr, _fwhm);
    data_processor_sptr->set_up(*_input_image_sptr);

    // Loop over twice!
    for (unsigned i=0; i<images.size(); ++i) {

        // Fill sinogram with zeros
        images[i]->fill(0.f);

        // The first time, use the data processor during the back projection
        if (i==0)
            projectors[i].BackProjectorByBin::set_up(_input_sino_sptr->get_proj_data_info_sptr(),_input_image_sptr,data_processor_sptr);
        // The second time, use the data processor after of the projection process
        else
            projectors[i].set_up(_input_sino_sptr->get_proj_data_info_sptr(),_input_image_sptr);

        // Back project
        projectors[i].start_accumulating_in_new_target();
        projectors[i].back_project(*_input_sino_sptr);
        projectors[i].get_output(*images[i]);

        // The second time, use the data processor after the back projection
        if (i!=0)
            data_processor_sptr->apply(*images[i]);
    }

    return compare_images(*images[0],*images[1]);
}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
    if (argc != 4) {
        cerr << "\n\tUsage: " << argv[0] << " image sinogram fwhm\n";
        return EXIT_FAILURE;
    }

    set_default_num_threads();

    TestDataProcessorProjectors test(argv[1], argv[2], float(atof(argv[3])));

    if (test.is_everything_ok())
        test.run_tests();

    return test.main_return_value();
}