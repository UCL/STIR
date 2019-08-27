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
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/ProjDataInMemory.h"
#include "stir/SeparableCartesianMetzImageFilter.h"

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for GPU projectors
*/
class TestDataProcessorProjectors : public RunTests
{
public:
    //! Constructor that can take some input data to run the test with
    TestDataProcessorProjectors(const std::string &sinogram_filename, const float fwhm);

    virtual ~TestDataProcessorProjectors() {}

    void run_tests();
protected:
    std::string _sinogram_filename;
    float _fwhm;
    shared_ptr<ProjDataInMemory> _input_sino_sptr;
    shared_ptr<VoxelsOnCartesianGrid<float> > post_data_processor_bck_proj();
    void pre_data_processor_fwd_proj(const VoxelsOnCartesianGrid<float> &input_image);
};

TestDataProcessorProjectors::TestDataProcessorProjectors(const std::string &sinogram_filename, const float fwhm) :
    _sinogram_filename(sinogram_filename),
    _fwhm(fwhm)
{
}

void
TestDataProcessorProjectors::
run_tests()
{
    try {
        // Open sinogram
        _input_sino_sptr = MAKE_SHARED<ProjDataInMemory>(*ProjData::read_from_file(_sinogram_filename));

        // Back project
        std::cerr << "Tests for post-data-processor back projection\n";
        shared_ptr<VoxelsOnCartesianGrid<float> > bck_projected_im_sptr =
                this->post_data_processor_bck_proj();

        // Forward project
        std::cerr << "Tests for pre-data-processor forward projection\n";
        this->pre_data_processor_fwd_proj(*bck_projected_im_sptr);
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

Succeeded compare_images(const VoxelsOnCartesianGrid<float> &im_1, const VoxelsOnCartesianGrid<float> &im_2)
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

Succeeded compare_sinos(const ProjDataInMemory &proj_data_1, const ProjDataInMemory &proj_data_2)
{
    if (*proj_data_1.get_proj_data_info_sptr() != *proj_data_2.get_proj_data_info_sptr()) {
        std::cout << "\nSinogram proj data info don't match\n";
        return Succeeded::no;
    }

    int min_segment_num = proj_data_1.get_min_segment_num();
    int max_segment_num = proj_data_1.get_max_segment_num();
    int min_view        = proj_data_1.get_min_view_num();
    int max_view        = proj_data_1.get_max_view_num();
    int min_tang_pos    = proj_data_1.get_min_tangential_pos_num();
    int max_tang_pos    = proj_data_1.get_max_tangential_pos_num();

    // Loop over all segments
    for (int segment=min_segment_num; segment<=max_segment_num; ++segment) {

        int min_axial_pos = proj_data_1.get_min_axial_pos_num(segment);
        int max_axial_pos = proj_data_1.get_max_axial_pos_num(segment);

        // Loop over all axial positions
        for (int axial_pos = min_axial_pos; axial_pos<=max_axial_pos; ++axial_pos) {

            // Get the corresponding STIR sinogram
            const Sinogram<float> &sino_1 = proj_data_1.get_sinogram(axial_pos,segment);
            const Sinogram<float> &sino_2 = proj_data_2.get_sinogram(axial_pos,segment);

            // Loop over the STIR view and tangential position
            for (int view=min_view; view<=max_view; ++view) {
                for (int tang_pos=min_tang_pos; tang_pos<=max_tang_pos; ++tang_pos) {

                    // Compare values
                    if (std::abs(sino_1.at(view).at(tang_pos) - sino_2.at(view).at(tang_pos)) > 1e-4f) {
                        std::cout << "\nSinogram values don't match\n";
                        return Succeeded::no;
                    }
                }
            }
        }
    }

    std::cout << "\nSinograms match!\n";
    return Succeeded::yes;
}

void
TestDataProcessorProjectors::
pre_data_processor_fwd_proj(const VoxelsOnCartesianGrid<float> &input_image)
{
    // Create two sinograms, images and forward projectors.
    //    One for pre-data processor forward projection,
    //    the other for data processor then forward projection
    shared_ptr<ProjMatrixByBin> PM_sptr(new  ProjMatrixByBinUsingRayTracing());
    std::vector<shared_ptr<ProjDataInMemory> > sinos(2, MAKE_SHARED<ProjDataInMemory>(*_input_sino_sptr));
    std::vector<shared_ptr<VoxelsOnCartesianGrid<float> > > images(2);
    images[0].reset(input_image.clone());
    images[1].reset(input_image.clone());
    std::vector<ForwardProjectorByBinUsingProjMatrixByBin> projectors(2, ForwardProjectorByBinUsingProjMatrixByBin(PM_sptr));

    // Set up the data processor
    shared_ptr<SeparableCartesianMetzImageFilter<float> > data_processor_sptr;
    get_data_processor(data_processor_sptr, _fwhm);
    data_processor_sptr->set_up(input_image);

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

    if (compare_sinos(*sinos[0],*sinos[1]) == Succeeded::no)
        everything_ok = false;
}

shared_ptr<VoxelsOnCartesianGrid<float> >
TestDataProcessorProjectors::
post_data_processor_bck_proj()
{
    // Create two images and two back projectors.
    //    One for pre-data processor back projection,
    //    the other for data processor then back projection
    std::vector<shared_ptr<VoxelsOnCartesianGrid<float> > > images(2);
    images[0] = MAKE_SHARED<VoxelsOnCartesianGrid<float> >(*_input_sino_sptr->get_proj_data_info_sptr());
    images[1].reset(images[0]->clone());
    shared_ptr<ProjMatrixByBin> PM_sptr(new  ProjMatrixByBinUsingRayTracing());
    std::vector<BackProjectorByBinUsingProjMatrixByBin> projectors(2, BackProjectorByBinUsingProjMatrixByBin(PM_sptr));

    // Set up the data processor
    shared_ptr<SeparableCartesianMetzImageFilter<float> > data_processor_sptr;
    get_data_processor(data_processor_sptr, _fwhm);
    data_processor_sptr->set_up(*images[0]);

    // Loop over twice!
    for (unsigned i=0; i<images.size(); ++i) {

        // Fill sinogram with zeros
        images[i]->fill(0.f);

        // The first time, use the data processor during the back projection
        if (i==0)
            projectors[i].BackProjectorByBin::set_up(_input_sino_sptr->get_proj_data_info_sptr(),images[i],data_processor_sptr);
        // The second time, use the data processor after of the projection process
        else
            projectors[i].set_up(_input_sino_sptr->get_proj_data_info_sptr(),images[i]);

        // Back project
        projectors[i].start_accumulating_in_new_target();
        projectors[i].back_project(*_input_sino_sptr);
        projectors[i].get_output(*images[i]);

        // The second time, use the data processor after the back projection
        if (i!=0)
            data_processor_sptr->apply(*images[i]);
    }

    if (compare_images(*images[0],*images[1]) == Succeeded::no)
        everything_ok = false;

    // Return one of the images to be used for the forward projection
    return images[0];
}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3) {
        std::cerr << "\n\tUsage: " << argv[0] << " sinogram [fwhm]\n";
        return EXIT_FAILURE;
    }

    float fwhm = 5.f;
    if (argc == 3)
        fwhm = float(atof(argv[2]));

    set_default_num_threads();

    TestDataProcessorProjectors test(argv[1], fwhm);

    if (test.is_everything_ok())
        test.run_tests();

    return test.main_return_value();
}