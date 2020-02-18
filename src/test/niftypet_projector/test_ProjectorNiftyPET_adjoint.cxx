/*
    Copyright (C) 2020, University College London - Richard Brown
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
  \ingroup projector_test

  \brief Test that the NiftyPET forward and back projectors are adjoint to one another.

  \author Richard Brown
*/

#include "stir/recon_buildblock/niftypet_projector/ForwardProjectorByBinNiftyPET.h"
#include "stir/recon_buildblock/niftypet_projector/BackProjectorByBinNiftyPET.h"
#include "stir/RunTests.h"
#include "stir/num_threads.h"
#include "stir/CPUTimer.h"
#include "stir/Verbosity.h"
#include "stir/recon_array_functions.h"
#include <stdlib.h>
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
    //! Constructor
    TestGPUProjectors(const unsigned num_attempts)
        : _num_attempts(num_attempts),
          _time_fwrd(0), _time_back(0),
          _cumulative_result(0),
          _num_cumulative_results(0) {}

    /// Destructor
    virtual ~TestGPUProjectors() {}

    /// Run tests
    void run_tests();

protected:

    /// Set up
    void set_up();

    /// Set up the image
    void set_up_image();
    /// Set up the sinogram
    void set_up_sino();

    /// Set up forward projector
    void set_up_projector_forward();
    /// Set up back projector
    void set_up_projector_back();

    /// Random image
    void random_image();
    /// Random sinogram
    void random_sino();

    /// Forward project
    void forward_project();
    /// Back project
    void back_project();

    /// test the adjoint multiple times
    void test_adjoints();
    /// test the adjoint
    void test_adjoint();
    /// test the inner product
    void test_inner_product();

    const unsigned _num_attempts;
    ForwardProjectorByBinNiftyPET _projector_fwrd;
    BackProjectorByBinNiftyPET    _projector_back;
    shared_ptr<
    VoxelsOnCartesianGrid<float> > _image_sptr;
    shared_ptr<ProjDataInMemory> _sino_sptr;
    shared_ptr<VoxelsOnCartesianGrid<float> > _projected_image_sptr;
    shared_ptr<ProjDataInMemory> _projected_sino_sptr;
    double _time_fwrd, _time_back;
    double _cumulative_result;
    unsigned _num_cumulative_results;
};

void
TestGPUProjectors::
set_up()
{
    std::cerr << "Setting up...\n";

    // random seed
    srand(time(NULL));

    set_up_sino();
    set_up_image();
    set_up_projector_forward();
    set_up_projector_back();
}

void
TestGPUProjectors::
set_up_image()
{
    std::cerr << "\tSetting up image...\n";

    BasicCoordinate<3, int> min_image_indices(make_coordinate(0,  -160, -160));
    BasicCoordinate<3, int> max_image_indices(make_coordinate(126, 159,  159));
    IndexRange<3> range = IndexRange<3>(min_image_indices,max_image_indices);

    _image_sptr.reset(new VoxelsOnCartesianGrid<float>(
                          _sino_sptr->get_exam_info_sptr(),
                          range,
                          CartesianCoordinate3D<float>(0.f,0.f,0.f),
                          CartesianCoordinate3D<float>(2.03125f, 2.08626f, 2.08626f)
                          ));

    float val(0.f);
    for(Array<3,float>::full_iterator iter = _image_sptr->begin_all(); iter != _image_sptr->end_all(); ++iter, ++val)
        *iter = val;

    // Truncate it to a small cylinder
    truncate_rim(*_image_sptr,17);

    // Projected image is clone
    _projected_image_sptr.reset(_image_sptr->clone());
}

void
TestGPUProjectors::
set_up_sino()
{
    std::cerr << "\tSetting up sinogram...\n";
    // Create scanner
    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::Siemens_mMR));

    // ExamInfo
    shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
    exam_info_sptr->imaging_modality = ImagingModality::PT;

    shared_ptr<ProjDataInfo> proj_data_info_sptr(
                ProjDataInfo::construct_proj_data_info(
                    scanner_sptr,
                    11, // span
                    /* mMR needs maxDelta of */60,
                    scanner_sptr->get_num_detectors_per_ring()/2,
                    scanner_sptr->get_max_num_non_arccorrected_bins(),
                    /* arc_correction*/false));

    // Create ProjData
    _sino_sptr.reset(new ProjDataInMemory(exam_info_sptr,proj_data_info_sptr));

    // Get dimensions of STIR sinogram
    int min_view      = _sino_sptr->get_min_view_num();
    int max_view      = _sino_sptr->get_max_view_num();
    int min_tang_pos  = _sino_sptr->get_min_tangential_pos_num();
    int max_tang_pos  = _sino_sptr->get_max_tangential_pos_num();

    int num_sinograms = _sino_sptr->get_num_axial_poss(0);
    for (int s=1; s<= _sino_sptr->get_max_segment_num(); ++s)
        num_sinograms += 2* _sino_sptr->get_num_axial_poss(s);

    unsigned num_elements = unsigned(num_sinograms * (1+max_view-min_view) * (1+max_tang_pos-min_tang_pos));

    // Create array
    std::vector<float> arr(num_elements);
    for (unsigned i=0; i<num_elements; ++i)
        arr[i] = float(i);

    // Fill
    _sino_sptr->fill_from(arr.begin());

    // Projected sinogram is clone
    _projected_sino_sptr = MAKE_SHARED<ProjDataInMemory>(*_sino_sptr);
}

void
TestGPUProjectors::
set_up_projector_forward()
{
    std::cerr << "\tSetting up forward projector...\n";
    _projector_fwrd.set_verbosity(false);
    _projector_fwrd.set_up(_sino_sptr->get_proj_data_info_sptr(),_image_sptr);
}

void
TestGPUProjectors::
set_up_projector_back()
{
    std::cerr << "\tSetting up back projector...\n";
    _projector_back.set_verbosity(false);
    _projector_back.set_up(_sino_sptr->get_proj_data_info_sptr(),_image_sptr);
}

static int get_rand(const int lower, const int upper)
{
    return rand() % upper + lower;
}

static float get_rand(const float lower, const float upper)
{
    return lower + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/(upper-lower)));
}

void
TestGPUProjectors::
random_image()
{
    std::cerr << "\tGetting random image...\n";
    _image_sptr->fill(0.f);
    BasicCoordinate<3, int> min_indices = _image_sptr->get_min_indices();
    BasicCoordinate<3, int> max_indices = _image_sptr->get_max_indices();

    const unsigned num_rand_voxels = get_rand(1,2000);

    for (unsigned i=0; i<num_rand_voxels; ++i) {
        // Get random index
        BasicCoordinate<3, int> rand_idx;
        for (unsigned j=1; j<=3; ++j)
            rand_idx.at(j) = get_rand(min_indices.at(j),max_indices.at(j));
        std::cout << "rand idx = " << rand_idx << "\n";
        float val = get_rand(0.f,100.f);
        _image_sptr->at(rand_idx) = val;
    }
    exit(0);
}

void
TestGPUProjectors::
random_sino()
{
    std::cerr << "\tGetting random sinogram...\n";
    _sino_sptr->fill(0.f);

    int min_segment_num = _sino_sptr->get_min_segment_num();
    int max_segment_num = _sino_sptr->get_max_segment_num();

    // Get number of elements
    unsigned num_elements(0);
    for (int segment_num = min_segment_num; segment_num<= max_segment_num; ++segment_num)
        num_elements += unsigned(_sino_sptr->get_max_axial_pos_num(segment_num) -
                                 _sino_sptr->get_min_axial_pos_num(segment_num)) + 1;
    num_elements *= unsigned(_sino_sptr->get_max_view_num() -
                             _sino_sptr->get_min_view_num()) + 1;
    num_elements *= unsigned(_sino_sptr->get_max_tangential_pos_num() -
                             _sino_sptr->get_min_tangential_pos_num()) + 1;

    // Create array
    std::vector<float> arr(num_elements,20);

//    const unsigned num_rand_voxels = get_rand(1,2000);

//    for (unsigned i=0; i<num_rand_voxels; ++i) {
//        // Get random index
//        unsigned idx = get_rand(0,num_elements-1);
//        float val = get_rand(0.f,100.f);
//        arr[idx] = val;
//    }

    _sino_sptr->fill_from(arr.begin());
}

void
TestGPUProjectors::
forward_project()
{
    std::cerr << "\tForward projecting...\n";
    _projected_sino_sptr->fill(0.f);

    CPUTimer timer;
    timer.start();

    _projector_fwrd.set_input(*_image_sptr);
    _projector_fwrd.forward_project(*_projected_sino_sptr);

    timer.stop();
    _time_fwrd += timer.value();
}

void
TestGPUProjectors::
back_project()
{
    std::cerr << "\tBack projecting...\n";
    _projected_image_sptr->fill(0.f);

    CPUTimer timer;
    timer.start();

    _projector_back.start_accumulating_in_new_target();
    _projector_back.back_project(*_sino_sptr);
    _projector_back.get_output(*_projected_image_sptr);

    timer.stop();
    _time_back += timer.value();
}

void
TestGPUProjectors::
test_adjoints()
{
    set_up();

    for (unsigned i=0; i<_num_attempts; ++i) {
        std::cerr << "\nTesting adjoint-ness of NiftyPET's forward/back projectors. Attempt " << i << " of " << _num_attempts << "...\n";
        test_adjoint();
        std::cerr << "\nAverage time per forward/back project: " << _time_fwrd/double(i+1) << " and " << _time_back/double(i+1) << " s.\n";
        std::cerr << "Cumulative adjoint result (0 good, 1 bad): " << _cumulative_result/float(_num_cumulative_results) << ".\n";
    }
}

static float get_inner_product(
        const VoxelsOnCartesianGrid<float> &im1,
        const VoxelsOnCartesianGrid<float> &im2)
{
    std::cerr << "\t\tChecking inner products between images...\n";

    return std::inner_product(im1.begin_all(),im1.end_all(),im2.begin_all(),0.f);
}

static float get_inner_product(
        const ProjDataInMemory &proj1,
        const ProjDataInMemory &proj2)
{
    std::cerr << "\t\tChecking inner products between sinograms...\n";

    int min_segment_num = proj1.get_min_segment_num();
    int max_segment_num = proj1.get_max_segment_num();

    // Get number of elements
    unsigned num_elements(0);
    for (int segment_num = min_segment_num; segment_num<= max_segment_num; ++segment_num)
        num_elements += unsigned(proj1.get_max_axial_pos_num(segment_num) -
                                 proj1.get_min_axial_pos_num(segment_num)) + 1;
    num_elements *= unsigned(proj1.get_max_view_num() -
                             proj1.get_min_view_num()) + 1;
    num_elements *= unsigned(proj1.get_max_tangential_pos_num() -
                             proj1.get_min_tangential_pos_num()) + 1;

    // Create arrays
    std::vector<float> arr1(num_elements), arr2(num_elements);
    proj1.copy_to(arr1.begin());
    proj2.copy_to(arr2.begin());

    return std::inner_product(arr1.begin(),arr1.end(),arr2.begin(),0.f);
}

static float find_max(const ProjDataInMemory &prj)
{
    int min_segment_num = prj.get_min_segment_num();
    int max_segment_num = prj.get_max_segment_num();

    // Get number of elements
    unsigned num_elements(0);
    for (int segment_num = min_segment_num; segment_num<= max_segment_num; ++segment_num)
        num_elements += unsigned(prj.get_max_axial_pos_num(segment_num) -
                                 prj.get_min_axial_pos_num(segment_num)) + 1;
    num_elements *= unsigned(prj.get_max_view_num() -
                             prj.get_min_view_num()) + 1;
    num_elements *= unsigned(prj.get_max_tangential_pos_num() -
                             prj.get_min_tangential_pos_num()) + 1;

    // Create arrays
    std::vector<float> arr(num_elements);
    prj.copy_to(arr.begin());
    return *std::max_element(arr.begin(),arr.end());
}

void TestGPUProjectors::
test_inner_product()
{
    std::cerr << "\tChecking inner products...\n";
    const float inner_product_images = get_inner_product(*_image_sptr,*_projected_image_sptr);
    const float inner_product_sinos  = get_inner_product(*_sino_sptr, *_projected_sino_sptr );

    // Check the adjoint is truly the adjoint with: |<x, Ty> - <y, Tsx>| / 0.5*(|<x, Ty>|+|<y, Tsx>|) < epsilon
    if (std::abs(inner_product_images) + std::abs(inner_product_sinos) > 1e-4f) {
        float adjoint_test =
                std::abs(inner_product_images - inner_product_sinos) /
                (0.5f * (std::abs(inner_product_images) + std::abs(inner_product_sinos)));
        std::cout << "\ninner product between two images    = " << inner_product_images << "\n";
        std::cout << "inner product between two sinograms = " << inner_product_sinos << "\n";
        std::cout << "|<x, Ty> - <y, Tsx>| / 0.5*(|<x, Ty>|+|<y, Tsx>|) = " << adjoint_test << "\n";
        _cumulative_result += adjoint_test;
        ++_num_cumulative_results;
    }

    std::cout << "\n max in input image = " << _image_sptr->find_max() << "\n";
    std::cout << "\n max in projected image = " << _projected_image_sptr->find_max() << "\n";
    std::cout << "\n max in input image = " << find_max(*_sino_sptr) << "\n";
    std::cout << "\n max in projected image = " << find_max(*_projected_sino_sptr) << "\n";

    ITKOutputFileFormat itk_output;
    itk_output.default_extension = ".nii";
    itk_output.write_to_file("/home/rich/Documents/Data/GPU_adjoint_test/input_image",*_image_sptr);
    itk_output.write_to_file("/home/rich/Documents/Data/GPU_adjoint_test/projected_image",*_projected_image_sptr);

    _sino_sptr->write_to_file("/home/rich/Documents/Data/GPU_adjoint_test/input_sino");
    _projected_sino_sptr->write_to_file("/home/rich/Documents/Data/GPU_adjoint_test/projected_sino");
}

void
TestGPUProjectors::
test_adjoint()
{
    random_image();
    random_sino();

    forward_project();
    back_project();

    test_inner_product();
}

void
TestGPUProjectors::
run_tests()
{
    try {
        cerr << "Testing whether forward and back projectors are adjoint...\n";
        this->test_adjoints();
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

void print_usage()
{
    std::cerr << "\n\tUsage: test_ProjectorNiftyPET_adjoint [-h] <number of attempts>\n";
}

int main(int argc, char **argv)
{
    set_default_num_threads();
    Verbosity::set(0);

    // Require a single argument
    if (argc != 2) {
        print_usage();
        return EXIT_SUCCESS;
    }

    // If help desired
    if (strcmp(argv[1],"-h") ==0) {
        print_usage();
        return EXIT_SUCCESS;
    }

    const unsigned num_attempts = std::stoi(argv[1]);

    TestGPUProjectors test(num_attempts);

    if (test.is_everything_ok())
        test.run_tests();

    return test.main_return_value();
}
