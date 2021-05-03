/*
    Copyright (C) 2020, University College London - Richard Brown
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup projector_test

  \brief Test that the NiftyPET forward and back projectors are adjoint to one another.

  \author Richard Brown
*/

#include "stir/recon_buildblock/NiftyPET_projector/ForwardProjectorByBinNiftyPET.h"
#include "stir/recon_buildblock/NiftyPET_projector/BackProjectorByBinNiftyPET.h"
#include "stir/RunTests.h"
#include "stir/num_threads.h"
#include "stir/CPUTimer.h"
#include "stir/Verbosity.h"
#include "stir/recon_array_functions.h"
#include "stir/Shape/Ellipsoid.h"
#include "stir/ProjDataInMemory.h"

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
    explicit TestGPUProjectors(const unsigned num_attempts)
        : _num_attempts(num_attempts),
          _time_fwrd(0), _time_back(0) {}

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
    /// test the adjoint multiple times
    void test_adjoints();

    const unsigned _num_attempts;
    ForwardProjectorByBinNiftyPET _projector_fwrd;
    BackProjectorByBinNiftyPET    _projector_back;
    shared_ptr<VoxelsOnCartesianGrid<float> >  _image_sptr;
    shared_ptr<ProjDataInMemory>               _sino_sptr;
    shared_ptr<VoxelsOnCartesianGrid<float> >  _projected_image_sptr;
    shared_ptr<ProjDataInMemory>               _projected_sino_sptr;
    double _time_fwrd, _time_back;
    std::vector<float> _results;
};

static int get_rand(const int lower, const int upper)
{
    return rand() % upper + lower;
}

static float get_rand(const float lower, const float upper)
{
    return lower + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/(upper-lower)));
}

static
CartesianCoordinate3D<float>
get_rand_point(const CartesianCoordinate3D<float> &min, const CartesianCoordinate3D<float> &max)
{
    return CartesianCoordinate3D<float>(
                get_rand(min.at(1), max.at(1)),
                get_rand(min.at(2), max.at(2)),
                get_rand(min.at(3), max.at(3)));
}

static
CartesianCoordinate3D<float>
get_rand_point(const float min, const float max)
{
    return CartesianCoordinate3D<float>(
                get_rand(min, max),
                get_rand(min, max),
                get_rand(min, max));
}

static float find_max(const ProjDataInMemory &prj)
{
    // Get number of elements
    unsigned num_elements = prj.size_all();

    // Create arrays
    std::vector<float> arr(num_elements);
    prj.copy_to(arr.begin());
    return *std::max_element(arr.begin(),arr.end());
}

void
TestGPUProjectors::
set_up()
{
    std::cerr << "Setting up...\n";

    // random seed
    srand(time(NULL));

    set_up_sino();
    set_up_image();

    std::cerr << "\tSetting up projectors...\n";
    _projector_fwrd.set_verbosity(false);
    _projector_fwrd.set_up(_sino_sptr->get_proj_data_info_sptr(),_image_sptr);
    _projector_back.set_verbosity(false);
    _projector_back.set_up(_sino_sptr->get_proj_data_info_sptr(),_image_sptr);
}

void
TestGPUProjectors::
set_up_image()
{
    std::cerr << "\tSetting up images...\n";

    BasicCoordinate<3, int> min_image_indices(make_coordinate(0,  -160, -160));
    BasicCoordinate<3, int> max_image_indices(make_coordinate(126, 159,  159));
    IndexRange<3> range = IndexRange<3>(min_image_indices,max_image_indices);

    _image_sptr = MAKE_SHARED<VoxelsOnCartesianGrid<float> >(
                _sino_sptr->get_exam_info_sptr(),
                range,
                CartesianCoordinate3D<float>(0.f,0.f,0.f),
                CartesianCoordinate3D<float>(2.03125f, 2.08626f, 2.08626f));

    // Fill
    _image_sptr->fill(0.f);

    // Make projected image a copy
    _projected_image_sptr.reset(_image_sptr->clone());
}

void
TestGPUProjectors::
set_up_sino()
{
    std::cerr << "\tSetting up sinograms...\n";
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

    _sino_sptr = MAKE_SHARED<ProjDataInMemory>(exam_info_sptr,proj_data_info_sptr);

    // Create vector to be able to fill sinogram
    const size_t num_elements = _sino_sptr->size_all();
    std::vector<float> arr(num_elements);
    for (unsigned i=0; i<num_elements; ++i)
        arr[i] = float(i);
    _sino_sptr->fill_from(arr.begin());

    _projected_sino_sptr = MAKE_SHARED<ProjDataInMemory>(*_sino_sptr);
}

static
void
get_random_sino(ProjData &sino)
{
    std::cerr << "Getting random sinogram...\n";

    const size_t num_elements = sino.size_all();
    std::vector<float> arr(num_elements,0.f);

    // Number of voxels to fill
    const unsigned num_rand_voxels = get_rand(100,2000);
    for (unsigned j=0; j<num_rand_voxels; ++j) {
        // Get random index
        const unsigned idx = get_rand(0,num_elements-1);
        const float val = get_rand(1.f,100.f);
        arr[idx] = val;
    }
    sino.fill_from(arr.begin());
}

static
void
get_random_image(VoxelsOnCartesianGrid<float> &im)
{
    std::cerr << "Getting random image...\n";

    im.fill(0.f);
    auto temp = *im.clone();

    CartesianCoordinate3D<float> min = im.get_physical_coordinates_for_indices(im.get_min_indices());
    CartesianCoordinate3D<float> max = im.get_physical_coordinates_for_indices(im.get_max_indices());
    CartesianCoordinate3D<int> num_samles = {10,10,10};
    const float min_radius = 20.f;
    const float max_radius = 100.f;
    const float min_intensity = 10.f;
    const float max_intensity = 100.f;

    // Keep looping until random image contains non-zeroes
    // (in case all ellipsoids were in part that got truncated)
    while (true) {

        const unsigned num_ellipsoids = get_rand(1,5);

        for (unsigned j=0; j<num_ellipsoids; ++j) {

            // Get radii, centre and intensity of ellipsoid
            const auto radii      = get_rand_point(min_radius, max_radius);
            const auto centre     = get_rand_point(min,max);
            const float intensity = get_rand(min_intensity,max_intensity);
            // Create shape
            Ellipsoid ellipsoid(radii,centre);
            // Get shape as image
            temp.fill(0.f);
            ellipsoid.construct_volume(temp, num_samles);
            temp *= intensity;
            // Add to output image
            im += temp;
        }

        // Truncate it to a small cylinder
        truncate_rim(im,17);

        // Check that image contains non-zeros. 
        if (im.find_max() > 1e-4f)
            break;
        else 
            std::cout << "\nRandom image contains all zeroes, regenerating...\n";
    }
}

static
void
forward_project(ProjData &sino, const VoxelsOnCartesianGrid<float> &im, ForwardProjectorByBinNiftyPET &projector, double &time)
{
    std::cerr << "Forward projecting...\n";
    sino.fill(0.f);

    CPUTimer timer;
    timer.start();

    projector.set_input(im);
    projector.forward_project(sino);

    timer.stop();
    time += timer.value();
}

static
void
back_project(VoxelsOnCartesianGrid<float> &im, const ProjData &sino, BackProjectorByBinNiftyPET &projector, double &time)
{
    std::cerr << "Back projecting...\n";
    im.fill(0.f);

    CPUTimer timer;
    timer.start();

    projector.start_accumulating_in_new_target();
    projector.back_project(sino);
    projector.get_output(im);

    timer.stop();
    time += timer.value();
}

static float get_inner_product(
        const VoxelsOnCartesianGrid<float> &im1,
        const VoxelsOnCartesianGrid<float> &im2)
{
    return std::inner_product(im1.begin_all(),im1.end_all(),im2.begin_all(),0.f);
}

static float get_inner_product(
        const ProjDataInMemory &proj1,
        const ProjDataInMemory &proj2)
{
    // Get number of elements
    const size_t num_elements = proj1.size_all();

    // Create arrays
    std::vector<float> arr1(num_elements), arr2(num_elements);
    proj1.copy_to(arr1.begin());
    proj2.copy_to(arr2.begin());

    return std::inner_product(arr1.begin(),arr1.end(),arr2.begin(),0.f);
}

static
float
test_inner_product(const VoxelsOnCartesianGrid<float> &im, const ProjData &sino,
                   const VoxelsOnCartesianGrid<float> &im_proj, const ProjData &sino_proj)
{
    std::cerr << "Checking inner products...\n";
    const float inner_product_images = get_inner_product(im,im_proj);
    const float inner_product_sinos  = get_inner_product(sino, sino_proj);

    std::cout << "\tinner product between images    = " << inner_product_images << "\n";
    std::cout << "\tinner product between sinograms = " << inner_product_sinos << "\n";

    if (std::abs(inner_product_images) + std::abs(inner_product_sinos) < 1e-4f) {
        std::cout << "\n\t\tCan't perform adjoint test as both equal zero...\n";
        std::cout << "\t\tmax in input image = " << im.find_max() << "\n";
        std::cout << "\t\tmax in projected image = " << im_proj.find_max() << "\n";
        std::cout << "\t\tmax in input image = " << find_max(sino) << "\n";
        std::cout << "\t\tmax in projected image = " << find_max(sino_proj) << "\n";
        return -1.f;
    }

    float adjoint_test =
            std::abs(inner_product_images - inner_product_sinos) /
            (0.5f * (std::abs(inner_product_images) + std::abs(inner_product_sinos)));
    std::cout << "\t|<x, Ty> - <y, Tsx>| / 0.5*(|<x, Ty>|+|<y, Tsx>|) = " << adjoint_test << "\n";
    return adjoint_test;
}

void
TestGPUProjectors::
test_adjoints()
{
    set_up();

    unsigned num_unsuccessful(0);

    while(_results.size() < _num_attempts) {

        unsigned i = _results.size();

        std::cout << "\nPerforming test " << i+1 << " of " << _num_attempts << "\n";

        // Even iterations, modify the image
        if (i%2==0) {
            get_random_image(*_image_sptr);
            forward_project(*_projected_sino_sptr, *_image_sptr, _projector_fwrd, _time_fwrd);
        }
        // Odd iterations (and first), modify the sinogram
        if (i==0 || i%2==1) {
            get_random_sino(*_sino_sptr);
            back_project(*_projected_image_sptr, *_sino_sptr, _projector_back, _time_back); 
        }

        const float adjoint_test = 
            test_inner_product(*_image_sptr, *_sino_sptr, *_projected_image_sptr, *_projected_sino_sptr);
        if (adjoint_test > 0.f) {
            _results.push_back(adjoint_test);
            std::cout << "\tAvg. test result = " << std::accumulate(_results.begin(), _results.end(), 0.0) /double(i+1) << 
                " (number of tests = " << i+1 << "), avg. time forward projecting = " << _time_fwrd/double(i+1) << " s, " <<
                "avg. time back projecting = " << _time_back/double(i+1) << " s.\n\n";

            // Check the result
            if (adjoint_test > 1e-4f)
                error("Adjoint test greater than threshold, failed!");

            // Reset unsuccessful counter
            num_unsuccessful = 0;
        }
        else {
            ++num_unsuccessful;
            if (num_unsuccessful==5)
                error("Too many (5) unsuccessful comparisons");
        }
    }
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
