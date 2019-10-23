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

#include "stir/recon_buildblock/niftypet_projector/ForwardProjectorByBinNiftyPET.h"
#include "stir/recon_buildblock/niftypet_projector/BackProjectorByBinNiftyPET.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/RunTests.h"
#include "stir/num_threads.h"
#include "stir/CPUTimer.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include <fstream>

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
  TestGPUProjectors()
      : _save_results(false) {}
  /// Destructor
  virtual ~TestGPUProjectors() {}
  /// Set save results
  void set_save_results(const bool save_results) { _save_results = save_results; }
  /// Set span
  void set_span(const int span) { _span = span; }
  /// Set sinogram filename
  void set_sinogram_filename(const std::string &sinogram_filename) { _sinogram_filename = sinogram_filename; }
  /// Set image filename
  void set_image_filename(const std::string &image_filename) { _image_filename = image_filename; }
  /// Run tests
  void run_tests();
protected:
  /// Set up the input sinogram to fwd project. If filename given, read it. Else create an mMR sino
  void set_up_input_sino();
  /// Set up the input image to back project. If filename given, read it. Else create an mMR image
  void set_up_input_image();
  /// Do the projections
  void run_projections();
  bool _save_results;
  int _span;
  std::string _image_filename;
  std::string _sinogram_filename;
  shared_ptr<DiscretisedDensity<3, float> > _image_sptr;
  shared_ptr<ProjDataInMemory> _proj_data_sptr;
};

static
Succeeded
compare_arrays(const std::vector<float> &vec1, const std::vector<float> &vec2)
{
    // Subtract
    std::vector<float> diff = vec1;
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), diff.begin(), std::minus<float>());

    // Get max difference
    const float max_diff = *std::max_element(diff.begin(), diff.end());

    // Get Euclidean distance with diff^2, accumulate and then sqrt
    std::vector<float> diff_sq = vec1;
    std::transform(diff.begin(), diff.end(), diff_sq.begin(), [](float f)->float { return f*f; });
    float sum_diff_sq = std::accumulate(diff_sq.begin(),diff_sq.end(),0.f);
    float euclidean_dist = sqrt(sum_diff_sq);

    std::cout << "Min array 1 / array 2 = " << *std::min_element(vec1.begin(),vec1.end()) << " / " << *std::min_element(vec2.begin(),vec2.end()) << "\n";
    std::cout << "Max array 1 / array 2 = " << *std::max_element(vec1.begin(),vec1.end()) << " / " << *std::max_element(vec2.begin(),vec2.end()) << "\n";
    std::cout << "Sum array 1 / array 2 = " <<  std::accumulate(vec1.begin(),vec1.end(),0.f) << " / " << std::accumulate(vec2.begin(),vec2.end(),0.f)      << "\n";
    std::cout << "Max diff = " << max_diff << "\n";
    std::cout << "Euclidean distance = " << euclidean_dist << "\n\n";

    return (std::abs(max_diff) < 1e-3f ? Succeeded::yes : Succeeded::no);
}

static
void
compare_images(bool &everything_ok, const DiscretisedDensity<3,float> &im_1, const DiscretisedDensity<3,float> &im_2)
{
    std::cout << "\nComparing images...\n";

    if (!im_1.has_same_characteristics(im_2)) {
        std::cout << "\nImages have different characteristics!\n";
        everything_ok = false;
    }

    Coordinate3D<int> min_indices, max_indices;

    im_1.get_regular_range(min_indices, max_indices);
    unsigned num_elements = 1;
    for (int i=0; i<3; ++i)
        num_elements *= unsigned(max_indices[i + 1] - min_indices[i + 1] + 1);

    std::vector<float> arr_1(num_elements), arr_2(num_elements);

    DiscretisedDensity<3,float>::const_full_iterator im_1_iter = im_1.begin_all_const();
    DiscretisedDensity<3,float>::const_full_iterator im_2_iter = im_2.begin_all_const();
    std::vector<float>::iterator arr_1_iter = arr_1.begin();
    std::vector<float>::iterator arr_2_iter = arr_2.begin();
    while (im_1_iter!=im_1.end_all_const()) {
        *arr_1_iter = *im_1_iter;
        *arr_2_iter = *im_2_iter;
        ++im_1_iter;
        ++im_2_iter;
        ++arr_1_iter;
        ++arr_2_iter;
    }

    // Compare values
    if (compare_arrays(arr_1,arr_2) == Succeeded::yes)
        std::cout << "Images match!\n";
    else {
        std::cout << "Images don't match!\n";
        everything_ok = false;
    }
}

static
void
compare_sinos(bool &everything_ok, const ProjData &proj_data_1, const ProjData &proj_data_2)
{
    std::cout << "\nComparing sinograms...\n";

    if (*proj_data_1.get_proj_data_info_sptr() != *proj_data_2.get_proj_data_info_sptr()) {
        std::cout << "\nSinogram proj data info don't match\n";
        everything_ok = false;
    }

    int min_segment_num = proj_data_1.get_min_segment_num();
    int max_segment_num = proj_data_1.get_max_segment_num();

    // Get number of elements
    unsigned num_elements(0);
    for (int segment_num = min_segment_num; segment_num<= max_segment_num; ++segment_num)
        num_elements += unsigned(proj_data_1.get_max_axial_pos_num(segment_num) - proj_data_1.get_min_axial_pos_num(segment_num)) + 1;
    num_elements *= unsigned(proj_data_1.get_max_view_num() - proj_data_1.get_min_view_num()) + 1;
    num_elements *= unsigned(proj_data_1.get_max_tangential_pos_num() - proj_data_1.get_min_tangential_pos_num()) + 1;

    // Create arrays
    std::vector<float> arr_1(num_elements), arr_2(num_elements);
    proj_data_1.copy_to(arr_1.begin());
    proj_data_2.copy_to(arr_2.begin());

    // Compare values
    if (compare_arrays(arr_1,arr_2) == Succeeded::yes)
        std::cout << "Sinograms match!\n";
    else {
        std::cout << "Sinograms don't match!\n";
        everything_ok = false;
    }
}

void project(double &time, shared_ptr<ProjDataInMemory> &sino_sptr, shared_ptr<DiscretisedDensity<3,float> > &image_sptr, const shared_ptr<const ProjDataInMemory> input_sino_sptr, const shared_ptr<const DiscretisedDensity<3,float> > input_image_sptr, ForwardProjectorByBin &fwrd_projector, BackProjectorByBin &back_projector, const std::string &name, const bool save_to_file)
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

    if (save_to_file) {
        sino_sptr->write_to_file(name + "_forward_projected.hs");
        shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > > output_file_format_sptr =
                OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr();
        output_file_format_sptr->write_to_file(name + "_back_projected",*image_sptr);
    }
}

inline bool file_exists (const std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}

void
TestGPUProjectors::
set_up_input_sino()
{
    if (!_sinogram_filename.empty()) {
        _proj_data_sptr = MAKE_SHARED<ProjDataInMemory>(*ProjData::read_from_file(_sinogram_filename));
    }
    else {
        // Create scanner
        shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::Siemens_mMR));

        // ExamInfo
        shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
        exam_info_sptr->imaging_modality = ImagingModality::PT;

        // ProjDataInfo
        int max_seg_num;
        if (_span == 11)
            max_seg_num = 5;
        else if (_span == 1)
            max_seg_num = 0;
        else
            throw std::runtime_error("Span of 1 and 11 only currently supported.");

        VectorWithOffset<int> ax_pos_per_segment(-max_seg_num,max_seg_num);
        ax_pos_per_segment[0] = 127;
        if (max_seg_num > 0) {
            ax_pos_per_segment[-1] = ax_pos_per_segment[1] = 115;
            ax_pos_per_segment[-2] = ax_pos_per_segment[2] =  93;
            ax_pos_per_segment[-3] = ax_pos_per_segment[3] =  71;
            ax_pos_per_segment[-4] = ax_pos_per_segment[4] =  49;
            ax_pos_per_segment[-5] = ax_pos_per_segment[5] =  27;
        }

        VectorWithOffset<int> max_ring_diff(-max_seg_num,max_seg_num);
        if (max_seg_num > 0) {
            max_ring_diff[-5] = -50;
            max_ring_diff[-4] = -39;
            max_ring_diff[-3] = -28;
            max_ring_diff[-2] = -17;
            max_ring_diff[-1] = -06;
        }
        max_ring_diff[ 0] =  05;
        if (max_seg_num > 0) {
            max_ring_diff[ 1] =  16;
            max_ring_diff[ 2] =  27;
            max_ring_diff[ 3] =  38;
            max_ring_diff[ 4] =  49;
            max_ring_diff[ 5] =  60;
        }

        VectorWithOffset<int> min_ring_diff(-max_seg_num,max_seg_num);
        for (int i=-max_seg_num; i<=max_seg_num; ++i)
            min_ring_diff[i] = -max_ring_diff[-i];

        shared_ptr<ProjDataInfoCylindricalNoArcCorr> proj_data_info_sptr
                (new ProjDataInfoCylindricalNoArcCorr
                 (scanner_sptr,ax_pos_per_segment,min_ring_diff,max_ring_diff,scanner_sptr->get_max_num_views(),344));

        // Create ProjData
        _proj_data_sptr.reset(new ProjDataInMemory(exam_info_sptr,proj_data_info_sptr));

        // Get number of elements
        unsigned num_elements(0);
        for (int segment_num = -max_seg_num; segment_num<= max_seg_num; ++segment_num)
            num_elements += unsigned(_proj_data_sptr->get_max_axial_pos_num(segment_num) - _proj_data_sptr->get_min_axial_pos_num(segment_num)) + 1;
        num_elements *= unsigned(_proj_data_sptr->get_max_view_num() - _proj_data_sptr->get_min_view_num()) + 1;
        num_elements *= unsigned(_proj_data_sptr->get_max_tangential_pos_num() - _proj_data_sptr->get_min_tangential_pos_num()) + 1;

        // Create array
        std::vector<float> arr(num_elements);
        for (unsigned i=0; i<num_elements; ++i)
            arr[i] = float(i);

        // Fill
        _proj_data_sptr->fill_from(arr.begin());
    }
}

void
TestGPUProjectors::
set_up_input_image()
{
    if (!_image_filename.empty()) {
        _image_sptr.reset(DiscretisedDensity<3,float>::read_from_file(_image_filename));
    }
    else {

        const BasicCoordinate<3, int> min_indices(make_coordinate(0,  -160, -160));
        const BasicCoordinate<3, int> max_indices(make_coordinate(126, 159,  159));
        IndexRange<3> range(min_indices,max_indices);

        _image_sptr.reset(new VoxelsOnCartesianGrid<float>(
                              _proj_data_sptr->get_exam_info_sptr(),
                              range,
                              CartesianCoordinate3D<float>(0.f,0.f,0.f),
                              CartesianCoordinate3D<float>(2.03125f, 2.08626f, 2.08626f)
                              ));

        // fill
        float val = 0.f;
        for (int z=min_indices[1]; z<=max_indices[1]; ++z)
            for (int y=min_indices[2]; y<=max_indices[2]; ++y)
                for (int x=min_indices[3]; x<=max_indices[3]; ++x)
                    (*_image_sptr)[z][y][x] = ++val;
    }

    shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > > output_file_format_sptr =
            OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr();
}

void
TestGPUProjectors::
run_projections()
{
    // Set up input sinogram and image
    set_up_input_sino();
    set_up_input_image();

    // Create output images and sinograms
    shared_ptr<DiscretisedDensity<3,float> > gpu_image_sptr, cpu_image_sptr;
    shared_ptr<ProjDataInMemory> gpu_sino_sptr, cpu_sino_sptr;
    double time_gpu, time_cpu;

    // Forward and back project - gpu
    ForwardProjectorByBinNiftyPET gpu_fwrd;
    BackProjectorByBinNiftyPET    gpu_back;
    project(time_gpu, gpu_sino_sptr, gpu_image_sptr, _proj_data_sptr, _image_sptr, gpu_fwrd, gpu_back, "gpu", _save_results);

    // Forward and back project - cpu
    shared_ptr<ProjMatrixByBin> PM_sptr(new  ProjMatrixByBinUsingRayTracing());
    ForwardProjectorByBinUsingProjMatrixByBin cpu_fwrd(PM_sptr);
    BackProjectorByBinUsingProjMatrixByBin    cpu_back(PM_sptr);
    project(time_cpu, cpu_sino_sptr, cpu_image_sptr, _proj_data_sptr, _image_sptr, cpu_fwrd, cpu_back, "cpu", _save_results);

    // comparison
    std::cout << "\nTime for forward and back projection\n";
    std::cout << "\tGPU: " << time_gpu << "\n";
    std::cout << "\tCPU: " << time_cpu << "\n";


    // Compare back projections
    compare_images(everything_ok, *gpu_image_sptr,*cpu_image_sptr);

    // Compare forward projections
    compare_sinos(everything_ok, *gpu_sino_sptr,*cpu_sino_sptr);
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

void print_usage()
{
    cerr << "\n\tUsage: test_ProjectorNiftyPET [-h] span [save_results [sinogram [image]]]\n";
    cerr << "\t\tSpan will only be used when creating a sinogram.\n";
    cerr << "\t\tTo enter your own sinogram, use any dummy value.\n";
    cerr << "\t\tCurrently only support Span 1 or 11.\n";
    cerr << "\t\tSave_results: 0 or 1. Default to 0.\n";
}

int main(int argc, char **argv)
{
    for (int i=0; i<argc; ++i) {
        if (strcmp(argv[i],"-h") ==0) {
            print_usage();
            return EXIT_SUCCESS;
        }
    }

    if (argc < 2 || argc > 5) {
        print_usage();
        return EXIT_FAILURE;
    }

    set_default_num_threads();

    TestGPUProjectors test;

    test.set_span(atoi(argv[1]));

    if (argc > 2)
        if (bool(atoi(argv[1])))
            test.set_save_results(true);
    if (argc > 3)
        test.set_sinogram_filename(argv[3]);
    if (argc > 4)
        test.set_image_filename(argv[4]);

    if (test.is_everything_ok())
        test.run_tests();

    return test.main_return_value();
}
