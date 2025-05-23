/*
    Copyright (C) 2019, University College London - Richard Brown
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0
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
    shared_ptr<ProjData> _input_sino_sptr;
    const std::vector<shared_ptr<DiscretisedDensity<3,float> > > post_data_processor_bck_proj();
    const std::vector<shared_ptr<ProjData> > pre_data_processor_fwd_proj(const DiscretisedDensity<3,float> &input_image);
};

TestDataProcessorProjectors::TestDataProcessorProjectors(const std::string &sinogram_filename, const float fwhm) :
    _sinogram_filename(sinogram_filename),
    _fwhm(fwhm)
{
}

static
Succeeded
compare_arrays(const std::vector<float> &vec1, const std::vector<float> &vec2)
{
    // Subtract
    std::vector<float> diff = vec1;
    std::transform(vec1.begin(), vec1.end(), vec2.begin(), diff.begin(), std::minus<float>());

    // Get max difference
    const float max_diff = *std::max_element(diff.begin(), diff.end());

    std::cout << "Min array 1 / array 2 = " << *std::min_element(vec1.begin(),vec1.end()) << " / " << *std::min_element(vec2.begin(),vec2.end()) << "\n";
    std::cout << "Max array 1 / array 2 = " << *std::max_element(vec1.begin(),vec1.end()) << " / " << *std::max_element(vec2.begin(),vec2.end()) << "\n";
    std::cout << "Sum array 1 / array 2 = " <<  std::accumulate(vec1.begin(),vec1.end(),0.f) << " / " << std::accumulate(vec2.begin(),vec2.end(),0.f)      << "\n";
    std::cout << "Max diff = " << max_diff << "\n\n";

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

void
TestDataProcessorProjectors::
run_tests()
{
    try {
        // Open sinogram
        _input_sino_sptr = ProjData::read_from_file(_sinogram_filename);

        // Back project
        std::cerr << "Tests for post-data-processor back projection\n";
        const std::vector<shared_ptr<DiscretisedDensity<3,float> > > bck_projected_ims =
                this->post_data_processor_bck_proj();

        // Forward project
        std::cerr << "Tests for pre-data-processor forward projection\n";
        const std::vector<shared_ptr<ProjData> > fwd_projected_sinos =
                this->pre_data_processor_fwd_proj(*bck_projected_ims[0]);

        // Compare back projections
        compare_images(everything_ok, *bck_projected_ims[0],*bck_projected_ims[1]);

        // Compare forward projections
        compare_sinos(everything_ok, *fwd_projected_sinos[0],*fwd_projected_sinos[1]);
    }
    catch(const std::exception &error) {
        std::cerr << "\nHere's the error:\n\t" << error.what() << "\n\n";
        everything_ok = false;
    }
    catch(...) {
        everything_ok = false;
    }
}

static
void
get_data_processor(shared_ptr<DataProcessor<DiscretisedDensity<3,float> > > &data_processor_sptr, const float fwhm)
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

static
shared_ptr<BackProjectorByBin>
get_back_projector_via_parser(const float fwhm = -1.f)
{
    std::string buffer;
    std::stringstream parameterstream(buffer);

    parameterstream << "Back Projector parameters:=\n";
    if (fwhm > 0)
        parameterstream
                    << "Post Data Processor := Separable Cartesian Metz\n"
                    << "Separable Cartesian Metz Filter Parameters :=\n"
                    << "  x-dir filter FWHM (in mm):= " << fwhm << "\n"
                    << "  y-dir filter FWHM (in mm):= " << fwhm << "\n"
                    << "  z-dir filter FWHM (in mm):= " << fwhm << "\n"
                    << "  x-dir filter Metz power:= .0\n"
                    << "  y-dir filter Metz power:= .0\n"
                    << "  z-dir filter Metz power:=.0\n"
                    << "END Separable Cartesian Metz Filter Parameters :=\n";
    parameterstream << "End Back Projector Parameters:=\n";

    shared_ptr<ProjMatrixByBin> PM_sptr(new ProjMatrixByBinUsingRayTracing);

    shared_ptr<BackProjectorByBin> back_proj_sptr =
            MAKE_SHARED<BackProjectorByBinUsingProjMatrixByBin>(PM_sptr);
    back_proj_sptr->parse(parameterstream);

    return back_proj_sptr;
}

static
shared_ptr<ForwardProjectorByBin>
get_forward_projector_via_parser(const float fwhm = -1.f)
{
    shared_ptr<ForwardProjectorByBin> fwd_proj;
    std::string buffer;
    std::stringstream parameterstream(buffer);

    parameterstream << "Forward Projector parameters:=\n";
    if (fwhm > 0)
        parameterstream
                    << "Pre Data Processor := Separable Cartesian Metz\n"
                    << "Separable Cartesian Metz Filter Parameters :=\n"
                    << "  x-dir filter FWHM (in mm):= " << fwhm << "\n"
                    << "  y-dir filter FWHM (in mm):= " << fwhm << "\n"
                    << "  z-dir filter FWHM (in mm):= " << fwhm << "\n"
                    << "  x-dir filter Metz power:= .0\n"
                    << "  y-dir filter Metz power:= .0\n"
                    << "  z-dir filter Metz power:=.0\n"
                    << "END Separable Cartesian Metz Filter Parameters :=\n";
    parameterstream << "End Forward Projector Parameters:=\n";

    shared_ptr<ProjMatrixByBin> PM_sptr(new ProjMatrixByBinUsingRayTracing);

    shared_ptr<ForwardProjectorByBin> fwd_proj_sptr =
            MAKE_SHARED<ForwardProjectorByBinUsingProjMatrixByBin>(PM_sptr);
    fwd_proj_sptr->parse(parameterstream);

    return fwd_proj_sptr;
}

const std::vector<shared_ptr<ProjData> >
TestDataProcessorProjectors::
pre_data_processor_fwd_proj(const DiscretisedDensity<3,float> &input_image)
{
    // Create two sinograms, images and forward projectors.
    //    One for pre-data processor forward projection,
    //    the other for data processor then forward projection
    std::vector<shared_ptr<ProjData> > sinos(2);
    std::vector<shared_ptr<DiscretisedDensity<3,float> > > images(2);
    std::vector<shared_ptr<ForwardProjectorByBin> > projectors(2);

    // Loop over twice!
    for (unsigned i=0; i<sinos.size(); ++i) {

        // Copy the sinogram and fill with zeros
        sinos[i] = MAKE_SHARED<ProjDataInMemory>(*_input_sino_sptr);
        sinos[i]->fill(0.f);

        // Copy input image
        images[i].reset(input_image.clone());

        // The first time, use the data processor during the forward projection
        projectors[i] = get_forward_projector_via_parser(i==0 ? _fwhm : -1);


        projectors[i]->set_up(_input_sino_sptr->get_proj_data_info_sptr()->create_shared_clone(),images[i]);

        // The second time, use the data processor before the forward projection
        if (i!=0) {
            // Set up the data processor
            shared_ptr<DataProcessor<DiscretisedDensity<3,float> > > data_processor_sptr;
            get_data_processor(data_processor_sptr, _fwhm);
            data_processor_sptr->apply(*images[i]);
        }

        // Forward project
        projectors[i]->set_input(*images[i]);
        projectors[i]->forward_project(*sinos[i]);
    }

    return sinos;
}

const std::vector<shared_ptr<DiscretisedDensity<3,float> > >
TestDataProcessorProjectors::
post_data_processor_bck_proj()
{
    // Create two images and two back projectors.
    //    One for pre-data processor back projection,
    //    the other for data processor then back projection
    std::vector<shared_ptr<DiscretisedDensity<3,float> > > images(2);
    std::vector<shared_ptr<BackProjectorByBin> > projectors(2);

    // Loop over twice!
    for (unsigned i=0; i<images.size(); ++i) {

        // Copy images and fill with zeros
        images[i] = MAKE_SHARED<VoxelsOnCartesianGrid<float> >(*_input_sino_sptr->get_proj_data_info_sptr());
        images[i]->fill(0.f);

        // The first time, use the data processor during the back projection
        projectors[i] = get_back_projector_via_parser(i==0 ? _fwhm : -1);

        projectors[i]->set_up(_input_sino_sptr->get_proj_data_info_sptr()->create_shared_clone(),images[i]);

        // Back project
        projectors[i]->start_accumulating_in_new_target();
        projectors[i]->back_project(*_input_sino_sptr);
        projectors[i]->get_output(*images[i]);

        // The second time, use the data processor after the back projection
        if (i!=0) {
            // Set up the data processor
            shared_ptr<DataProcessor<DiscretisedDensity<3,float> > > data_processor_sptr;
            get_data_processor(data_processor_sptr, _fwhm);
            data_processor_sptr->apply(*images[i]);
        }
    }

    return images;
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