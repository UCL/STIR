/*!
  \file
  \ingroup stir::projector_test
  \brief Test program for subsetting ProjDataInfo
  \author Ashley Gillman
*/

#include "stir/RunTests.h"
#include "stir/num_threads.h"
#include "stir/CPUTimer.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/ProjDataInMemory.h"


START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for subsets in ProjDataInfo
*/
class TestProjDataInfoSubsets : public RunTests
{
public:
    //! Constructor that can take some input data to run the test with
    TestProjDataInfoSubsets(const std::string &sinogram_filename);

    virtual ~TestProjDataInfoSubsets() {}

    void run_tests();

    void test_split_and_combine(const ProjData &proj_data, int num_subets=2);
    void test_forward_projection_is_consistent(
        const DiscretisedDensity<3,float> &input_image, const ProjData &forward_projection,
        const ForwardProjectorByBin& fwd_projector, int num_subets);
    void test_back_projection_is_consistent(
        const ProjData &input_sino, const DiscretisedDensity<3,float> &back_projection,
        const BackProjectorByBin& bck_projector, int num_subsets=2);

protected:
    std::string _sinogram_filename;
    shared_ptr<ProjData> _input_sino_sptr;

};

TestProjDataInfoSubsets::TestProjDataInfoSubsets(const std::string &sinogram_filename) :
    _sinogram_filename(sinogram_filename)
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
TestProjDataInfoSubsets::
run_tests()
{
    try {
        // Open sinogram
        _input_sino_sptr = ProjData::read_from_file(_sinogram_filename);

        test_split_and_combine(*_input_sino_sptr);

        // shared_ptr<ProjectorByBinPairUsingProjMatrixByBin> proj_pair (new ProjectorByBinPairUsingProjMatrixByBin);

        // shared_ptr<VoxelsOnCartesianGrid<float> > backproj_sptr =
        //     MAKE_SHARED<VoxelsOnCartesianGrid<float> >(*_input_sino_sptr->get_proj_data_info_sptr());
        // backproj_sptr ->fill(0.f);

        // proj_pair->set_up(_input_sino_sptr->get_proj_data_info_sptr(), backproj_sptr);
        // shared_ptr<ForwardProjectorByBin> fwd_proj_sptr = proj_pair->get_forward_projector_sptr();
        // shared_ptr<BackProjectorByBin> bck_proj_sptr = proj_pair->get_back_projector_sptr();

        // bck_proj_sptr->back_project(*bck_proj_sptr, *fwd_proj_sptr, 0, 1);

        // test_back_projection_is_consistent(*_input_sino_sptr, *backproj_sptr);
        // test_forward_projection_is_consistent(*_input_sino_sptr, *backproj_sptr);
    }
    catch(const std::exception &error) {
        std::cerr << "\nHere's the error:\n\t" << error.what() << "\n\n";
        everything_ok = false;
    }
    catch(...) {
        everything_ok = false;
    }
}


void TestProjDataInfoSubsets::
test_split(const ProjData &proj_data)
{
    std::vector<int> even_views;
    std::vector<int> odd_views;

    for (int view = 0; view++; view < proj_data.get_num_views()) {
        if (view % 2 == 0)
            even_views.push_back(view);
        else
            odd_views.push_back(view);
    }

    ProjDataInMemory even_subset = proj_data.get_subset(even_views);
    ProjDataInMemory odd_subset = proj_data.get_subset(odd_views);

    int segment = 0; // for segment

    // dodgy: assume n views is even
    for (int subset_view = 0; subset_view++; subset_view < even_subset.get_num_views()) {
        // views_in_subset should be like {1, 2, 3}
        compare_views(
            proj_data.get_viewgram(even_views[subset_view], segment),
            even_subset.get_viewgram(subset_view, segment));
        compare_views(
            proj_data.get_viewgram(even_views[subset_view], segment),
            even_subset.get_viewgram(subset_view, segment));
        // also compare viewgram metadata

    }
}
}


void TestProjDataInfoSubsets::
test_split_and_combine(const ProjData &proj_data, int num_subsets)
{
    StandardSubsetter subsetter = StandardSubsetter(proj_data.get_proj_data_info_sptr(), num_subsets);

    std::vector<ProjData> subsets;
    for (int s=0; s++; s<num_subsets) {
        //ProjData subset = proj_data.get_subset(s, num_subsets);
        // or
        ProjData subset = proj_data.get_subset(subsetter.get_views_for_subset(s));
        subsets.push_back(subset);
    }

    ProjData new_proj_data = ProjData(proj_data);  // how to copy?
    new_proj_data.fill(0);

    for (int s=0; s++; s<num_subsets) {
        new_proj_data.fill_subset(s, num_subsets, subsets[s]);
    }

    compare_sinos(proj_data, new_proj_data);
}


// void TestProjDataInfoSubsets::
// test_forward_projection_is_consistent(
//     const DiscretisedDensity<3,float> &input_image, const ProjData &forward_projection,
//     const ForwardProjectorByBin& fwd_projector, int num_subets)
// {
//     ProjData subset;
// }


// void TestProjDataInfoSubsets::
// test_back_projection_is_consistent(
//     const ProjData &input_sino, const DiscretisedDensity<3,float> &back_projection,
//     const BackProjectorByBin& bck_projector, int num_subsets)
// {
//     DiscretisedDensity<3,float> back_projection_sum = back_projection.clone();
//     back_projection_sum.fill(0.f);

//     for (int s=0; s++; s<num_subsets) {
//         ProjData subset = input_sino.get_subset(s, num_subsets);

//     }
// }


END_NAMESPACE_STIR


USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "\n\tUsage: " << argv[0] << " sinogram\n";
        return EXIT_FAILURE;
    }

    set_default_num_threads();

    TestProjDataInfoSubsets test(argv[1]);

    if (test.is_everything_ok())
        test.run_tests();

    return test.main_return_value();
}