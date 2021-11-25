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
#include "stir/Viewgram.h"


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

    void test_split(const ProjData &proj_data);
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


void
TestProjDataInfoSubsets::
run_tests()
{
    try {
        // Open sinogram
        _input_sino_sptr = ProjData::read_from_file(_sinogram_filename);

        test_split(*_input_sino_sptr);
        // test_split_and_combine(*_input_sino_sptr);

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

    for (int view = 0; view < proj_data.get_num_views(); ++ view)
      {
        if (view % 2 == 0)
            even_views.push_back(view);
        else
            odd_views.push_back(view);
      }

    ProjDataInMemory even_subset = proj_data.get_subset(even_views);
    ProjDataInMemory odd_subset = proj_data.get_subset(odd_views);

    int segment = 0; // for segment

    // dodgy: assume n views is even
    for (int subset_view = 0; subset_view < even_subset.get_num_views(); subset_view++)
      {
        check_if_equal(
            proj_data.get_viewgram(even_views[subset_view], segment),
            even_subset.get_viewgram(subset_view, segment),
            "Are viewgrams equal?");
      }
}


// void TestProjDataInfoSubsets::
// test_split_and_combine(const ProjData &proj_data, int num_subsets)
// {
//     StandardSubsetter subsetter = StandardSubsetter(proj_data.get_proj_data_info_sptr(), num_subsets);

//     std::vector<ProjData> subsets;
//     for (int s=0; s++; s<num_subsets) {
//         //ProjData subset = proj_data.get_subset(s, num_subsets);
//         // or
//         ProjData subset = proj_data.get_subset(subsetter.get_views_for_subset(s));
//         subsets.push_back(subset);
//     }

//     ProjData new_proj_data = ProjData(proj_data);  // how to copy?
//     new_proj_data.fill(0);

//     for (int s=0; s++; s<num_subsets) {
//         new_proj_data.fill_subset(s, num_subsets, subsets[s]);
//     }

//     compare_sinos(proj_data, new_proj_data);
// }


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
