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
#include "stir/Scanner.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"

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
    static shared_ptr<ProjData> construct_test_proj_data();

};

TestProjDataInfoSubsets::TestProjDataInfoSubsets(const std::string &sinogram_filename) :
    _sinogram_filename(sinogram_filename)
{
}

shared_ptr<ProjData> TestProjDataInfoSubsets::construct_test_proj_data()
{
  shared_ptr<Scanner> scanner_ptr(new Scanner(Scanner::E953));
  shared_ptr<ProjDataInfo>
    proj_data_info_sptr(ProjDataInfo::construct_proj_data_info(scanner_ptr,
                                                               /*span*/5, scanner_ptr->get_num_rings()-1,
                                                               /*views*/ scanner_ptr->get_num_detectors_per_ring()/2/8, 
                                                               /*tang_pos*/64, 
                                                               /*arc_corrected*/ false));
  shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
  return std::make_shared<ProjDataInMemory>(exam_info_sptr, proj_data_info_sptr);
}
    
void
TestProjDataInfoSubsets::
run_tests()
{
    try
      {
        // Open sinogram
        if (_sinogram_filename.empty())
          _input_sino_sptr = construct_test_proj_data();
        else
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
  int view;    
  int num_subsets = 4;

  std::vector<int> subset_views;

  for (int subset = 0; subset < num_subsets; ++subset){
    // create a vector containg every num_subsest-th view starting at subset
    // for num_subsets = 4 and subset = 0 this is [0, 4, 8, 12, ...]
    // for num_subsets = 4 and subset = 1 this is [1, 5, 9, 13, ...]

    subset_views.clear();
    view = subset;

    while(view < proj_data.get_num_views()){
      subset_views.push_back(view);
      view += num_subsets;
    }

    auto subset_proj_data_uptr = std::move(proj_data.get_subset(subset_views));
    auto& subset_proj_data = *subset_proj_data_uptr;

    // loop over views in the subset data and compare them against the original "full" data
    for(std::size_t i = 0; i < subset_views.size(); ++i)
      {
        // i runs from 0, 1, ... views_in_subset - 1 and indicates the view number in the subset
        // the corresponding view in the original data is at subset_views[i]

        // loop over all segments to check viewgram for all segments
        for (int segment_num = proj_data.get_min_segment_num(); segment_num < proj_data.get_max_segment_num(); ++segment_num)
          {
            check_if_equal(
                           proj_data.get_viewgram(subset_views[i], segment_num),
                           subset_proj_data.get_viewgram(i, segment_num), "Are viewgrams equal?");
            // TODO also compare viewgram metadata
          }
      }
  }      
}


// void TestProjDataInfoSubsets::
// test_split_and_combine(const ProjData &proj_data, int num_subsets)
// {
//     StandardSubsetter subsetter = StandardSubsetter(proj_data.get_proj_data_info_sptr(), num_subsets);

//     std::vector<ProjData> subsets;
//     for (int s=0; s++; s<num_subsets) {
//         //ProjData& subset = *proj_data.get_subset(s, num_subsets);
//         // or
//         ProjData& subset = *proj_data.get_subset(subsetter.get_views_for_subset(s));
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
//         ProjData& subset = *input_sino.get_subset(s, num_subsets);

//     }
// }


END_NAMESPACE_STIR


USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
    if (argc > 2) {
        std::cerr << "\n\tUsage: " << argv[0] << " [projdata_filename]\n";
        return EXIT_FAILURE;
    }

    set_default_num_threads();

    TestProjDataInfoSubsets test(argc>1? argv[1] : "");

    if (test.is_everything_ok())
        test.run_tests();

    return test.main_return_value();
}
