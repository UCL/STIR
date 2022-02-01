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
// #include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
// #include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/ProjDataInMemory.h"
#include "stir/Viewgram.h"
#include "stir/Scanner.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoSubsetByView.h"
#include "stir/Shape/Ellipsoid.h"
#include "stir/Shape/DiscretisedShape3D.h"
// ^ should we have to include this? Should be included from Ellipsoid? Bug?
#include "stir/Shape/EllipsoidalCylinder.h"
#include "stir/Shape/Ellipsoid.h"
#include "stir/Shape/Box3D.h"
#include "stir/Shape/DiscretisedShape3D.h"

using std::endl;
using std::cerr;
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
    //void test_split_and_combine(const ProjData &proj_data, int num_subsets=2);
    void test_forward_projection_is_consistent(
        const shared_ptr<const VoxelsOnCartesianGrid<float> >& input_image_sptr,
        const shared_ptr<const ProjData>& template_sino_sptr,
        bool use_symmetries, int num_subsets=10);
    void test_forward_projection_is_consistent_with_unbalanced_subset(
        const shared_ptr<const VoxelsOnCartesianGrid<float> >& input_image_sptr,
        const shared_ptr<const ProjData>& template_sino_sptr,
        bool use_symmetries, int num_subsets=10);
    void test_back_projection_is_consistent(
        const ProjData &input_sino, const VoxelsOnCartesianGrid<float> &template_image,
        BackProjectorByBin& bck_projector, int num_subsets=2);

protected:
    std::string _sinogram_filename;
    shared_ptr<ProjData> _input_sino_sptr;
    shared_ptr<VoxelsOnCartesianGrid<float> > _test_image_sptr;
    static shared_ptr<ProjData> construct_test_proj_data();
    static shared_ptr<VoxelsOnCartesianGrid<float> > construct_test_image_data(
        const ProjDataInfo &template_projdatainfo);
    static shared_ptr<VoxelsOnCartesianGrid<float> > construct_projector_pair(
        const ProjDataInfo &template_projdatainfo);
    static shared_ptr<ProjectorByBinPairUsingProjMatrixByBin> construct_projector_pair(
      const shared_ptr<const ProjDataInfo>& template_projdatainfo_sptr,
      const shared_ptr<const VoxelsOnCartesianGrid<float> >& template_image_sptr,
      bool use_symmetries=true);

    ProjDataInMemory generate_full_forward_projection(
      const shared_ptr<const VoxelsOnCartesianGrid<float> >& input_image_sptr,
      const shared_ptr<const ProjData>& template_sino_sptr,
      bool use_symmetries);
    void test_forward_projection_for_one_subset(
      const shared_ptr<const VoxelsOnCartesianGrid<float> >& input_image_sptr,
      ProjDataInMemory& full_forward_projection,
      std::unique_ptr<ProjDataInMemory>& subset_forward_projection_uptr,
      bool use_symmetries);

};

TestProjDataInfoSubsets::TestProjDataInfoSubsets(const std::string &sinogram_filename) :
    _sinogram_filename(sinogram_filename)
{
}

shared_ptr<ProjData> TestProjDataInfoSubsets::construct_test_proj_data()
{
  cerr << "\tGenerating default ProjData from E953" << endl;
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

shared_ptr<VoxelsOnCartesianGrid<float> > TestProjDataInfoSubsets::construct_test_image_data(const ProjDataInfo &template_projdatainfo)
{
  cerr << "\tGenerating default image of Ellipsoid" << endl;
  shared_ptr<VoxelsOnCartesianGrid<float> > image = std::make_shared<VoxelsOnCartesianGrid<float> >(template_projdatainfo);

  // make radius 0.8 FOV
  auto radius = BasicCoordinate<3,float>(image->get_lengths()) * image->get_voxel_size() / 2.F;
    //image->get_physical_coordinates_for_indices(image->get_min_indices()) * 0.8;
  auto centre = image->get_physical_coordinates_for_indices((image->get_min_indices() + image->get_max_indices()) / 2);

  // object at centre of image
  Ellipsoid ellipsoid(radius, centre);

  ellipsoid.construct_volume(*image, Coordinate3D<int>(1,1,1));

  cerr << boost::format("\t Generated ellipsoid image, min=%f, max=%f") % image->find_min() % image->find_max() << endl;
  return image;
}

shared_ptr<ProjectorByBinPairUsingProjMatrixByBin> TestProjDataInfoSubsets::construct_projector_pair(
    const shared_ptr<const ProjDataInfo>& template_projdatainfo_sptr,
    const shared_ptr<const VoxelsOnCartesianGrid<float> >& template_image_sptr,
    bool use_symmetries)
{
  cerr << "\tSetting up default projector pair, ProjectorByBinPairUsingProjMatrixByBin" << endl;
  auto proj_matrix_sptr = std::make_shared<ProjMatrixByBinUsingRayTracing>();
  proj_matrix_sptr->set_do_symmetry_180degrees_min_phi(use_symmetries);
  proj_matrix_sptr->set_do_symmetry_90degrees_min_phi(use_symmetries);
  proj_matrix_sptr->set_do_symmetry_shift_z(use_symmetries);
  proj_matrix_sptr->set_do_symmetry_swap_s(use_symmetries);
  proj_matrix_sptr->set_do_symmetry_swap_segment(use_symmetries);
  auto proj_pair_sptr = std::make_shared<ProjectorByBinPairUsingProjMatrixByBin>(proj_matrix_sptr);

  proj_pair_sptr->set_up(template_projdatainfo_sptr, template_image_sptr);
  return proj_pair_sptr;
}
    
void
TestProjDataInfoSubsets::
run_tests()
{
  cerr << "-------- Testing ProjDataInfoCylindricalArcCorr --------\n";
  try
    {
      // Open sinogram
      if (_sinogram_filename.empty())
        _input_sino_sptr = construct_test_proj_data();
      else
        _input_sino_sptr = ProjData::read_from_file(_sinogram_filename);

      // make an image (TODO: or open?)
      _test_image_sptr = construct_test_image_data(*_input_sino_sptr->get_proj_data_info_sptr());

      shared_ptr<ProjectorByBinPair> proj_pair
        = construct_projector_pair(_input_sino_sptr->get_proj_data_info_sptr(), _test_image_sptr);

      test_split(*_input_sino_sptr);

      // test_split_and_combine(*_input_sino_sptr);

      test_forward_projection_is_consistent(
        _test_image_sptr, _input_sino_sptr, /*use_symmetries =*/false);
      cerr << "\trepeat with an 'unusual' number of subsets, 13" << endl;
      test_forward_projection_is_consistent(
        _test_image_sptr, _input_sino_sptr, /*use_symmetries =*/false, /*sum_subsets =*/ 13);

      test_forward_projection_is_consistent_with_unbalanced_subset(
        _test_image_sptr, _input_sino_sptr, /*use_symmetries =*/false);

      // test_back_projection_is_consistent(*_input_sino_sptr, *back_proj_sptr);
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
  cerr << "\tTesting ability to split a ProjData into consistent subsets" << endl;
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

    auto subset_proj_data_uptr = proj_data.get_subset(subset_views);
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


void TestProjDataInfoSubsets::
test_forward_projection_is_consistent(
    const shared_ptr<const VoxelsOnCartesianGrid<float> >& input_image_sptr,
    const shared_ptr<const ProjData>& template_sino_sptr,
    bool use_symmetries, int num_subsets)
{
  cerr << "\tTesting Subset forward projection is consistent" << endl;

  auto full_forward_projection = generate_full_forward_projection(
    input_image_sptr, template_sino_sptr, use_symmetries);


  //ProjData subset;
  for (int subset_n = 0; subset_n < num_subsets; ++subset_n) {
    std::vector<int> subset_views;
    int view = subset_n;

    while(view < full_forward_projection.get_num_views()){
      subset_views.push_back(view);
      view += num_subsets;
    }
    auto subset_forward_projection_uptr = full_forward_projection.get_subset(subset_views);

    test_forward_projection_for_one_subset(
      input_image_sptr, full_forward_projection, subset_forward_projection_uptr, use_symmetries);
  }
}


void TestProjDataInfoSubsets::
test_forward_projection_is_consistent_with_unbalanced_subset(
    const shared_ptr<const VoxelsOnCartesianGrid<float> >& input_image_sptr,
    const shared_ptr<const ProjData>& template_sino_sptr,
    bool use_symmetries, int num_subsets)
{
  cerr << "\tTesting Subset forward projection is consistent with unbalanced subset" << endl;

  if (num_subsets >= template_sino_sptr->get_num_views()) {
    cerr << "Error: Template provided doesn't have enough views to conduct this test with "
         << num_subsets << " subsets." << std::endl;
    everything_ok = false;
  }

  auto full_forward_projection = generate_full_forward_projection(
    input_image_sptr, template_sino_sptr, use_symmetries);


  for (int subset_n = 0; subset_n < num_subsets; ++subset_n) {
    // subset 0 to subset num_subsets-2 get 1 the view
    // final subset (num_subsets-1) gets the remainder
    std::vector<int> subset_views;

    if (subset_n < num_subsets - 1) {
        subset_views.push_back(subset_n);
    }
    else {
      for (unsigned int view = num_subsets - 1; view < full_forward_projection.get_num_views(); view++) {
        subset_views.push_back(view);
      }
    }
    auto subset_forward_projection_uptr = full_forward_projection.get_subset(subset_views);

    cerr << "\tTesting unbalanced subset " << subset_n << ": views " << subset_views << endl;
    test_forward_projection_for_one_subset(
      input_image_sptr, full_forward_projection, subset_forward_projection_uptr, use_symmetries);
  }
}

ProjDataInMemory TestProjDataInfoSubsets::
generate_full_forward_projection(
    const shared_ptr<const VoxelsOnCartesianGrid<float> >& input_image_sptr,
    const shared_ptr<const ProjData>& template_sino_sptr,
    bool use_symmetries)
{
  if (input_image_sptr->find_max() == 0) {
    cerr << "Tests are run with redundant empty image" << endl;
    everything_ok = false;
  }

  auto fwd_projector_sptr = construct_projector_pair(
    template_sino_sptr->get_proj_data_info_sptr(), input_image_sptr, use_symmetries)
      ->get_forward_projector_sptr();
  
  auto full_forward_projection = ProjDataInMemory(*template_sino_sptr);
  fwd_projector_sptr->set_input(*input_image_sptr);
  fwd_projector_sptr->forward_project(full_forward_projection);

  if (full_forward_projection.get_viewgram(0, 0).find_max() == 0) {
    cerr << "segment 0, view 0 of reference forward projection is empty!" << endl;
    everything_ok = false;
  }

  return full_forward_projection;
}


void TestProjDataInfoSubsets::
test_forward_projection_for_one_subset(
  const shared_ptr<const VoxelsOnCartesianGrid<float> >& input_image_sptr,
  ProjDataInMemory& full_forward_projection,
  std::unique_ptr<ProjDataInMemory>& subset_forward_projection_uptr,
  bool use_symmetries)
{
  auto subset_proj_data_info_sptr = std::static_pointer_cast<const ProjDataInfoSubsetByView>(
    subset_forward_projection_uptr->get_proj_data_info_sptr());

  auto subset_proj_pair_sptr = construct_projector_pair(
    subset_forward_projection_uptr->get_proj_data_info_sptr(), input_image_sptr, use_symmetries);

  subset_proj_pair_sptr->get_forward_projector_sptr()->forward_project(*subset_forward_projection_uptr);

  auto subset_views = subset_proj_data_info_sptr->get_org_views();

  // loop over views in the subset data and compare them against the original "full" data
  for(std::size_t i = 0; i < subset_views.size(); ++i)
    {
      // i runs from 0, 1, ... views_in_subset - 1 and indicates the view number in the subset
      // the corresponding view in the original data is at subset_views[i]

      // loop over all segments to check viewgram for all segments
      for (int segment_num = full_forward_projection.get_min_segment_num(); segment_num < full_forward_projection.get_max_segment_num(); ++segment_num)
        {
          check_if_equal(
                          full_forward_projection.get_viewgram(subset_views[i], segment_num),
                          subset_forward_projection_uptr->get_viewgram(i, segment_num), "Are viewgrams equal?");
          // TODO also compare viewgram metadata
        }
    }
}


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
    test.run_tests();

    return test.main_return_value();
}
