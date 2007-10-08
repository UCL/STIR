//
// $Id$
//
/*
    Copyright (C) 2003- $Date$ , Hammersmith Imanet Ltd
    For GE Internal use only
*/
/*!
  \file
  \ingroup motion_test
  \brief Test program for stir::RigidObject3DTransformation functions
  \see stir::RigidObject3DTransformationTests
  \author Kris Thielemans
  \author Sanida Mustafovic
  
  $Date$
  $Revision$
*/
#include "stir/RunTests.h"
#include "local/stir/Quaternion.h"
#include <iostream>
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/shared_ptr.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Scanner.h"
#include "stir/Bin.h"
#include "stir/Coordinate4D.h"
#include "stir/stream.h"
#include "stir/round.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneDensel.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "local/stir/recon_buildblock/ProjMatrixByDenselUsingRayTracing.h"
#include "stir/Densel.h"
//#include "stir/display.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "local/stir/recon_buildblock/DataSymmetriesForDensels_PET_CartesianGrid.h" // necessary for shared_ptr in ProjMatrixElemsForOneDensel.h
#ifdef DO_TIMINGS
#include "stir/CPUTimer.h"
#endif
#include "stir/LORCoordinates.h"
#include "stir/geometry/line_distances.h"
#include "stir/index_at_maximum.h"
//#include "local/stir/motion/Polaris_MT_File.h"
//#include "local/stir/motion/RigidObject3DMotionFromPolaris.h"
#include <algorithm>
#include <cmath>
#ifdef BOOST_NO_STDC_NAMESPACE
namespace std { using ::sqrt; using ::fabs; }
#endif


START_NAMESPACE_STIR


/*! \ingroup motion_test
  \brief Various tests on RigidObject3DTransformation

  Indirectly tests ProjDataInfo::get_bin and ProjDataInfo::get_LOR.

  Tests include checks on 
  - compose() and inverse()
  - if LORs through a voxel have a close distance to the central 
    centre of the voxel (before and after transformation),
  - consistency between moving a voxel and moving an LOR using the following
    procedure: forward project voxel, move LOR, backproject voxel, 
    find location of maximum, compare this
    with moving the voxel directly.

  \todo tests on inverse of transform_bin fail with some rotations (too large 
  difference in round-trip).
*/
class RigidObject3DTransformationTests: public RunTests
{
public:  
  void run_tests();
private:
  void test_transform_bin();
  void test_transform_bin_with_inverse(const ProjDataInfo& proj_data_info, const bool is_uncompressed);
  void test_transform_bin_vs_transform_point(const shared_ptr<ProjDataInfo>& proj_data_info_sptr);
  void test_find_closest_transformation();
};

void
RigidObject3DTransformationTests::run_tests()
{
  std::cerr << "\nTesting RigidObject3DTransformation\n";

  std::cerr << "\tTesting inverse and check if it's a rigid transformation\n";
  {
    Quaternion<float> quat(1,-2,3,8);
    quat.normalise();
    const CartesianCoordinate3D<float> translation(111,-12,152);
    
    const RigidObject3DTransformation ro3dtrans(quat, translation);
    
    RigidObject3DTransformation ro3dtrans_inverse =ro3dtrans;
    ro3dtrans_inverse =ro3dtrans_inverse.inverse();
    {
      const Quaternion<float> quat_original = ro3dtrans.get_quaternion();
      const Quaternion<float> quat_inverse = ro3dtrans_inverse.get_quaternion();
      
      const Quaternion<float> unity = quat_original * quat_inverse;
      
      check_if_equal(unity[1], 1.F, "test on inverse quat -- scalar");
      check_if_equal(unity[2], 0.F, "test on inverse quat -- vector1");
      check_if_equal(unity[3], 0.F, "test on inverse quat -- vector2");
      check_if_equal(unity[4], 0.F, "test on inverse quat -- vector3");
    }      

    
    const CartesianCoordinate3D<float> transformed_origin =
      ro3dtrans.transform_point(CartesianCoordinate3D<float>(0,0,0));
    for (int i=0; i<100; ++i)
    {
      const CartesianCoordinate3D<float> point(210.F*i,-55.F-i,2.F+2*i);
      const CartesianCoordinate3D<float> transformed_point =ro3dtrans.transform_point(point);
      //Testing norm of the original and transformed point 
      {
	const float original_distance = norm(point /* - origin*/);
	const float transformed_distance = norm(transformed_point-transformed_origin);
	check_if_equal(original_distance, transformed_distance, "test on distance to see if it's a rigid transformation");
      }
      // Testing to see if inverse gets us back
      {
	
	const CartesianCoordinate3D<float> transformed_back_point =
	    ro3dtrans_inverse.transform_point(transformed_point);
	// compare with original by checking norm of difference
	// divide by norm(point) such that we're looking at a relative measure
	check_if_zero(norm(point-transformed_back_point)/norm(point), 
	              "test on inverse transformation of transformed point");
	
      }
    }
  }

#if 0
  std::cerr << "\n\tTesting reading of mt files" <<std::endl;
  
  const string fdef_filename = "H09990.fdef";
  TimeFrameDefinitions tfdef(fdef_filename);
  const float polaris_time_offset =  3241;
  const string mt_filename = "H09990.mt";
  
  shared_ptr<Polaris_MT_File> mt_file_ptr = 
    new Polaris_MT_File(mt_filename);
  RigidObject3DMotionFromPolaris ro3dmfromp(mt_filename,mt_file_ptr);
  ro3dmfromp.set_polaris_time_offset(polaris_time_offset);
  
  int number_of_frames = tfdef.get_num_frames();
  for ( int frame = 1; frame<=number_of_frames;frame++)
  {
    float start = tfdef.get_start_time(frame);
    float end  = tfdef.get_end_time(frame);
    
    RigidObject3DTransformation ro3dtrans;
    
    ro3dmfromp.get_motion(ro3dtrans,start); //+polaris_time_offset);
    const Quaternion<float> quat_s = ro3dtrans.get_quaternion();
    const CartesianCoordinate3D<float> trans_s =ro3dtrans.get_translation();
    std::cerr << " Quaternion is " << quat_s << std::endl;
    std::cerr << " Translation is " << trans_s << std::endl;
    int i = 1;
    while (i<= end)
    {
      RigidObject3DTransformation ro3dtrans_test;
      ro3dmfromp.get_motion(ro3dtrans_test,start+i);
      const Quaternion<float> quat = ro3dtrans.get_quaternion();
      const CartesianCoordinate3D<float> trans =ro3dtrans.get_translation();
      
      check_if_equal( quat_s[1], quat[1],"test on -scalar");
      check_if_equal( quat_s[2], quat[2],"test on -vector-1");
      check_if_equal( quat_s[3], quat[3],"test on -vector-2");
      check_if_equal( quat_s[4], quat[4],"test on -vector-3");
      i+=100;
    }  
  }
#endif
  std::cerr << "\n\tTesting compose " << std::endl;
  {
    Quaternion<float> quat_1(1,-2,3,8);
    quat_1.normalise();
    const CartesianCoordinate3D<float> translation_1(111,-12,152);    
    const RigidObject3DTransformation ro3dtrans_1(quat_1, translation_1);
    
    Quaternion<float> quat_2(1,-3,12,4);
    quat_2.normalise();
    const CartesianCoordinate3D<float> translation_2(1,-54,12);    
    const RigidObject3DTransformation ro3dtrans_2(quat_2, translation_2);
    
    Quaternion<float> quat_3(2,-7,24,1);
    quat_3.normalise();
    const CartesianCoordinate3D<float> translation_3(9,4,34);
    const RigidObject3DTransformation ro3dtrans_3(quat_3, translation_3);
#ifdef DO_TIMINGS    
    CPUTimer timer;
    timer.reset();
    CPUTimer compose_timer;
    compose_timer.reset();
#endif
    const RigidObject3DTransformation composed_ro3dtrans1=
      compose(ro3dtrans_3,
              compose(ro3dtrans_2,ro3dtrans_1));
    
    for (int i=0; i<1000; ++i)
    {
      const CartesianCoordinate3D<float> point(210.F*i,-55.F-i,2.F+2*i);
#ifdef DO_TIMINGS
      timer.start();
#endif
      const CartesianCoordinate3D<float> transformed_point_3 =
	ro3dtrans_3.
	transform_point(ro3dtrans_2.
	                transform_point(
	                                ro3dtrans_1.
	                                transform_point(point)));
#ifdef DO_TIMINGS      
      timer.stop();
      
      compose_timer.start();
#endif            
      const CartesianCoordinate3D<float> transformed_point_composed =
	composed_ro3dtrans1.transform_point(point);
#ifdef DO_TIMINGS
      compose_timer.stop();
#endif
      check_if_zero(norm(transformed_point_3-transformed_point_composed)/norm(transformed_point_3),
	            "test on compose");
    }
#ifdef DO_TIMINGS
    std::cerr << " Individual multiplications: " <<  timer.value() << " s CPU time"<<std::endl;
    std::cerr << "Combined: " <<  compose_timer.value() << " s CPU time"<<std::endl;
#endif
  }

  test_find_closest_transformation();

  test_transform_bin();
}



void 
RigidObject3DTransformationTests::
test_transform_bin()
{
    shared_ptr<Scanner> scanner_ptr = new Scanner(Scanner::E953);
    // we make the scanner longer to avoid problems with rotations 
    // (almost) orthogonal to the scanner axis. Otherwise most
    // LORs would be transformed out of the scanner, and we won't get
    // a get intersection of the backprojections
    scanner_ptr->set_num_rings(40); 
    std::cerr << "\n\tTests with proj_data_info without mashing and axial compression, no arc-correction\n";
    shared_ptr<ProjDataInfo> proj_data_info_sptr =
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				    /*span*/1, scanner_ptr->get_num_rings()-1,
				    /*views*/ scanner_ptr->get_num_detectors_per_ring()/2, 
				    /*tang_pos*/scanner_ptr->get_num_detectors_per_ring()/2, 
				    /*arc_corrected*/ false);
    ProjDataInfoCylindricalNoArcCorr& proj_data_info =
      dynamic_cast<ProjDataInfoCylindricalNoArcCorr &>(*proj_data_info_sptr);

    test_transform_bin_with_inverse(proj_data_info, /*is_uncompressed=*/true);
    // TODO ProjMatrixByDensel cannot do span=1 yet 
    // test_transform_bin_vs_transform_point(proj_data_info_sptr);
#ifdef NEW_ROT
    // old get_bin() cannot handle spanned data, so tests disabled ifndef NEW_ROT
    std::cerr << "\n\tTests with proj_data_info with mashing and axial compression, no arc-correction\n";
    proj_data_info_sptr =
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				    /*span*/3, scanner_ptr->get_num_rings()-1,
				    /*views*/ scanner_ptr->get_num_detectors_per_ring()/6, 
				    /*tang_pos*/scanner_ptr->get_num_detectors_per_ring()/3, 
				    /*arc_corrected*/ false);
    test_transform_bin_with_inverse(*proj_data_info_sptr,/*is_uncompressed=*/false);
    test_transform_bin_vs_transform_point(proj_data_info_sptr);

    std::cerr << "\n\tTests with proj_data_info without mashing and axial compression, arc-correction\n";
    proj_data_info_sptr =
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				    /*span*/1, scanner_ptr->get_num_rings()-1,
				    /*views*/ scanner_ptr->get_num_detectors_per_ring()/2, 
				    /*tang_pos*/scanner_ptr->get_num_detectors_per_ring()/2, 
				    /*arc_corrected*/ true);
    test_transform_bin_with_inverse(*proj_data_info_sptr,/*is_uncompressed=*/true);
    // TODO ProjMatrixByDensel cannot do span=1 yet 
    // test_transform_bin_vs_transform_point(proj_data_info_sptr);

    std::cerr << "\n\tTests with proj_data_info with mashing and axial compression, arc-correction\n";
    proj_data_info_sptr =
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				    /*span*/3, scanner_ptr->get_num_rings()-1,
				    /*views*/ scanner_ptr->get_num_detectors_per_ring()/6, 
				    /*tang_pos*/scanner_ptr->get_num_detectors_per_ring()/3, 
				    /*arc_corrected*/ true);
    test_transform_bin_with_inverse(*proj_data_info_sptr,/*is_uncompressed=*/false);
    test_transform_bin_vs_transform_point(proj_data_info_sptr);
#endif

  }

// first define some functions to check 'closeness' of 2 bins
static Bin
swap_direction(const Bin& bin, const int num_views)
{
  return Bin(-bin.segment_num(), 
	     bin.view_num() < num_views/2? bin.view_num()+ num_views : bin.view_num() - num_views,
	     bin.axial_pos_num(), 
	     -bin.tangential_pos_num());
}

#if 0

// attempt to check if neigbouring bins.
// failed because it's tricky to check axial_pos in the spanned case.
static Bin
bin_diff_no_reorder(const Bin& org_bin, const Bin& transformed_bin, const float ax_poss_offset)
{
  Bin diff;
  diff.segment_num() =
    (org_bin.segment_num() - transformed_bin.segment_num());
  diff.view_num() = 
    (org_bin.view_num() - transformed_bin.view_num());
  // this didn't work
  diff.axial_pos_num() = 
    (org_bin.axial_pos_num() - transformed_bin.axial_pos_num() - std::abs(round(ax_poss_offset*diff.segment_num())));
  diff.tangential_pos_num() = 
    (org_bin.tangential_pos_num() - transformed_bin.tangential_pos_num());
  return diff;
}


static int 
sup_norm(const Bin& bin)
{
  return std::max(abs(bin.segment_num()), 
	     std::max(abs(bin.view_num()),
		 std::max(abs(bin.axial_pos_num()), 
		     abs(bin.tangential_pos_num()))));
}

static Bin
bin_diff(const Bin& org_bin, const Bin& transformed_bin, const int num_views, const float ax_poss_offset)
{
  const Bin diff1=bin_diff_no_reorder(org_bin, transformed_bin, ax_poss_offset);
  const Bin diff2=bin_diff_no_reorder(org_bin, swap_direction(transformed_bin, num_views), ax_poss_offset);
  return 
    sup_norm(diff1)<sup_norm(diff2) 
    ? diff1
    : diff2;
}

#else

static Coordinate4D<float>
bin_diff_no_reorder(const Bin& org_bin, const Bin& transformed_bin, const ProjDataInfo& proj_data_info)
{
  Coordinate4D<float> diff;
  diff[1] =
    (org_bin.segment_num() - transformed_bin.segment_num());
  diff[2] =
    (org_bin.view_num() - transformed_bin.view_num());
  diff[3] =
    (proj_data_info.get_m(org_bin) -
     proj_data_info.get_m(transformed_bin))/
    proj_data_info.get_sampling_in_m(org_bin);
  diff[4] =
    (proj_data_info.get_s(org_bin) -
     proj_data_info.get_s(transformed_bin))/
    proj_data_info.get_sampling_in_s(org_bin);
  return diff;
}

static float 
sup_norm(const Coordinate4D<float>& c)
{
  return std::max(std::fabs(c[1]),
		  std::max(std::fabs(c[2]),
			   std::max(std::fabs(c[3]),
				    std::fabs(c[4]))));
}

static Coordinate4D<float>
bin_diff(const Bin& org_bin, const Bin& transformed_bin, const ProjDataInfo& proj_data_info)
{
  const Coordinate4D<float> diff1=bin_diff_no_reorder(org_bin, transformed_bin, proj_data_info);
  const Coordinate4D<float> diff2=bin_diff_no_reorder(org_bin, swap_direction(transformed_bin, proj_data_info.get_num_views()), proj_data_info);
  return 
    sup_norm(diff1)<sup_norm(diff2) 
    ? diff1
    : diff2;
}
#endif

void
RigidObject3DTransformationTests::
test_transform_bin_with_inverse(const ProjDataInfo& proj_data_info, const bool is_uncompressed)
{
  std::cerr <<"\n\t\ttesting transform_bin and inverse()\n";
  // std::cerr <<proj_data_info.parameter_info();
  // take some arbitrary (but not too large) rotation which is not along any of the axes
  Quaternion<float> quat(1.F,0.05F,-.03F,.1F);
  quat.normalise();
  const CartesianCoordinate3D<float> translation(11,-12,15);
    
  const RigidObject3DTransformation ro3dtrans(quat, translation);
  const RigidObject3DTransformation ro3dtrans_inverse = ro3dtrans.inverse();

  unsigned num_bins_checked = 0;
  unsigned num_bins_in_range = 0;
  Coordinate4D<float> max_diff(0,0,0,0);
#ifdef NEW_ROT
  const float allowed_diff =
    is_uncompressed ? 1.1F : 1.1F;
#else
  const float allowed_diff =
    2.1F;
#endif

  for (int segment_num=proj_data_info.get_min_segment_num();
       segment_num<=proj_data_info.get_max_segment_num();
       ++segment_num)
    {
      for (int view_num=proj_data_info.get_min_view_num();
	   view_num<=proj_data_info.get_max_view_num();
	   view_num+=5)
	{
	  // loop over axial_positions. Avoid using first and last position, as 
	  // the discretisation error can easily bring the transformed_bin back
	  // outside the range. We could test for that, but it would make
	  // the code much more complicated, and not give anything useful back.
	  for (int axial_pos_num=proj_data_info.get_min_axial_pos_num(segment_num)+1;
	       axial_pos_num<=proj_data_info.get_max_axial_pos_num(segment_num)-1;
	       axial_pos_num+=3)
	    {
	      for (int tangential_pos_num=proj_data_info.get_min_tangential_pos_num()+1;
		   tangential_pos_num<=proj_data_info.get_max_tangential_pos_num()-1;
		   tangential_pos_num+=17)
		{
		  ++num_bins_checked;

		  const Bin org_bin(segment_num,view_num,axial_pos_num,tangential_pos_num, /* value*/1);
		  Bin transformed_bin = org_bin;
		  ro3dtrans.transform_bin(transformed_bin, proj_data_info, proj_data_info);
	    
		  if (transformed_bin.get_bin_value()>0) // only check when the transformed_bin is within the range
		    {
		      ro3dtrans_inverse.transform_bin(transformed_bin, proj_data_info, proj_data_info);
		      // the inverse might still take it outside the range unfortunately
		      if (transformed_bin.get_bin_value()<=0)
			continue;

		      ++num_bins_in_range;

		      const Coordinate4D<float> diff = bin_diff(org_bin, transformed_bin, proj_data_info);
		      if (sup_norm(diff)>sup_norm(max_diff))
			max_diff = diff;
			
		      if (!check(org_bin.get_bin_value() == transformed_bin.get_bin_value(), "transform_bin_with_inverse: value") ||
			  !check(sup_norm(diff)<=allowed_diff, "transform_bin_with_inverse: different bin"))
			{
			  std::cerr << "\tProblem at  segment=" << org_bin.segment_num() 
				    << ", axial pos=" << org_bin.axial_pos_num()
				    << ", view=" << org_bin.view_num() 
				    << ", tangential_pos=" << org_bin.tangential_pos_num() 
				    << '\n';
			  std::cerr << "\tround-trip to  segment=" << transformed_bin.segment_num() 
				    << ", axial pos=" << transformed_bin.axial_pos_num()
				    << ", view= " << transformed_bin.view_num() 
				    << ", tangential_pos=" << transformed_bin.tangential_pos_num() 
				    << " value=" << transformed_bin.get_bin_value()
				    << " sup_norm(diff)=" << sup_norm(diff)
				    <<'\n';
			}
		    }
		} // tangential_pos
	    } // axial_pos
	} // view
    } //segment

  std::cerr << "\t\t" << num_bins_checked << " num_bins checked, " << num_bins_in_range << " transformed in range\n"
	    << "\t\tmax deviation (sup_norm):"
	    << sup_norm(max_diff) 
	    << '\n';
}

void
RigidObject3DTransformationTests::
test_transform_bin_vs_transform_point(const shared_ptr<ProjDataInfo>& proj_data_info_sptr)
{
  std::cerr << "\n\t\ttesting consistency transform_point and transform_bin\n";

  shared_ptr<DiscretisedDensity<3,float> > density_sptr =
    new VoxelsOnCartesianGrid<float> (*proj_data_info_sptr);
  const CartesianCoordinate3D<float> origin =
    density_sptr->get_origin();
  const CartesianCoordinate3D<float> voxel_size =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float> const&>(*density_sptr).get_grid_spacing();
  // somewhat arbitrary: not a lot more than a voxel
  const float allowed_deviation=static_cast<float>(norm(voxel_size)/std::sqrt(3.)*1.9);

  Quaternion<float> quat(.7F,.2F,-.1F,.1F);
  quat.normalise();
  const CartesianCoordinate3D<float> translation(-11,-12,15);
    
  const RigidObject3DTransformation ro3dtrans(quat, translation);
    
  //RigidObject3DTransformation ro3dtrans_inverse =ro3dtrans;
  //ro3dtrans_inverse =ro3dtrans_inverse.inverse();

    
  ProjMatrixByDenselUsingRayTracing pm_by_densel;
  pm_by_densel.set_up(proj_data_info_sptr, density_sptr);
  ProjMatrixByBinUsingRayTracing pm_by_bin;
  pm_by_bin.set_up(proj_data_info_sptr, density_sptr);

  double max_deviation = 0;
  double max_deviation_transformed = 0;
  double max_deviation_org = 0;
  {
    ProjMatrixElemsForOneDensel bins;
    ProjMatrixElemsForOneBin lor;
      
    const Densel densel((density_sptr->get_min_index()+density_sptr->get_max_index())/2+5,5,10);
    const CartesianCoordinate3D<float> densel_coord =
      BasicCoordinate<3,float>(densel) * voxel_size + origin;
    const CartesianCoordinate3D<float> transformed_densel_coord =
      ro3dtrans.transform_point(densel_coord);
    const CartesianCoordinate3D<float> transformed_densel_float =
      (transformed_densel_coord - origin)/voxel_size;
    const CartesianCoordinate3D<float> image_centre =
      CartesianCoordinate3D<float>((density_sptr->get_min_index()+density_sptr->get_max_index())/2.F,0,0)
      * voxel_size;

    density_sptr->fill(0);

    pm_by_densel.get_proj_matrix_elems_for_one_densel(bins, densel);

    unsigned num_contributing_bins=0;
    LORInAxialAndNoArcCorrSinogramCoordinates<float> lorNAC;
    LORAs2Points<float> lor2pts;
    const double ring_radius =
      static_cast<double>(proj_data_info_sptr->get_scanner_ptr()->get_effective_ring_radius());
    for (ProjMatrixElemsForOneDensel::const_iterator bin_iter = bins.begin();
	 bin_iter != bins.end();
	 ++bin_iter)
      {
	{
	  proj_data_info_sptr->get_LOR(lorNAC, *bin_iter);
	  lorNAC.get_intersections_with_cylinder(lor2pts, ring_radius);
	  const float deviation_org =
	    distance_between_line_and_point(lor2pts,  
					    CartesianCoordinate3D<float>(densel_coord-image_centre));
	  if (max_deviation_org < deviation_org)
	    max_deviation_org = deviation_org;
	}
        Bin transformed_bin = *bin_iter;
	ro3dtrans.transform_bin(transformed_bin, *proj_data_info_sptr, *proj_data_info_sptr);
	if (transformed_bin.get_bin_value()>0)
	  {
	    ++num_contributing_bins;

	    {
	      proj_data_info_sptr->get_LOR(lorNAC, transformed_bin);
	      lorNAC.get_intersections_with_cylinder(lor2pts, ring_radius);
	      const float deviation_transformed =
		distance_between_line_and_point(lor2pts, 
						CartesianCoordinate3D<float>(transformed_densel_coord-image_centre));
	      if (max_deviation_transformed < deviation_transformed)
		max_deviation_transformed = deviation_transformed;
	    }

	    transformed_bin.set_bin_value(bin_iter->get_bin_value());
	    pm_by_bin.get_proj_matrix_elems_for_one_bin(lor, transformed_bin);
	    lor.back_project(*density_sptr, transformed_bin);
	  }
      }
    check(num_contributing_bins>30,
	  "num_contributing_bins should be large enough for the back-projection test to make sense");
    std::cerr << "\t\tnum_contributing_bins " << num_contributing_bins 
	 << " out of " << bins.end() - bins.begin() << '\n';
    const CartesianCoordinate3D<int> densel_from_bins =
      indices_at_maximum(*density_sptr);

    const double deviation = 
      norm(BasicCoordinate<3,float>(densel_from_bins) - transformed_densel_float);
    if (max_deviation < deviation)
      max_deviation = deviation;
    //display(*density_sptr,density_sptr->find_max());
    if (!check(deviation<allowed_deviation,"deviation determined via backprojection"))
      {
	std::cerr << "Org: " << densel << " transformed: " << transformed_densel_float << " by bin: " << densel_from_bins << "\n"
	     << "\t(all are in index-units, not in mm)\n";
	(*density_sptr)[round(transformed_densel_float)] =
	  density_sptr->find_max()*2;
	OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
	  write_to_file("STIRImage", *density_sptr);
	std::cerr << "Image written as STIRImage.* (with transformed densel at twice max value)\n";
      }
    check(max_deviation_org<allowed_deviation,"deviation of pixel with original LOR is too large");
      {
	std::cerr << "\t\tdeviation of pixel with original LOR (in mm): " << max_deviation_org << '\n';
      }
    check(max_deviation_transformed<allowed_deviation,"deviation of pixel with LOR after transformations is too large");
      {
	std::cerr << "\t\tdeviation of pixel with  LOR after transformations (in mm): " << max_deviation_transformed << '\n';
      }

  }
  std::cerr << "\t\tmax deviation via backprojection (in index units) : " << max_deviation << '\n';

}

void
RigidObject3DTransformationTests::
test_find_closest_transformation()
{
  std::cerr << "\n\tTests for find_closest_transformation\n";

  // set fairly low tolerance as we don't find the closest transformation to very high precision
  const double old_tolerance = this->get_tolerance();
  this->set_tolerance(.01);

  Quaternion<float> quat(.7F,.2F,-.1F,.15F);
  quat.normalise();
  const CartesianCoordinate3D<float> translation(-11,-12,15);    
  const RigidObject3DTransformation transformation(quat, translation);

  std::vector<CartesianCoordinate3D<float> > points;
  points.push_back(CartesianCoordinate3D<float>(1,2,3));
  points.push_back(CartesianCoordinate3D<float>(1,3,-3));
  points.push_back(CartesianCoordinate3D<float>(-1.4F,2,6));
  points.push_back(CartesianCoordinate3D<float>(1,2,33.3F));

  std::vector<CartesianCoordinate3D<float> > transformed_points;
  for (std::vector<CartesianCoordinate3D<float> >::const_iterator iter = points.begin();
       iter != points.end();
       ++iter)
    transformed_points.push_back(transformation.transform_point(*iter));

  check_if_zero(RigidObject3DTransformation::RMSE(transformation, points.begin(), points.end(), transformed_points.begin()),
		"RMSE for exact match is too large");

  RigidObject3DTransformation result;
  if (!check(
	    RigidObject3DTransformation::
	    find_closest_transformation(result,
					points.begin(), points.end(), transformed_points.begin(),
					Quaternion<float>(1,0,0,0)) ==
	    Succeeded::yes,
	    "find_closest_transformation for exact match returned Succeeded::no"))
    return;
  if (!check_if_zero(norm(quat-result.get_quaternion()), 
		     "find_closest_transformation for exact match: non-matching quaternion") ||
      !check_if_zero(norm(translation-result.get_translation()), 
		     "find_closest_transformation for exact match: non-matching translation"))
    {
      std::cerr << "Transformation found: " << result
		<< "\nbut should have been " << transformation
		<< "\nInput was:\nPoints : \n" << points
		<< "Transformed points : \n" << transformed_points
		<< '\n';
    }

  check_if_zero(RigidObject3DTransformation::RMSE(result, points.begin(), points.end(), transformed_points.begin()),
		"find_closest_transformation for exact match: RMSE is too large");

  this->set_tolerance(old_tolerance);
}

END_NAMESPACE_STIR

int main()
{
  stir::RigidObject3DTransformationTests tests;
  tests.run_tests();
  
  return tests.main_return_value();
}
