//
// $Id$
//
/*!
  \file
  \ingroup test
  \brief Test program for RigidObject3DTransformation functions
  \author Kris Thielemans
  \author Sanida Mustafovic
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$ , Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/RunTests.h"
#include "local/stir/Quaternion.h"
#include <iostream>
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "local/stir/listmode/TimeFrameDefinitions.h"
#include "stir/shared_ptr.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Scanner.h"
#include "stir/Bin.h"
#include "stir/stream.h"
#include "stir/IO/DefaultOutputFileFormat.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneDensel.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "local/stir/recon_buildblock/ProjMatrixByDenselUsingRayTracing.h"
#include "stir/Densel.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "local/stir/recon_buildblock/DataSymmetriesForDensels_PET_CartesianGrid.h" // necessary for shared_ptr in ProjMatrixElemsForOneDensel.h
#ifdef DO_TIMINGS
#include "stir/CPUTimer.h"
#endif

//#include "local/stir/motion/Polaris_MT_File.h"
//#include "local/stir/motion/RigidObject3DMotionFromPolaris.h"
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif


START_NAMESPACE_STIR

// TODO generalise and move to somewhere else
// (note: copied from find_fwhm)
template<int num_dimensions,class elemT>                         
static                    
BasicCoordinate<num_dimensions,int> 
index_at_maximum(const Array<num_dimensions,elemT>& input_array)
{
  const elemT current_maximum = input_array.find_max();
  BasicCoordinate<num_dimensions,int>  max_location, min_index, max_index; 
  
  bool found=false;    
  min_index[1] = input_array.get_min_index();
  max_index[1] = input_array.get_max_index();
	for ( int k = min_index[1]; k<= max_index[1] && !found; ++k)
	{
	  min_index[2] = input_array[k].get_min_index();
	  max_index[2] = input_array[k].get_max_index();
	  for ( int j = min_index[2]; j<= max_index[2] && !found; ++j)
	  {
	    min_index[3] = input_array[k][j].get_min_index();
	    max_index[3] = input_array[k][j].get_max_index();
	    for ( int i = min_index[3]; i<= max_index[3] && !found; ++i)
	      {
		if (input_array[k][j][i] == current_maximum)
		   {
		     max_location[1] = k;
		     max_location[2] = j;
		     max_location[3] = i;
		   }
	      }
	  }
	}
  found = true;		
  return max_location;	
}                            

// TODO move somewhere else (pinched from Scatter.inl)
template<int num_dimensions>
static inline 
BasicCoordinate<num_dimensions,float> 
convert_int_to_float(const BasicCoordinate<num_dimensions,int>& cint)
{	  
  BasicCoordinate<num_dimensions,float> cfloat;
  
  for(int i=1;i<=num_dimensions;++i)
    cfloat[i]=(float)cint[i];
	  return cfloat;
}

class RigidObject3DTransformationTests: public RunTests
{
public:  
  void run_tests();
private:
  void test_transform_bin_with_inverse(const ProjDataInfo& proj_data_info);
  void test_transform_bin_vs_transform_point(const shared_ptr<ProjDataInfo>& proj_data_info_sptr);

};

void
RigidObject3DTransformationTests::run_tests()
{
  // testing inverse
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

    
    for (int i=0; i<1000; ++i)
    {
      const CartesianCoordinate3D<float> point(210.F*i,-55.F-i,2.F+2*i);
      const CartesianCoordinate3D<float> transformed_point =ro3dtrans.transform_point(point);
      //Testing norm of the original and transformed point 
      {
	const float norm_original = sqrt(square(point.z()) +square(point.y())+square(point.x()));
	const float norm_transformed = sqrt(square(point.z()) +square(point.y())+square(point.x()));
	
	check_if_equal(norm_original, norm_transformed, "test on norm");
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
  cerr << "Testing reading of mt files" <<endl;
  
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
    cerr << " Quaternion is " << quat_s << endl;
    cerr << " Translation is " << trans_s << endl;
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
  // cerr << " Testing compose " << endl;
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
    cerr << " Individual multiplications: " <<  timer.value() << " s CPU time"<<endl;
    cerr << "Combined: " <<  compose_timer.value() << " s CPU time"<<endl;
#endif
  }

  // tests using transform_bin
  {
    shared_ptr<Scanner> scanner_ptr = new Scanner(Scanner::E953);
    // we make the scanner longer to avoid problems with rotations 
    // (almost) orthogonal to the scanner axis. Otherwise most
    // LORs would be transformed out of the scanner, and we won't get
    // a get intersection of the backprojections
    scanner_ptr->set_num_rings(40); 
    cerr << "\nTests with proj_data_info without mashing and axial compression, no arc-correction\n";
    shared_ptr<ProjDataInfo> proj_data_info_sptr =
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				    /*span*/1, scanner_ptr->get_num_rings()-1,
				    /*views*/ scanner_ptr->get_num_detectors_per_ring()/2, 
				    /*tang_pos*/scanner_ptr->get_num_detectors_per_ring()/2, 
				    /*arc_corrected*/ false);
    ProjDataInfoCylindricalNoArcCorr& proj_data_info =
      dynamic_cast<ProjDataInfoCylindricalNoArcCorr &>(*proj_data_info_sptr);

    test_transform_bin_with_inverse(proj_data_info);
    // TODO ProjMatrixByDensel cannot do span=1 yet 
    // test_transform_bin_vs_transform_point(proj_data_info_sptr);
#ifdef NEW_ROT
    cerr << "\nTests with proj_data_info with mashing and axial compression, no arc-correction\n";
    proj_data_info_sptr =
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				    /*span*/3, scanner_ptr->get_num_rings()-1,
				    /*views*/ scanner_ptr->get_num_detectors_per_ring()/6, 
				    /*tang_pos*/scanner_ptr->get_num_detectors_per_ring()/4, 
				    /*arc_corrected*/ false);
    test_transform_bin_with_inverse(*proj_data_info_sptr);
    test_transform_bin_vs_transform_point(proj_data_info_sptr);

    cerr << "\nTests with proj_data_info without mashing and axial compression, arc-correction\n";
    proj_data_info_sptr =
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				    /*span*/1, scanner_ptr->get_num_rings()-1,
				    /*views*/ scanner_ptr->get_num_detectors_per_ring()/2, 
				    /*tang_pos*/scanner_ptr->get_num_detectors_per_ring()/2, 
				    /*arc_corrected*/ true);
    test_transform_bin_with_inverse(*proj_data_info_sptr);
    // TODO ProjMatrixByDensel cannot do span=1 yet 
    // test_transform_bin_vs_transform_point(proj_data_info_sptr);

    cerr << "\nTests with proj_data_info with mashing and axial compression, arc-correction\n";
    proj_data_info_sptr =
      ProjDataInfo::ProjDataInfoCTI(scanner_ptr,
				    /*span*/3, scanner_ptr->get_num_rings()-1,
				    /*views*/ scanner_ptr->get_num_detectors_per_ring()/6, 
				    /*tang_pos*/scanner_ptr->get_num_detectors_per_ring()/4, 
				    /*arc_corrected*/ true);
    test_transform_bin_with_inverse(*proj_data_info_sptr);
    test_transform_bin_vs_transform_point(proj_data_info_sptr);
#endif

  }

}

static Bin
abs_bin_diff_no_reorder(const Bin& org_bin, const Bin& transformed_bin)
{
  Bin diff;
  diff.segment_num() =
    std::abs(org_bin.segment_num() - transformed_bin.segment_num());
  diff.view_num() = 
    std::abs(org_bin.view_num() - transformed_bin.view_num());
  diff.axial_pos_num() = 
    std::abs(org_bin.axial_pos_num() - transformed_bin.axial_pos_num());
  diff.tangential_pos_num() = 
    std::abs(org_bin.tangential_pos_num() - transformed_bin.tangential_pos_num());
  return diff;
}

static Bin
swap_direction(const Bin& bin, const int num_views)
{
  return Bin(-bin.segment_num(), 
	     bin.view_num() < num_views/2? bin.view_num()+ num_views : bin.view_num() - num_views,
	     bin.axial_pos_num(), 
	     -bin.tangential_pos_num());
}

static int 
sup_norm(const Bin& bin)
{
  return max(abs(bin.segment_num()), 
	     max(abs(bin.view_num()),
		 max(abs(bin.axial_pos_num()), 
		     abs(bin.tangential_pos_num()))));
}

static Bin
abs_bin_diff(const Bin& org_bin, const Bin& transformed_bin, const int num_views)
{
  const Bin diff1=abs_bin_diff_no_reorder(org_bin, transformed_bin);
  const Bin diff2=abs_bin_diff_no_reorder(org_bin, swap_direction(transformed_bin, num_views));
  return 
    sup_norm(diff1)<sup_norm(diff2) 
    ? diff1
    : diff2;
}


void
RigidObject3DTransformationTests::
test_transform_bin_with_inverse(const ProjDataInfo& proj_data_info)
{
  cerr <<"\ttesting transform_bin and inverse()\n";
  Quaternion<float> quat(1,-2,3,8);
  quat.normalise();
  const CartesianCoordinate3D<float> translation(11,-12,15);
    
  const RigidObject3DTransformation ro3dtrans;//KT(quat, translation);
    
  RigidObject3DTransformation ro3dtrans_inverse =ro3dtrans;
  ro3dtrans_inverse =ro3dtrans_inverse.inverse();

  unsigned num_bins_checked = 0;
  Bin max_diff(0,0,0,0);
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
#ifndef NEW_ROT
		  {
		    const ProjDataInfoCylindricalNoArcCorr& proj_data_info_na =
		      dynamic_cast<const ProjDataInfoCylindricalNoArcCorr &>(proj_data_info);
		    ro3dtrans.transform_bin(transformed_bin, proj_data_info_na, proj_data_info_na);
		  }
#else
		  ro3dtrans.transform_bin(transformed_bin, proj_data_info, proj_data_info);
#endif
	    
		  if (transformed_bin.get_bin_value()>0) // only check when the transformed_bin is within the range
		    {
#ifndef NEW_ROT
		      {
			const ProjDataInfoCylindricalNoArcCorr& proj_data_info_na =
			  dynamic_cast<const ProjDataInfoCylindricalNoArcCorr &>(proj_data_info);
			ro3dtrans_inverse.transform_bin(transformed_bin, proj_data_info_na, proj_data_info_na);
		      }
#else
		      ro3dtrans_inverse.transform_bin(transformed_bin, proj_data_info, proj_data_info);
#endif
		      const Bin diff = abs_bin_diff(org_bin, transformed_bin, proj_data_info.get_num_views());
		      if (transformed_bin.get_bin_value()>0)
			{
			  if (sup_norm(diff)>sup_norm(max_diff))
			    max_diff = diff;
			}
			
		      if (!check(org_bin.get_bin_value() == transformed_bin.get_bin_value(), "transform_bin_with_inverse: value") ||
			  !check(sup_norm(diff)<3, "transform_bin_with_inverse: different bin"))
			{
			  cerr << "\tProblem at  segment = " << org_bin.segment_num() 
			       << ", axial pos " << org_bin.axial_pos_num()
			       << ", view = " << org_bin.view_num() 
			       << ", tangential_pos_num = " << org_bin.tangential_pos_num() << "\n";
			  if (transformed_bin.get_bin_value()>0)
			    cerr << "round-trip to  segment = " << transformed_bin.segment_num() 
				 << ", axial pos " << transformed_bin.axial_pos_num()
				 << ", view = " << transformed_bin.view_num() 
				 << ", tangential_pos_num = " << transformed_bin.tangential_pos_num() 
				 << " value=" << transformed_bin.get_bin_value()
				 <<"\n";
			}
		    }
		} // tangential_pos
	    } // axial_pos
	} // view
    } //segment
  cerr << '\t' << num_bins_checked << " num_bins checked\n\tMax deviation:\n"
       << "\tsegment = " << max_diff.segment_num ()
       << ", axial pos " << max_diff.axial_pos_num()
       << ", view = " << max_diff.view_num() 
       << ", tangential_pos_num = " << max_diff.tangential_pos_num() << "\n";
}

void
RigidObject3DTransformationTests::
test_transform_bin_vs_transform_point(const shared_ptr<ProjDataInfo>& proj_data_info_sptr)
{
  cerr << "\n\ttesting consistency transform_point and transform_bin\n";

  shared_ptr<DiscretisedDensity<3,float> > density_sptr =
    new VoxelsOnCartesianGrid<float> (*proj_data_info_sptr);
  const CartesianCoordinate3D<float> origin =
    density_sptr->get_origin();
  const CartesianCoordinate3D<float> voxel_size =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float> const&>(*density_sptr).get_grid_spacing();
    
  Quaternion<float> quat(.7F,.2F,.4F,.3F);
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
  {
    ProjMatrixElemsForOneDensel bins;
    ProjMatrixElemsForOneBin lor;
      
    const Densel densel((density_sptr->get_min_index()+density_sptr->get_max_index())/2+5,5,10);
    pm_by_densel.get_proj_matrix_elems_for_one_densel(bins, densel);

    unsigned num_contributing_bins=0;
    for (ProjMatrixElemsForOneDensel::const_iterator bin_iter = bins.begin();
	 bin_iter != bins.end();
	 ++bin_iter)
      {
	Bin transformed_bin = *bin_iter;
#ifndef NEW_ROT
	{
	  ProjDataInfoCylindricalNoArcCorr& proj_data_info =
	    dynamic_cast<ProjDataInfoCylindricalNoArcCorr &>(*proj_data_info_sptr);
	  ro3dtrans.transform_bin(transformed_bin, proj_data_info, proj_data_info);
	}
#else
	ro3dtrans.transform_bin(transformed_bin, *proj_data_info_sptr, *proj_data_info_sptr);
#endif
	if (transformed_bin.get_bin_value()>0)
	  {
	    ++num_contributing_bins;
	    transformed_bin.set_bin_value(bin_iter->get_bin_value());
	    pm_by_bin.get_proj_matrix_elems_for_one_bin(lor, transformed_bin);
	    lor.back_project(*density_sptr, transformed_bin);
	  }
      }
    cerr << "num_contributing_bins " << num_contributing_bins 
	 << " out of " << bins.end() - bins.begin() << '\n';
    const CartesianCoordinate3D<int> densel_from_bins =
      index_at_maximum(*density_sptr);

    const CartesianCoordinate3D<float> densel_coord =
      convert_int_to_float(densel) * voxel_size + origin;
    const CartesianCoordinate3D<float> transformed_densel_coord =
      ro3dtrans.transform_point(densel_coord);
    const CartesianCoordinate3D<float> transformed_densel_float =
      (transformed_densel_coord - origin)/voxel_size;
    const double deviation = 
      norm(convert_int_to_float(densel_from_bins) - transformed_densel_float);
    if (max_deviation < deviation)
      max_deviation = deviation;
	cerr << "Org: " << densel << " transformed: " << transformed_densel_float << "by bin: " << densel_from_bins << "\n";
    if (!check(deviation<1.1,"deviation of pixel"))
      {
	cerr << "Org: " << densel << " transformed: " << transformed_densel_float << "by bin: " << densel_from_bins << "\n";
	DefaultOutputFileFormat output_file_format;
	output_file_format.write_to_file("STIRImage", *density_sptr);
	cerr << "Image written as STIRImage.*\n";

      }
  }
  cerr << "\tmax deviation : " << max_deviation << '\n';

}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  RigidObject3DTransformationTests tests;
  tests.run_tests();
  
  return tests.main_return_value();
}
