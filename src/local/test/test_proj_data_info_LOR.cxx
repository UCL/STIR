//
// $Id$
//
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    For GE internal use only.
*/
/*!

  \file
  \brief Test program to check consistency between projectors and LOR->detectors functions

  Tests detector points and LOR functions by checking intersections of LORs

  \warning Preliminary. Needs spanned but unmashed template projection data.
  \warning When trying different scanners/dimensions/voxels, run only in debug mode to catch asserts. 
  \warning for some voxels, the round-trip in bin coordinates does not work 
  (it ends up in the wrong plane). This is because the current axial-coordinate
   handling in this test is unsafe.
   \todo test stir::ProjDataInfo::get_LOR and stir::ProjDataInfo::get_bin
   \todo use stir::RunTests
  \author Sanida Mustafovic
  \author Kris Thielemans

  $Date$
  $Revision$
*/
#include "local/stir/recon_buildblock/ProjMatrixByDenselUsingRayTracing.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfo.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Bin.h"
#include "stir/stream.h"
#ifndef STIR_NO_NAMESPACES
using std::cout;
using std::endl;
#endif

#include "stir/LORCoordinates.h"
#include "stir/geometry/line_distances.h"

int
main(int argc,char *argv[])
{
 USING_NAMESPACE_STIR

  if(argc<=2 || (argc-1)%3!=0) 
  {
    cerr<<"Usage: " << argv[0] << "  z0 y0 x0 [ z1 y1 x1 [...]]\n";
    exit(EXIT_FAILURE);
  }
 --argc; ++argv; // skip program name

    // TODO ProjMatrixByDensel cannot do span=1 yet 
    // test_transform_bin_vs_transform_point(proj_data_info_sptr);
    cerr << "\nTests with proj_data_info without mashing and but with axial compression, no arc-correction\n";
    shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E962));
    shared_ptr<ProjDataInfo> proj_data_info_sptr =
      ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
				    /*span*/3, 1,// can only handle segment 0 right now
				    /*views*/ scanner_sptr->get_num_detectors_per_ring()/2, 
				    /*tang_pos*/scanner_sptr->get_num_detectors_per_ring()/2, 
				    /*arc_corrected*/ false);
  
  const ProjDataInfoCylindricalNoArcCorr* proj_data_cyl_no_arc_ptr =
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr*>(proj_data_info_sptr.get());
  if (proj_data_cyl_no_arc_ptr==0)
      error("Currently only works with non-arccorrected data");

   shared_ptr<ProjMatrixByDensel> proj_matrix_ptr = 
    new ProjMatrixByDenselUsingRayTracing;
   
   shared_ptr<DiscretisedDensity<3,float> > image_sptr =
        new VoxelsOnCartesianGrid<float>(*proj_data_info_sptr);
   proj_matrix_ptr->set_up(proj_data_info_sptr,
			   image_sptr);
  
   cerr << proj_matrix_ptr->parameter_info();
   const CartesianCoordinate3D<float> voxel_size =
     static_cast<const VoxelsOnCartesianGrid<float>&>(*image_sptr).get_voxel_size();

   ProjMatrixElemsForOneDensel probs;
   bool problem=false;
   while (argc>0)
     {
       const int x = atoi(argv[0]);
       const int y = atoi(argv[1]);
       const int z = atoi(argv[2]);
       Densel densel(z,y,x);
       CartesianCoordinate3D<float> voxel_coords = 
	 CartesianCoordinate3D<float>(z,y,x) * 
	 voxel_size;

       proj_matrix_ptr->get_proj_matrix_elems_for_one_densel(probs, densel);
     
       Bin bin0, bin90;
       for (ProjMatrixElemsForOneDensel::const_iterator element_ptr = probs.begin();
	    element_ptr != probs.end();
	    ++element_ptr)
	 {
	   if (element_ptr->view_num()==0)
	       bin0 = *element_ptr;
	   if(element_ptr->view_num()==proj_data_info_sptr->get_num_views()/2)
	       bin90 = *element_ptr;
	 }
       // note: set value to 1 for comparisons later
       bin0.set_bin_value(1);
       bin90.set_bin_value(1);
       
       int det1_0; 
       int det2_0;
       int ring1_0;
       int ring2_0;
       int det1_90; 
       int det2_90;
       int ring1_90;
       int ring2_90;
#if 0
       // TODO doesn't work yet because this can't handle span, while densel needs it...
       proj_data_cyl_no_arc_ptr->get_det_pair_for_bin(det1_0, ring1_0,
						      det2_0, ring2_0,
						      bin0);
       proj_data_cyl_no_arc_ptr->get_det_pair_for_bin(det1_90, ring1_90,
						      det2_90, ring2_90,
						      bin90);
#else
       proj_data_cyl_no_arc_ptr->get_det_num_pair_for_view_tangential_pos_num(det1_0,det2_0,
									      bin0.view_num(),
									      bin0.tangential_pos_num());
       // warning: dangerous: 
       // only here because of span and works only for segment 0
       ring1_0 = bin0.axial_pos_num()/2;
       ring2_0 = bin0.axial_pos_num()/2;
       Bin bin_check_0;
       proj_data_cyl_no_arc_ptr->get_bin_for_det_pair(bin_check_0,
						      det1_0,ring1_0,
						      det2_0,ring2_0);
       bin_check_0.set_bin_value(1);
       assert(bin0 == bin_check_0);
       proj_data_cyl_no_arc_ptr->get_det_num_pair_for_view_tangential_pos_num(det1_90,det2_90,
									      bin90.view_num(),
									      bin90.tangential_pos_num());
       ring1_90 = bin90.axial_pos_num()/2;
       ring2_90 = bin90.axial_pos_num()/2;

       Bin bin_check_90;
       proj_data_cyl_no_arc_ptr->get_bin_for_det_pair(bin_check_90,
						      det1_90,ring1_90,
						      det2_90,ring2_90);
       bin_check_90.set_bin_value(1);
       assert(bin90 ==bin_check_90);
#endif

       CartesianCoordinate3D<float> coord_1_0;
       CartesianCoordinate3D<float> coord_2_0;
       CartesianCoordinate3D<float> coord_1_90;
       CartesianCoordinate3D<float> coord_2_90;
   
       proj_data_cyl_no_arc_ptr->
	 find_cartesian_coordinates_given_scanner_coordinates (coord_1_0,coord_2_0,
							       ring1_0,ring2_0, 
							       det1_0, det2_0);

       proj_data_cyl_no_arc_ptr->find_cartesian_coordinates_given_scanner_coordinates (coord_1_90,coord_2_90,
										       ring1_90,ring2_90, 
										       det1_90, det2_90);

#if 0
       cout << coord_1_0<<endl;
       cout << coord_2_0 <<endl;

       cout << coord_1_90<<endl;
       cout << coord_2_90<<endl;
#endif
       CartesianCoordinate3D<float> intersection;
       const float distance_between_LORs = 
	 coordinate_between_2_lines(intersection,
				    LORAs2Points<float>(coord_1_0, coord_2_0),
				    LORAs2Points<float>(coord_1_90, coord_2_90));
       if (distance_between_LORs > norm(voxel_size)/sqrt(3.) )
	 {
	   problem = true;
	   warning("Problem:distance between LORs is too large %g", 
		   distance_between_LORs);
	 }
       const float diff = norm(intersection - voxel_coords);
       cout << "distance between intersection and voxel (in mm) " << diff << endl;
       if (diff > norm(voxel_size))
	 cout << endl<< intersection << voxel_coords <<endl;

       argc-=3; argv+=3;
     }
   return problem? EXIT_FAILURE: EXIT_SUCCESS;

}
