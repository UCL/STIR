#include "local/stir/recon_buildblock/ProjMatrixByDenselUsingRayTracing.h"
#include "stir/IO/interfile.h"
#include "stir/ProjDataFromStream.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfo.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/SegmentByView.h"
// for ask_filename...
#include <iostream> 
#include <fstream>
#ifndef STIR_NO_NAMESPACES
using std::iostream;
using std::ofstream;
using std::ifstream;
using std::ios;
using std::cout;
using std::endl;
#endif

#include "stir/stream.h"
START_NAMESPACE_STIR


END_NAMESPACE_STIR
USING_NAMESPACE_STIR

int
main(int argc,char *argv[])
{
  if(argc!=14) 
  {
    cerr<<"Usage: " << argv[0] << " [proj_data-file] list_of_coordinates\n"
        <<"The projdata-file will be used to get the scanner, mashing etc. details" 
	<< endl; 
  }

  
  ProjDataInfo* new_data_info_ptr;
  shared_ptr<ProjData> proj_data_ptr;
  
  if(argc==14)
  {
    proj_data_ptr = ProjData::read_from_file(argv[1]); 
    new_data_info_ptr= proj_data_ptr->get_proj_data_info_ptr()->clone();
  }
  else
  {
    new_data_info_ptr= ProjDataInfo::ask_parameters();
  }

  const ProjDataInfoCylindricalNoArcCorr* proj_data_cyl_no_arc_ptr =
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr*>(new_data_info_ptr);

   shared_ptr<ProjMatrixByDensel> proj_matrix_ptr = 
    new ProjMatrixByDenselUsingRayTracing;
   
   const float zoom = 1;
   int xy_size = static_cast<int>(proj_data_ptr->get_num_tangential_poss()*zoom);
      int z_size = 2*proj_data_ptr->get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings()-1;

   shared_ptr<DiscretisedDensity<3,float> > image_sptr =
        new VoxelsOnCartesianGrid<float>(*(proj_data_ptr->get_proj_data_info_ptr()),
                                         zoom,
                                         CartesianCoordinate3D<float>(0,0,0),
                                         Coordinate3D<int>(z_size,xy_size,xy_size));
   proj_matrix_ptr->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(),
			     image_sptr);
  
   cerr << proj_matrix_ptr->parameter_info();

   ProjMatrixElemsForOneDensel probs;
   int i=1;
   while (i<=4*3)
   {
     float x = atoi(argv[i+1]);
     float y = atoi(argv[i+2]);
     float z = atoi(argv[i+3]);
     Densel densel(z,y,x);
     CartesianCoordinate3D<float> voxel_coords = 
	CartesianCoordinate3D<float>(z,y,x) * 
	static_cast<const VoxelsOnCartesianGrid<float>&>(*image_sptr).get_voxel_size();

     proj_matrix_ptr->get_proj_matrix_elems_for_one_densel(probs, densel);
     
     int segment_num = 0;	    
     int axial_pos_0,tan_pos_0,axial_pos_90,tan_pos_90;
     for (ProjMatrixElemsForOneDensel::const_iterator element_ptr = probs.begin();
     element_ptr != probs.end();
     ++element_ptr)
     {
	      if (element_ptr->view_num()==0)
	      {
		axial_pos_0 = element_ptr->axial_pos_num();
		tan_pos_0 = element_ptr->tangential_pos_num();
	      }
	      if(proj_data_ptr->get_num_views()/2)
	      {
		axial_pos_90 = element_ptr->axial_pos_num();
		tan_pos_90 = element_ptr->tangential_pos_num();
	      }
     }
     
    Bin bin0 (segment_num,0,axial_pos_0,tan_pos_0);
    Bin bin90(segment_num,proj_data_ptr->get_num_views()/2,axial_pos_90,tan_pos_90);
    if (fabs(proj_data_ptr->get_proj_data_info_ptr()->get_phi(bin0))>.001)
	error("Did not select view at 0 degrees\n");

   if (fabs(proj_data_ptr->get_proj_data_info_ptr()->get_phi(bin90) - _PI/2)>.001)
	error("Did not select view at 90 degrees\n");


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
    ring1_0 = bin0.axial_pos_num()/2;
    ring2_0 = bin0.axial_pos_num()/2;
    Bin bin_check_0;
    proj_data_cyl_no_arc_ptr->get_bin_for_det_pair(bin_check_0,
			 det1_0,ring1_0,
			 det2_0,ring2_0);
    bin_check_0.set_bin_value(0);
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
    bin_check_90.set_bin_value(0);
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

#if 1
    cout << coord_1_0<<endl;
    cout << coord_2_0 <<endl;

    cout << coord_1_90<<endl;
    cout << coord_2_90<<endl;
#endif

    if (fabs(coord_1_0.z()-coord_2_0.z())>.001) 
      error("Check intersection of coordinates\n"); 
    if (
        fabs(coord_1_90.z()-coord_2_90.z())>.001) 
	error("Check intersection of coordinates\n"); 
    if (
	fabs(coord_1_0.z()-coord_1_90.z())>.001) 
	error("Check intersection of coordinates\n"); 
    if (
        fabs(coord_2_0.z()-coord_2_90.z())>.001) 
	error("Check intersection of coordinates\n"); 
    if (fabs(coord_1_0.y()-coord_2_0.y())<.001 &&
	fabs(coord_1_90.x()-coord_2_90.x())<.001)
    {
      cout << " view 0 is horizontal and view 90 is vertical\n";
      CartesianCoordinate3D<float> intersection(coord_1_0.z(),coord_1_0.y(),coord_1_90.x());
      cout << endl<< intersection << voxel_coords <<endl;
    }
    else if (fabs(coord_1_90.y()-coord_2_90.y())<.001 &&
	fabs(coord_1_0.x()-coord_2_0.x())<.001)
    {
      cout << " view 0 is vertical and view 90 is horizontal\n";
      CartesianCoordinate3D<float> intersection(coord_1_0.z(),coord_1_90.y(),coord_1_0.x());
      cout << endl<< intersection << voxel_coords <<endl;
    }
    else
	error("Check intersection of coordinates: view 0 or 90 nor horizontal\n"); 

    i+=3;
   }
   return EXIT_SUCCESS;

}