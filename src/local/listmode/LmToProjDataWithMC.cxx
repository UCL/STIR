
#include "local/stir/listmode/LmToProjDataWithMC.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "local/stir/listmode/lm.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/Succeeded.h"
#include <time.h>
#include "stir/is_null_ptr.h"




START_NAMESPACE_STIR

void 
LmToProjDataWithMC::set_defaults()
{
  LmToProjData::set_defaults();
 //mt_file_ptr = 0;
  norm_filename = "";
  mt_filename ="";
  attenuation_filename ="";
  ro3d_ptr = 0;
  transmission_duration = 300; // default value 5 min.
}

void 
LmToProjDataWithMC::initialise_keymap()
{
  LmToProjData::initialise_keymap();
  parser.add_start_key("LmToProjDataWithMC Parametres");
  parser.add_key("mt_filename", &mt_filename);
  parser.add_parsing_key("Rigid Object 3D Motion Type", &ro3d_ptr); 
  parser.add_key("attenuation_filename", &attenuation_filename);
  parser.add_key("transmission_duration", &transmission_duration);
  parser.add_stop_key("END");
}

#if 1
LmToProjDataWithMC::
LmToProjDataWithMC(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    parse(par_filename) ;
  else
    ask_parameters();

}
#endif

bool
LmToProjDataWithMC::
post_processing()
{
  if (LmToProjData::post_processing())
    return true;
   
  if (is_null_ptr(ro3d_ptr))
  {
    warning("Invalid Rigid Object 3D Motion object\n");
    return true;
  }

  //shared_ptr<Scanner> scanner_ptr = 
    //new Scanner(*template_proj_data_info_ptr->get_scanner_ptr());
  
   // compute average motion in respect to the transmission scan
  float att_start_time, att_end_time;
  if (attenuation_filename !="")
  {
  find_ref_pos_from_att_file (att_start_time, att_end_time,transmission_duration,
			       attenuation_filename);

  RigidObject3DTransformation av_motion = ro3d_ptr->compute_average_motion(att_start_time,att_end_time);
  
  ro3d_move_to_reference_position =av_motion.inverse();
    
  }
  else
  { 
    att_start_time=0;
    att_end_time=0;
    Quaternion<float> quat(1,0,0,0);
    RigidObject3DTransformation av_motion(quat,CartesianCoordinate3D<float>(0,0,0));
    ro3d_move_to_reference_position =av_motion.inverse();
  }

  
  ro3d_ptr->synchronise(*lm_data_ptr);

  return false;
}

#if 0
LmToProjDataWithMC::
LmToProjDataWithMC(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    parse(par_filename) ;
  else
    ask_parameters();
}

#endif

void 
LmToProjDataWithMC::
find_ref_pos_from_att_file (float& att_start_time, float& att_end_time, 
			    float transmission_duration,
			    const string attenuation_filename)
{
	MatrixFile* AttnFile = matrix_open(attenuation_filename.c_str(), MAT_READ_ONLY, AttenCor );
		/* Acquisition date and time - main head */
	time_t sec_time = AttnFile->mhptr->scan_start_time;

	struct tm* AttnTime = localtime( &sec_time  ) ;
	matrix_close( AttnFile ) ;
	att_start_time = ( AttnTime->tm_hour * 3600.0 ) + ( AttnTime->tm_min * 60.0 ) + AttnTime->tm_sec ;
	att_end_time = att_start_time + transmission_duration;
}
 
void 
LmToProjDataWithMC::get_bin_from_record(Bin& bin, const CListRecord& record,
					const double time,
					const ProjDataInfoCylindrical& proj_data_info) const
{
  //first do the normalisation
  record.event.get_bin(bin, dynamic_cast<const ProjDataInfoCylindrical&>(*proj_data_info_cyl_uncompressed_ptr)); 
  const float bin_efficiency = normalisation_ptr->get_bin_efficiency(bin);
   
  //Do the motion correction
  
  // find detectors
  int det_num_a;
  int det_num_b;
  int ring_a;
  int ring_b;
  record.event.get_detectors(det_num_a,det_num_b,ring_a,ring_b);

  // find corresponding cartesian coordinates
  CartesianCoordinate3D<float> coord_1;
  CartesianCoordinate3D<float> coord_2;
  const Scanner * const scanner_ptr = 
    template_proj_data_info_ptr->get_scanner_ptr();

  find_cartesian_coordinates_given_scanner_coordinates(coord_1,coord_2,
    ring_a,ring_b,det_num_a,det_num_b,*scanner_ptr);
  
  // now do the movement
  
  //Polaris_MT_File::Record mt_record;
  RigidObject3DTransformation ro3dtrans;

  ro3d_ptr->get_motion(ro3dtrans,time);
   
  const CartesianCoordinate3D<float> coord_1_transformed = 
    ro3d_move_to_reference_position.transform_point(ro3dtrans.transform_point(coord_1));

  const CartesianCoordinate3D<float> coord_2_transformed = 
    ro3d_move_to_reference_position.transform_point(ro3dtrans.transform_point(coord_2));
  
  int det_num_a_trans;
  int det_num_b_trans;
  int ring_a_trans;
  int ring_b_trans;
  
  // given two CartesianCoordinates find the intersection     
  find_scanner_coordinates_given_cartesian_coordinates(det_num_a_trans,det_num_b_trans,
    ring_a_trans, ring_b_trans,
    coord_1_transformed,
    coord_2_transformed, 
    *scanner_ptr);
#if 1  
  if ( ring_a_trans > scanner_ptr->get_num_rings() 
	|| ring_a_trans <0 || ring_b_trans <0 || 
	ring_b_trans > scanner_ptr->get_num_rings() ||
        dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(proj_data_info).
        get_bin_for_det_pair(bin,
			 det_num_a_trans, ring_a_trans,
			 det_num_b_trans, ring_b_trans) == Succeeded::no)
  {
    // set to some hopefully sensible value, such that
    // counts are reported ok by LmToProjData::compute
    bin.segment_num() = 
	(ring_b_trans-ring_a_trans)/
	  (proj_data_info.get_max_ring_difference(0) -
	  proj_data_info.get_min_ring_difference(0) + 1);

    bin.set_bin_value(-1);
  }
  else
    bin.set_bin_value(1/bin_efficiency);

#else
  int view_t,elem_t; 
  transform_detector_pair_into_view_bin(view_t,elem_t,det_num_a_trans,det_num_b_trans, 
    *scanner_ptr);
  if ( ring_a_trans < scanner_ptr->get_num_rings() 
	&& ring_a_trans >=0 && ring_b_trans >=0 && 
	ring_b_trans < scanner_ptr->get_num_rings())
  {
    //cerr << "here" <<endl;
    CListRecord new_record;
    new_record.event.set_sinogram_and_ring_coordinates(view_t,elem_t,ring_a_trans,ring_b_trans);
    new_record.event.get_bin(bin, proj_data_info); 
    if (bin.get_bin_value()>0)
	bin.set_bin_value(1/bin_efficiency);

  }
  else
  {
    // set to some hopefully sensible value, such that
    // counts are reported ok by LmToProjData::compute
    bin.segment_num() = 
	(ring_b_trans-ring_a_trans)/
	  (proj_data_info.get_max_ring_difference(0) -
	  proj_data_info.get_min_ring_difference(0) + 1);

    bin.set_bin_value(-1);
  }
#endif  
  
}


void 
LmToProjDataWithMC::find_cartesian_coordinates_given_scanner_coordinates (CartesianCoordinate3D<float>& coord_1,
				 CartesianCoordinate3D<float>& coord_2,
				 const int Ring_A,const int Ring_B, 
				 const int det1, const int det2, 
				 const Scanner& scanner) const
{
  int num_detectors = scanner.get_num_detectors_per_ring();

  float df1 = (2.*_PI/num_detectors)*(det1);
  float df2 = (2.*_PI/num_detectors)*(det2);
  float x1 = scanner.get_ring_radius()*cos(df1);
  float y1 = scanner.get_ring_radius()*sin(df1);
  float x2 = scanner.get_ring_radius()*cos(df2);
  float y2 = scanner.get_ring_radius()*sin(df2);
  float z1 = Ring_A*scanner.get_ring_spacing();
  float z2 = Ring_B*scanner.get_ring_spacing();
   
  coord_1.z() = z1;
  coord_1.y() = y1;
  coord_1.x() = x1;

  coord_2.z() = z2;
  coord_2.y() = y2;
  coord_2.x() = x2;

}

#if 0
void 
LmToProjDataWithMC::find_scanner_coordinates_given_cartesian_coordinates(int& det1, int& det2, int& ring1, int& ring2,
							  const CartesianCoordinate3D<float>& coord_1_in,
							  const CartesianCoordinate3D<float>& coord_2_in, 
							  const Scanner& scanner) const
{
  int num_detectors=scanner.get_num_detectors_per_ring();
  float ring_spacing=scanner.get_ring_spacing();

  det1 = (int)(round(((2.*_PI)+atan2(coord_1_in.y(),coord_1_in.x()))/(2.*_PI/num_detectors)))% num_detectors;
  det2 = (int)(round(((2.*_PI)+atan2(coord_2_in.y(),coord_2_in.x()))/(2.*_PI/num_detectors)))% num_detectors;
  ring1 = round(coord_1_in.z()/ring_spacing);
  ring2 = round(coord_2_in.z()/ring_spacing);
}
#endif

Succeeded
LmToProjDataWithMC::find_scanner_coordinates_given_cartesian_coordinates(int& det1, int& det2, int& ring1, int& ring2,
							  const CartesianCoordinate3D<float>& c1,
							  const CartesianCoordinate3D<float>& c2,
							  const Scanner& scanner) const
{
  const int num_detectors=scanner.get_num_detectors_per_ring();
  const float ring_spacing=scanner.get_ring_spacing();
  const float ring_radius=scanner.get_ring_radius();

  const CartesianCoordinate3D<float> d = c2 - c1;
  /* parametrisation of LOR is 
     c = l*d+c1
     l has to be such that c.x^2 + c.y^2 = R^2
     i.e.
     (l*d.x+c1.x)^2+(l*d.y+c1.y)^2==R^2
     l^2*(d.x^2+d.y^2) + 2*l*(d.x*c1.x + d.y*c1.y) + c1.x^2+c2.y^2-R^2==0
     write as a*l^2+2*b*l+e==0
     l = (-b +- sqrt(b^2-a*e))/a
     argument of sqrt simplifies to
     R^2*(d.x^2+d.y^2)-(d.x*c1.y-d.y*c1.x)^2
  */
  const float dxy2 = (square(d.x())+square(d.y()));
  const float argsqrt=
    (square(ring_radius)*dxy2-square(d.x()*c1.y()-d.y()*c1.x()));
  if (argsqrt<=0)
    return Succeeded::no; // LOR is outside detector radius
  const float root = sqrt(argsqrt);

  const float l1 = (- (d.x()*c1.x() + d.y()*c1.y())+root)/dxy2;
  const float l2 = (- (d.x()*c1.x() + d.y()*c1.y())-root)/dxy2;
  const CartesianCoordinate3D<float> coord_det1 = d*l1 + c1;
  const CartesianCoordinate3D<float> coord_det2 = d*l2 + c1;
  assert(fabs(square(coord_det1.x())+square(coord_det1.y())-square(ring_radius))<square(ring_radius)*10.E-5);
  assert(fabs(square(coord_det2.x())+square(coord_det2.y())-square(ring_radius))<square(ring_radius)*10.E-5);

  det1 = stir::round(((2.*_PI)+atan2(coord_det1.y(),coord_det1.x()))/(2.*_PI/num_detectors))% num_detectors;
  det2 = stir::round(((2.*_PI)+atan2(coord_det2.y(),coord_det2.x()))/(2.*_PI/num_detectors))% num_detectors;
  ring1 = round(coord_det1.z()/ring_spacing);
  ring2 = round(coord_det2.z()/ring_spacing);

#ifndef NDEBUG
  {

    CartesianCoordinate3D<float> check1, check2;
    find_cartesian_coordinates_given_scanner_coordinates (check1, check2,
							  ring1,ring2, 
							  det1, det2, 
							  scanner);
    assert(norm(coord_det1-check1)<ring_spacing);
    assert(norm(coord_det2-check2)<ring_spacing);
  }
#endif
  return Succeeded::yes;
}

void 
LmToProjDataWithMC::transform_detector_pair_into_view_bin (int& view,int& bin, 
					    const int det1,const int det2, 
					    const Scanner& scanner) const
{ 
  const int num_detectors = scanner.get_num_detectors_per_ring();
  const int h=num_detectors/2;
  const int x = (det1>det2)?det1:det2;
  const int y = (det1<det2)?det1:det2;
  const int a=((x+y+h+1)%num_detectors)/2;
  const int b=a+h;
  int te=abs(x-y-h);
  if ((y<a)||(b<x)) te = -te;
  bin=te;
  view=a;
}





END_NAMESPACE_STIR
