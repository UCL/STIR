//
// $Id$
//
/*!

  \file
  \ingroup utilities
  \brief A utility to rorate projection data along the axial direction

  This can be used as a crude way for motion correction, when the motion is only in 
  z-direction.

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/ProjData.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/SegmentByView.h"
#include "stir/ProjDataInterfile.h"
#include "stir/Bin.h"
#include "stir/Succeeded.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/Scanner.h"
#include "stir/round.h"
#include <string>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::cerr;
using std::endl;
using std::min;
using std::max;
#endif

USING_NAMESPACE_STIR

void
find_cartesian_coordinates_given_scanner_coordinates (CartesianCoordinate3D<float>& coord_1,
				 CartesianCoordinate3D<float>& coord_2,
				 const int Ring_A,const int Ring_B, 
				 const int det1, const int det2, 
				 const Scanner& scanner) 
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
// return Succeeded::yes if ok
Succeeded
find_scanner_coordinates_given_cartesian_coordinates(int& det1, int& det2, int& ring1, int& ring2,
							  const CartesianCoordinate3D<float>& c1,
							  const CartesianCoordinate3D<float>& c2,
							  const Scanner& scanner)
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

class TF
{
public:
  TF(const shared_ptr<ProjDataInfo>& out_proj_data_info_ptr,
     const shared_ptr<ProjDataInfo>& in_proj_data_info_ptr,
     const float angleX)
    : out_proj_data_info_ptr(out_proj_data_info_ptr),
      in_proj_data_info_ptr(in_proj_data_info_ptr),
      cosa(cos(angleX)), sina(sin(angleX))
  {
     out_proj_data_info_noarccor_ptr = 
       dynamic_cast<ProjDataInfoCylindricalNoArcCorr*>(out_proj_data_info_ptr.get());
     in_proj_data_info_noarccor_ptr = 
       dynamic_cast<ProjDataInfoCylindricalNoArcCorr*>(in_proj_data_info_ptr.get());
     if (out_proj_data_info_noarccor_ptr == 0 ||
	 in_proj_data_info_noarccor_ptr == 0)
       error("Wrong type of proj_data_info\n");

#if 0
  out_min_segment_num = out_proj_data_info_ptr->get_min_segment_num();
  out_max_segment_num = out_proj_data_info_ptr->get_max_segment_num();
  out_max_ax_pos_num.grow(out_min_segment_num, out_max_segment_num);
  out_min_ax_pos_num.grow(out_min_segment_num, out_max_segment_num);
  for (int segment_num = out_projdata.get_min_segment_num();
       segment_num <= out_projdata.get_max_segment_num();
       ++segment_num)    
    {
      out_max_ax_pos_num[segment_num] = out_segment.get_max_axial_pos_num();
      out_min_ax_pos_num[segment_num] = out_segment.get_min_axial_pos_num();
    }
  out_max_view_num = out_segment.get_max_view_num();
  out_min_view_num = out_segment.get_min_view_num();
  out_max_tang_pos_num = out_segment.get_max_tangential_pos_num();
  out_min_tang_pos_num = out_segment.get_min_tangential_pos_num();
#endif
  }

  void transform_bin(Bin& bin) const
  {
    /*
    const float in_theta = in_proj_data_info_ptr->get_theta(bin);
    const float in_phi = in_proj_data_info_ptr->get_phi(bin);
    const float in_t = in_proj_data_info_ptr->get_t(bin);
    const float in_s = in_proj_data_info_ptr->get_s(bin);
    */
    
    // find detectors
    int det_num_a;
    int det_num_b;
    int ring_a;
    int ring_b;
    in_proj_data_info_noarccor_ptr->get_det_pair_for_bin(
							  det_num_a, ring_a,
							  det_num_b, ring_b, bin);

  // find corresponding cartesian coordinates
  CartesianCoordinate3D<float> coord_1;
  CartesianCoordinate3D<float> coord_2;
  const Scanner * const scanner_ptr = 
    in_proj_data_info_ptr->get_scanner_ptr();

  find_cartesian_coordinates_given_scanner_coordinates(coord_1,coord_2,
    ring_a,ring_b,det_num_a,det_num_b,*scanner_ptr);
  
  // now do the movement
  
   
  const CartesianCoordinate3D<float> 
    coord_1_transformed(coord_1.z()*cosa-coord_1.y()*sina,
			coord_1.y()*cosa+coord_1.z()*sina,
			coord_1.x());
  const CartesianCoordinate3D<float> 
    coord_2_transformed(coord_2.z()*cosa-coord_2.y()*sina,
			coord_2.y()*cosa+coord_2.z()*sina,
			coord_2.x());
  int det_num_a_trans;
  int det_num_b_trans;
  int ring_a_trans;
  int ring_b_trans;
  
  // given two CartesianCoordinates find the intersection     
  if (find_scanner_coordinates_given_cartesian_coordinates(det_num_a_trans,det_num_b_trans,
							   ring_a_trans, ring_b_trans,
							   coord_1_transformed,
							   coord_2_transformed, 
							   *scanner_ptr) ==
      Succeeded::no)
    bin.set_bin_value(-1);


  if (ring_a_trans<0 ||
      ring_a_trans>=scanner_ptr->get_num_rings() ||
      ring_b_trans<0 ||
      ring_b_trans>=scanner_ptr->get_num_rings() ||
      out_proj_data_info_noarccor_ptr->get_bin_for_det_pair(bin,
							     det_num_a_trans, ring_a_trans,
							     det_num_b_trans, ring_b_trans) ==
      Succeeded::no)
    bin.set_bin_value(-1);
      
  }
  
private:
  shared_ptr<ProjDataInfo> out_proj_data_info_ptr;
  shared_ptr<ProjDataInfo> in_proj_data_info_ptr;
  ProjDataInfoCylindricalNoArcCorr *out_proj_data_info_noarccor_ptr;
  ProjDataInfoCylindricalNoArcCorr *in_proj_data_info_noarccor_ptr;
  float cosa;
  float sina;
#if 0
  int out_min_segment_num;
  int out_max_segment_num;
  VectorWithOffset<int> out_max_ax_pos_num;
  VectorWithOffset<int> out_min_ax_pos_num;
  int out_max_view_num;
  int out_min_view_num;
  int out_max_tang_pos_num;
  int out_min_tang_pos_num;
#endif
};

int main(int argc, char **argv)
{
  if (argc < 4 || argc > 6)
    {
      cerr << "Usage:\n"
	   << argv[0] << " output_filename input_projdata_name rotation_angle_around_x_in_degrees [max_in_segment_num_to_process [max_in_segment_num_to_process ]]\n"
	   << "max_in_segment_num_to_process defaults to all segments\n";
      exit(EXIT_FAILURE);
    }
  const string  output_filename = argv[1];
  shared_ptr<ProjData> in_projdata_ptr = ProjData::read_from_file(argv[2]);  
  const float angle_around_x =  atoi(argv[3]) *_PI/180;
  const int max_in_segment_num_to_process = argc <=4 ? in_projdata_ptr->get_max_segment_num() : atoi(argv[4]);
  const int max_out_segment_num_to_process = argc <=5 ? max_in_segment_num_to_process : atoi(argv[5]);

  ProjDataInfo * proj_data_info_ptr =
    in_projdata_ptr->get_proj_data_info_ptr()->clone();
  proj_data_info_ptr->reduce_segment_range(-max_out_segment_num_to_process,max_out_segment_num_to_process);

  ProjDataInterfile out_projdata(proj_data_info_ptr, output_filename, ios::out); 

  TF move_lor(out_projdata.get_proj_data_info_ptr()->clone(),
	      in_projdata_ptr->get_proj_data_info_ptr()->clone(),
	      angle_around_x);
  const int out_min_segment_num = out_projdata.get_min_segment_num();
  const int out_max_segment_num = out_projdata.get_max_segment_num();
  VectorWithOffset<shared_ptr<SegmentByView<float> > > out_seg_ptr(out_min_segment_num, out_max_segment_num);
  for (int segment_num = out_projdata.get_min_segment_num();
       segment_num <= out_projdata.get_max_segment_num();
       ++segment_num)    
    out_seg_ptr[segment_num] = 
      new SegmentByView<float>(out_projdata.get_empty_segment_by_view(segment_num));

  for (int segment_num = -max_in_segment_num_to_process;
       segment_num <= max_in_segment_num_to_process;
       ++segment_num)    
    {       
      const SegmentByView<float> in_segment = 
        in_projdata_ptr->get_segment_by_view( segment_num);
      cerr << "segment_num "<< segment_num << endl;
      const int in_max_ax_pos_num = in_segment.get_max_axial_pos_num();
      const int in_min_ax_pos_num = in_segment.get_min_axial_pos_num();
      const int in_max_view_num = in_segment.get_max_view_num();
      const int in_min_view_num = in_segment.get_min_view_num();
      const int in_max_tang_pos_num = in_segment.get_max_tangential_pos_num();
      const int in_min_tang_pos_num = in_segment.get_min_tangential_pos_num();
      for (int view_num=in_min_view_num; view_num<=in_max_view_num; ++view_num)
	for (int ax_pos_num=in_min_ax_pos_num; ax_pos_num<=in_max_ax_pos_num; ++ax_pos_num)
	  for (int tang_pos_num=in_min_tang_pos_num; tang_pos_num<=in_max_tang_pos_num; ++tang_pos_num)
	    {
	      Bin bin(segment_num, view_num, ax_pos_num, tang_pos_num,
		      in_segment[view_num][ax_pos_num][tang_pos_num]);
	      if (bin.get_bin_value()<=0)
		continue;
	      move_lor.transform_bin(bin);
	      if (bin.get_bin_value()>0)
		(*out_seg_ptr[bin.segment_num()])[bin.view_num()]
						 [bin.axial_pos_num()]
						 [bin.tangential_pos_num()] +=
		  bin.get_bin_value();
	    }
    }

  Succeeded succes = Succeeded::yes;
  for (int segment_num = out_projdata.get_min_segment_num();
       segment_num <= out_projdata.get_max_segment_num();
       ++segment_num)    
    {       
      if (out_projdata.set_segment(*out_seg_ptr[segment_num]) == Succeeded::no)
             succes = Succeeded::no;
    }

    return succes == Succeeded::yes ? EXIT_SUCCESS : EXIT_FAILURE;
}
