//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    For internal GE use only.
*/
/*!
  \file 
  \ingroup motion
  \brief implementation of class RigidObject3DTransformation
  \author Sanida Mustafovic
  \author Kris Thielemans

  $Date$
  $Revision$

*/
#include "local/stir/motion/RigidObject3DTransformation.h"
#include "stir/IndexRange2D.h"
#include "stir/LORCoordinates.h"
#include "stir/stream.h"
#include "stir/IndexRange2D.h"
#include "stir/Succeeded.h"
#include "stir/stream.h"
#include "stir/more_algorithms.h"
#include "stir/numerics/max_eigenvector.h"
#include <vector>
#include <cmath>
#ifndef NEW_ROT
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#endif

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

#define DO_XY_SWAP

#define EPSILON	 0.00001
#if 0
#define usqrt(x)(float)sqrt((double)(x))
#define uatan2(y, x)(float)atan2((double)(y), (double)(x))
#define ucos(x)	(float)cos((double)(x))
#define usin(x)	(float)sin((double)(x))
#else
#define usqrt(x) std::sqrt(x)
#define uatan2(y,x) std::atan2(y,x)
#define ucos(x) std::cos(x)
#define usin(x) std::sin(x)
#endif

//#define FIRSTROT  /* rotate first before translation. Note: FIRSTROT code effectively computes inverse transformation of !FIRSTROT */
#define WITHQUAT  /* implements transformation using matrices (same result, but slower) */

#if !defined(WITHQUAT) && !defined(FIRSTROT)
#error did not implement FIRSTROT yet with matrices
#endif

#ifndef WITHQUAT
#include "stir/numerics/MatrixFunction.h"
#endif

START_NAMESPACE_STIR

RigidObject3DTransformation::
RigidObject3DTransformation ()
: quat(Quaternion<float>(1,0,0,0)), 
  translation(CartesianCoordinate3D<float>(0,0,0))
{}


RigidObject3DTransformation::
RigidObject3DTransformation (const Quaternion<float>& quat_v, const CartesianCoordinate3D<float>& translation_v)
: quat(quat_v), translation(translation_v)
{
  // test if quaternion normalised
  assert(fabs(square(quat[1]) + square(quat[2]) + square(quat[3]) +square(quat[4]) - 1)<1E-3);
  // alternatively wwe could just normalise it here
}

RigidObject3DTransformation 
RigidObject3DTransformation::inverse() const
{
#ifdef FIRSTROT
  /* Formula for inverse is a bit complicated because of
    fixed order of first rotation and then translation

     tr_point= transform(point) =
		 q*point*conj(q) + trans
	invtransform(tr_point) = 
		invq*(q*point*conj(q)+trans)*conj(invq) -
                invq*trans*conj(invq)
            = point
   */
  const Quaternion<float> invq = stir::inverse(quat);
  const Quaternion<float>
    qtrans(0,translation.x(),translation.y(),translation.z());
  const Quaternion<float> qinvtrans =
    invq * qtrans * conjugate(invq);
  const CartesianCoordinate3D<float>
    invtrans(qinvtrans[4],qinvtrans[3],qinvtrans[2]);
  return RigidObject3DTransformation(invq, invtrans*(-1));
#else
    /* Formula for inverse is a bit complicated because of
    fixed order of first translation and then rotation

     tr_point= transform(point) =
		 conj(q)*(point-trans)*q
	invtransform(tr_point) = 
	          conj(invq)*(tr_point - invtrans)*invq
		= conj(invq)*(conj(q)*(point-trans)*q - invtrans)*invq
            = point
	so -conj(q)*(trans)*q + -invtrans==0
   */
  const Quaternion<float> invq = stir::inverse(quat);
  const Quaternion<float>
    qtrans(0,translation.x(),translation.y(),translation.z());
  const Quaternion<float> qinvtrans =
    conjugate(quat) * qtrans * quat;
  const CartesianCoordinate3D<float>
    invtrans(qinvtrans[4],qinvtrans[3],qinvtrans[2]);
  return RigidObject3DTransformation(invq, invtrans*(-1));
#endif
}

Quaternion<float>
RigidObject3DTransformation::get_quaternion()  const
{
  return quat;
}

CartesianCoordinate3D<float> 
RigidObject3DTransformation::get_translation() const
{
  return translation;
}
    
Coordinate3D<float> 
RigidObject3DTransformation::get_euler_angles() const
{
  Coordinate3D<float> euler_angles;
  quaternion_2_euler(euler_angles, (*this).get_quaternion());
  
  return euler_angles;

}
#if 0
Succeeded 
RigidObject3DTransformation::set_euler_angles()
{


}

#endif

CartesianCoordinate3D<float> 
RigidObject3DTransformation::transform_point(const CartesianCoordinate3D<float>& point) const
{
  CartesianCoordinate3D<float> swapped_point =
#ifndef DO_XY_SWAP
    point;
#else
      stir_to_right_handed(point);
#endif

  const Quaternion<float> quat_norm_tmp = quat;
#if 0 // no longer normalise here, but in Polaris_MT_File   
  //cerr << quat << endl;
  {
    const float quat_norm=square(quat[1]) + square(quat[2]) + square(quat[3]) +square(quat[4]);
     
    if (fabs(quat_norm-1)>1E-3)
    //Quaternion<float> quat_norm = quat;
    quat_norm_tmp.normalise();
    //if (fabs(quat_norm-1)>1E-3)
      //warning("Non-normalised quaternion: %g", quat_norm);

  }
#endif
#ifdef WITHQUAT

  //transformation with quaternions 
  const Quaternion<float> point_q (0,swapped_point.x(),swapped_point.y(),swapped_point.z());
  
#ifdef FIRSTROT
  Quaternion<float> tmp = quat_norm_tmp * point_q * conjugate(quat_norm_tmp);

  tmp[2] += translation.x();
  tmp[3] += translation.y();
  tmp[4] += translation.z();
  const CartesianCoordinate3D<float> transformed_point (tmp[4],tmp[3],tmp[2]);
#else  
  // first include transation and then do the transfromation with quaternion where the other is now 
  // swapped
  Quaternion<float> tmp1=point_q;
  tmp1[2] -= translation.x();
  tmp1[3] -= translation.y();
  tmp1[4] -= translation.z();
  
  const Quaternion<float> tmp =  conjugate(quat_norm_tmp) * tmp1 *quat_norm_tmp ;

  const CartesianCoordinate3D<float> transformed_point (tmp[4],tmp[3],tmp[2]);
#endif

#else // for rotational matrix 
  // transformation with rotational matrix
  Array<2,float> matrix = Array<2,float>(IndexRange2D(0,2,0,2));

  quaternion_2_m3(matrix,quat); 

    
  Array<1,float> tmp(matrix.get_min_index(), matrix.get_max_index());

  tmp[matrix.get_min_index()]=swapped_point.x();
  tmp[matrix.get_min_index()+1]=swapped_point.y();
  tmp[matrix.get_max_index()]=swapped_point.z();

  Array<1,float> out(matrix.get_min_index(), matrix.get_max_index());
  // rotation
  out = matrix_multiply(matrix,tmp);
  //translation
  out[matrix.get_min_index()] += translation.x();
  out[matrix.get_min_index()+1] += translation.y();
  out[matrix.get_max_index()] += translation.z();

  const CartesianCoordinate3D<float> transformed_point(out[out.get_max_index()],out[out.get_min_index()+1],out[out.get_min_index()]);

#endif

  return 
#ifndef DO_XY_SWAP
    transformed_point;
#else
    right_handed_to_stir(transformed_point);
#endif
}

void 
RigidObject3DTransformation::
transform_bin(Bin& bin,const ProjDataInfo& out_proj_data_info,
	             const ProjDataInfo& in_proj_data_info) const
{

  const float value = bin.get_bin_value();
#ifndef NEW_ROT
  CartesianCoordinate3D<float> coord_1;
  CartesianCoordinate3D<float> coord_2;
  dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(in_proj_data_info).
    find_cartesian_coordinates_of_detection(coord_1,coord_2,bin);
  
  // now do the movement
  
  const CartesianCoordinate3D<float> 
    coord_1_transformed = transform_point(coord_1);
  
  const CartesianCoordinate3D<float> 
    coord_2_transformed = transform_point(coord_2);
  
  dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(out_proj_data_info).
    find_bin_given_cartesian_coordinates_of_detection(bin,
                                                      coord_1_transformed,
					              coord_2_transformed);
#else
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
  in_proj_data_info.get_LOR(lor, bin);
  LORAs2Points<float> lor_as_points;
  lor.get_intersections_with_cylinder(lor_as_points, lor.radius());
  // TODO origin
  // currently, the origin used for  proj_data_info is in the centre of the scanner,
  // while for standard images it is in the centre of the first ring.
  // This is pretty horrible though, as the transform_point function has no clue 
  // where the origin is
  // Note that the present shift will make this version compatible with the 
  // version above, as find_bin_given_cartesian_coordinates_of_detection
  // also uses an origin in the centre of the first ring
  const float z_shift = 
    (in_proj_data_info.get_scanner_ptr()->get_num_rings()-1)/2.F *
    in_proj_data_info.get_scanner_ptr()->get_ring_spacing();
  lor_as_points.p1().z() += z_shift;
  lor_as_points.p2().z() += z_shift;
  LORAs2Points<float> 
    transformed_lor_as_points(transform_point(lor_as_points.p1()),
			      transform_point(lor_as_points.p2()));
  assert(*in_proj_data_info.get_scanner_ptr() ==
	 *out_proj_data_info.get_scanner_ptr());
  transformed_lor_as_points.p1().z() -= z_shift;
  transformed_lor_as_points.p2().z() -= z_shift;
  bin = out_proj_data_info.get_bin(transformed_lor_as_points);
#endif
  if (bin.get_bin_value()>0)
    bin.set_bin_value(value);
}
  

void
RigidObject3DTransformation::get_relative_transformation(RigidObject3DTransformation& output, const RigidObject3DTransformation& reference)
{
#if 1
  error("RigidObject3DTransformation::get_relative_transformatio not implemented\n");
#else
  CartesianCoordinate3D<float> trans;
  Quaternion<float> quat; 
  quat = (*this).quat-reference.quat;
  trans = (*this).translation-reference.translation;
 
  output(quat,trans); 
#endif

}


void 
RigidObject3DTransformation::
quaternion_2_euler(Coordinate3D<float>& Euler_angles, const Quaternion<float>& quat)
{
    Array<2,float> matrix = Array<2,float>(IndexRange2D(0,2,0,2));
    quaternion_2_m3(matrix,quat);
    m3_2_euler(Euler_angles, matrix);
}


void 
RigidObject3DTransformation::quaternion_2_m3(Array<2,float>& mat, const Quaternion<float>& quat)	/* Quaternion to 3x3 non-homogeneous rotation matrix */
{
    //assert( mat.get_min_index() == 0 && mat.get_max_index()==2)
	//scalar s, xs, ys, zs;
	//scalar wx, wy, wz, xy, xz, yz, xx, yy, zz;

	float s, xs, ys, zs;
	float wx, wy, wz, xy, xz, yz, xx, yy, zz;
	Quaternion<float> quat_tmp = quat;

	if ( (square(quat[1]) + square(quat[2]) + square(quat[3]) +square(quat[4]))!=1)
	quat_tmp.normalise();
	
	//cerr << quat_tmp[1]<< "  "<< quat_tmp[2] << "   " << quat_tmp[3] << "   "<< quat_tmp[4] << "  ";
	//cerr << endl;


	// TODO check the sizes here -- before 0->3; now 1->4 
	s = 2.0F/((quat_tmp[1]*quat_tmp[1])+(quat_tmp[2]*quat_tmp[2])+(quat_tmp[3]*quat_tmp[3])+(quat_tmp[4]*quat_tmp[4]));

	xs = quat_tmp[2]*s;  ys = quat_tmp[3]*s;  zs = quat_tmp[4]*s;
	wx = quat_tmp[1]*xs; wy = quat_tmp[1]*ys; wz = quat_tmp[1]*zs;
	xx = quat_tmp[2]*xs; xy = quat_tmp[2]*ys; xz = quat_tmp[2]*zs;
	yy = quat_tmp[3]*ys; yz = quat_tmp[3]*zs; zz = quat_tmp[4]*zs;

	mat[0][0] = 1.0F-yy-zz;
	mat[0][1] = xy-wz;
	mat[0][2] = xz+wy;
	mat[1][0] = xy+wz;
	mat[1][1] = 1.0F-xx-zz;
	mat[1][2] = yz-wx;
	mat[2][0] = xz-wy;
	mat[2][1] = yz+wx;
	mat[2][2] = 1.0F-xx-yy;
}

void 
RigidObject3DTransformation::m3_2_euler(Coordinate3D<float>& Euler_angles, const Array<2,float>& mat)      /* 3x3 non-homogeneous rotation matrix to Euler angles */
{ 
    //assert ( mat.get_min_index() ==0 && mat.get_max_inedx()==2)

	float cx, cy, cz;
	float sx, sy, sz;

	sy = -mat[2][0];
	cy = 1-(sy*sy);
	if (cy > EPSILON) {
		cy = usqrt(cy);
		cx = mat[2][2]/cy;
		sx = mat[2][1]/cy;
		cz = mat[0][0]/cy;
		sz = mat[1][0]/cy;
	}
	else {
		cy = 0.0;
		cx = mat[1][1];
		sx = -mat[1][2];
		cz = 1.0;
		sz = 0.0;
	}

	//Euler_angles[0] = uatan2(sx, cx);
	//Euler_angles[1] = uatan2(sy, cy);
	//Euler_angles[2] = uatan2(sz, cz);
	  Euler_angles[3] = uatan2(sx, cx);
	  Euler_angles[2] = uatan2(sy, cy);
	  Euler_angles[1] = uatan2(sz, cz);


}

void 
RigidObject3DTransformation::euler_2_quaternion(Quaternion<float>& quat,const Coordinate3D<float>& Euler_angles)		/* Euler angles to a Quaternion */
{
	float cx, cy, cz;
	float sx, sy, sz;

	//cx = ucos(Euler_angles[0]/2.0);  sx = usin(Euler_angles[0]/2.0);
	//cy = ucos(Euler_angles[1]/2.0);  sy = usin(Euler_angles[1]/2.0);
	//cz = ucos(Euler_angles[2]/2.0);  sz = usin(Euler_angles[2]/2.0);
	cx = ucos(Euler_angles[3]/2.0);  sx = usin(Euler_angles[3]/2.0);
	cy = ucos(Euler_angles[2]/2.0);  sy = usin(Euler_angles[2]/2.0);
	cz = ucos(Euler_angles[1]/2.0);  sz = usin(Euler_angles[1]/2.0);

	quat[1] = cx*cy*cz+sx*sy*sz;
	quat[2] = sx*cy*cz-cx*sy*sz;
	quat[3] = cx*sy*cz+sx*cy*sz;
	quat[4] = cx*cy*sz-sx*sy*cz;
	if (quat[1] < 0.0)
		quat.neg_quaternion();
}

RigidObject3DTransformation 
compose (const RigidObject3DTransformation& apply_last,
	 const RigidObject3DTransformation& apply_first)
{
 const Quaternion<float> quat_tmp (0,apply_last.get_translation().x(),apply_last.get_translation().y(),apply_last.get_translation().z());

 const Quaternion<float> rotated_last_translation_quat =
   apply_first.get_quaternion()*quat_tmp*conjugate(apply_first.get_quaternion());
 const CartesianCoordinate3D<float> trans(rotated_last_translation_quat[4],
					  rotated_last_translation_quat[3],
					  rotated_last_translation_quat[2]);

 return RigidObject3DTransformation(apply_first.get_quaternion()*apply_last.get_quaternion(),
				     apply_first.get_translation()+trans);
}

std::ostream&
operator<<(std::ostream& out,
	   const RigidObject3DTransformation& rigid_object_transformation)
{
  out << '{' 
      << rigid_object_transformation.get_quaternion()
      << ','
      << rigid_object_transformation.get_translation()
      << "}";
  return out;
}


/****************** find_closest_transformation **************/
namespace detail
{

  template <class Iter1T, class Iter2T>
  static
  Array<2,float> 
  construct_Horn_matrix(Iter1T start_orig_points,
			Iter1T end_orig_points,
			Iter2T start_transformed_points,
			const CartesianCoordinate3D<float>& orig_average,
			const CartesianCoordinate3D<float>& transf_average)
  {
    Array<2,float> m(IndexRange2D(4,4));
    Iter1T orig_iter=start_orig_points;
    Iter2T transf_iter=start_transformed_points;
    while (orig_iter!=end_orig_points)
      {
	const CartesianCoordinate3D<float> orig=*orig_iter - orig_average; 
	const CartesianCoordinate3D<float> transf=*transf_iter - transf_average; 
	m[0][0] +=
	  -(-orig.x()*transf.x()-orig.y()*transf.y()-orig.z()*transf.z());
	m[0][1] +=
	  -(-transf.y()*orig.z()+orig.y()*transf.z());
	m[0][2] +=
          -(transf.x()*orig.z()-orig.x()*transf.z());
	m[0][3] +=
	  -(-transf.x()*orig.y()+orig.x()*transf.y());

	m[1][1] +=
	  -(-orig.x()*transf.x()+orig.y()*transf.y()+orig.z()*transf.z());
	m[1][2] +=
	  -(-transf.x()*orig.y()-orig.x()*transf.y());
	m[1][3] +=
	  -(-transf.x()*orig.z()-orig.x()*transf.z());

	m[2][2] +=
	  -(orig.x()*transf.x()-orig.y()*transf.y()+orig.z()*transf.z());
	m[2][3] +=
	  -(-transf.y()*orig.z()-orig.y()*transf.z());

	m[3][3] +=
	  -(orig.x()*transf.x()+orig.y()*transf.y()-orig.z()*transf.z());

	++orig_iter;
	++transf_iter;
      }

    // now make symmetric
    m[1][0] = m[0][1];
    m[2][0] = m[0][2];
    m[3][0] = m[0][3];
    m[2][1] = m[1][2];
    m[3][1] = m[1][3];
    m[3][2] = m[2][3];

    //std::cerr << "\nHorn: " << m;
    return m;
  }

  template <class Iter1T, class Iter2T>
  static
  Array<2,float> 
  construct_Horn_matrix(Iter1T start_orig_points,
			Iter1T end_orig_points,
			Iter2T start_transformed_points)
  {
    const CartesianCoordinate3D<float> orig_average =
      average(start_orig_points, end_orig_points);
    const CartesianCoordinate3D<float> transf_average =
      average(start_transformed_points, 
	      start_transformed_points + (end_orig_points - start_orig_points));

    return construct_Horn_matrix(start_orig_points, end_orig_points,
				 orig_average, transf_average);
  }
} // end of namespace detail

template <class Iter1T, class Iter2T>
double
RigidObject3DTransformation::
RMS(const RigidObject3DTransformation& transformation,
    Iter1T start_orig_points,
    Iter1T end_orig_points,
    Iter2T start_transformed_points)
{
  double result = 0;
  Iter1T orig_iter=start_orig_points;
  Iter2T transf_iter=start_transformed_points;
  while (orig_iter!=end_orig_points)
    {
      result += norm_squared(transformation.transform_point(*orig_iter) -
			     *transf_iter);
      ++orig_iter;
      ++transf_iter;
    }
  
  return std::sqrt(result / (end_orig_points - start_orig_points));
}

template <class Iter1T, class Iter2T>
Succeeded
RigidObject3DTransformation::
find_closest_transformation(RigidObject3DTransformation& result,
			    Iter1T start_orig_points,
			    Iter1T end_orig_points,
			    Iter2T start_transformed_points,
			    const Quaternion<float>& initial_rotation)
{
#ifdef DO_XY_SWAP
  error("Currently find_closest_transformation does not work with these conventions");
#endif
    const CartesianCoordinate3D<float> orig_average =
      average(start_orig_points, end_orig_points);
    const CartesianCoordinate3D<float> transf_average =
      average(start_transformed_points, 
	      start_transformed_points + (end_orig_points - start_orig_points));

  const Array<2,float> horn_matrix = 
    detail::construct_Horn_matrix(start_orig_points,
				  end_orig_points,
				  start_transformed_points,
				  orig_average, transf_average);
  
  float max_eigenvalue;
  Array<1,float> max_eigenvector;
  Array<1,float> start(4);
  std::copy(initial_rotation.begin(), initial_rotation.end(), start.begin());
  if (max_eigenvector_using_power_method(max_eigenvalue,
					 max_eigenvector,
					 horn_matrix,
					 start,
					 /* tolerance*/ .0005,
					 /*max_num_iterations*/ 10000UL)
      == Succeeded::no)
    {
      warning("find_closest_transformation failed because power method did not converge");
      return Succeeded::no;
    }
  Quaternion<float> q;
  std::copy(max_eigenvector.begin(), max_eigenvector.end(), q.begin());

  const RigidObject3DTransformation 
    centred_transf(conjugate(q),CartesianCoordinate3D<float>(0,0,0));

  const CartesianCoordinate3D<float> translation =
    orig_average - centred_transf.transform_point(transf_average);

  result = RigidObject3DTransformation(q, translation);

  return Succeeded::yes;
}


template 
Succeeded
RigidObject3DTransformation::
find_closest_transformation<>(RigidObject3DTransformation& result,
			      std::vector<CartesianCoordinate3D<float> >::const_iterator start_orig_points,
			      std::vector<CartesianCoordinate3D<float> >::const_iterator end_orig_points,
			      std::vector<CartesianCoordinate3D<float> >::const_iterator start_transformed_points,
			      const Quaternion<float>& initial_rotation);
template 
Succeeded
RigidObject3DTransformation::
find_closest_transformation<>(RigidObject3DTransformation& result,
			      std::vector<CartesianCoordinate3D<float> >::iterator start_orig_points,
			      std::vector<CartesianCoordinate3D<float> >::iterator end_orig_points,
			      std::vector<CartesianCoordinate3D<float> >::iterator start_transformed_points,
			      const Quaternion<float>& initial_rotation);


template
double
RigidObject3DTransformation::
RMS<>(const RigidObject3DTransformation&,
      std::vector<CartesianCoordinate3D<float> >::const_iterator start_orig_points,
      std::vector<CartesianCoordinate3D<float> >::const_iterator end_orig_points,
      std::vector<CartesianCoordinate3D<float> >::const_iterator start_transformed_points);

template
double
RigidObject3DTransformation::
RMS<>(const RigidObject3DTransformation&,
      std::vector<CartesianCoordinate3D<float> >::iterator start_orig_points,
      std::vector<CartesianCoordinate3D<float> >::iterator end_orig_points,
      std::vector<CartesianCoordinate3D<float> >::iterator start_transformed_points);

END_NAMESPACE_STIR
