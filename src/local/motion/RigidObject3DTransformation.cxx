#include "local/stir/motion/RigidObject3DTransformation.h"
#include "stir/IndexRange2D.h"
#include "stir/stream.h"
#include <math.h>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

#define EPSILON	 0.00001
#define usqrt(x)(float)sqrt((double)(x))
#define uatan2(y, x)(float)atan2((double)(y), (double)(x))
#define ucos(x)	(float)cos((double)(x))
#define usin(x)	(float)sin((double)(x))

START_NAMESPACE_STIR


Array<1,float> 
matrix_multiply( Array<2,float>&matrix, Array<1,float>& vec)
{
  
  Array<1,float> tmp (vec.get_min_index(), vec.get_max_index());
  for ( int i= matrix.get_min_index(); i<=matrix.get_max_index(); i++)
  {
    float elem =0;
    const Array<1,float> row = matrix[i];
    for ( int j = row.get_min_index(); j<=row.get_max_index();j++)
    {
      elem= vec[j]*row[j];
      tmp[i] += elem;
    }
    
  }
  return tmp;
}
  


RigidObject3DTransformation::RigidObject3DTransformation ()
{}


RigidObject3DTransformation::RigidObject3DTransformation (const Quaternion<float>& quat_v, const CartesianCoordinate3D<float>& translation_v)
: quat(quat_v), translation(translation_v)
{
}

RigidObject3DTransformation 
RigidObject3DTransformation::inverse() const
{
  Quaternion<float> quat_copy =quat;
  quat_copy.inverse();
  RigidObject3DTransformation ro3d_trans (quat_copy,translation*(-1));
  return ro3d_trans;
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
    
CartesianCoordinate3D<float> 
RigidObject3DTransformation::get_euler_angles() const
{
  CartesianCoordinate3D<float> euler_angles;
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
  CartesianCoordinate3D<float> swapped_point(point.z(), point.x(), point.y());
 
  //cerr << quat << endl;
  {
    const float quat_norm=square(quat[1]) + square(quat[2]) + square(quat[3]) +square(quat[4]);
    if (fabs(quat_norm-1)>1E-3)
      warning("Non-normalised quaternion: %g", quat_norm);

  }
#if 1

#if 1 // put to 0 for translation only!!!
  
  
  //transformation with quaternions 
  const Quaternion<float> point_q (0,swapped_point.x(),swapped_point.y(),swapped_point.z());
  Quaternion<float> conj_quat = quat;
  conj_quat.conjugate();
  
  Quaternion<float> tmp = quat * point_q * conj_quat;

  tmp[2] += translation.x();
  tmp[3] += translation.y();
  tmp[4] += translation.z();
  const CartesianCoordinate3D<float> transformed_point (tmp[4],tmp[3],tmp[2]);
  
#else
  //translation only
 
  const CartesianCoordinate3D<float> transformed_point (swapped_point.z()+translation.z(),
						  swapped_point.y()+translation.y(),
						  swapped_point.x()+translation.x());
#endif
#else
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
  return CartesianCoordinate3D<float> (transformed_point.z(), transformed_point.x(), transformed_point.y());
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
RigidObject3DTransformation::quaternion_2_euler(CartesianCoordinate3D<float>& Euler_angles, const Quaternion<float>& quat)
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
	s = 2.0/((quat_tmp[1]*quat_tmp[1])+(quat_tmp[2]*quat_tmp[2])+(quat_tmp[3]*quat_tmp[3])+(quat_tmp[4]*quat_tmp[4]));

	xs = quat_tmp[2]*s;  ys = quat_tmp[3]*s;  zs = quat_tmp[4]*s;
	wx = quat_tmp[1]*xs; wy = quat_tmp[1]*ys; wz = quat_tmp[1]*zs;
	xx = quat_tmp[2]*xs; xy = quat_tmp[2]*ys; xz = quat_tmp[2]*zs;
	yy = quat_tmp[3]*ys; yz = quat_tmp[3]*zs; zz = quat_tmp[4]*zs;

	mat[0][0] = 1.0-yy-zz;
	mat[0][1] = xy-wz;
	mat[0][2] = xz+wy;
	mat[1][0] = xy+wz;
	mat[1][1] = 1.0-xx-zz;
	mat[1][2] = yz-wx;
	mat[2][0] = xz-wy;
	mat[2][1] = yz+wx;
	mat[2][2] = 1.0-xx-yy;
}

void 
RigidObject3DTransformation::m3_2_euler(CartesianCoordinate3D<float>& Euler_angles, const Array<2,float>& mat)      /* 3x3 non-homogeneous rotation matrix to Euler angles */
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
	  Euler_angles.x() = uatan2(sx, cx);
	  Euler_angles.y() = uatan2(sy, cy);
	  Euler_angles.z() = uatan2(sz, cz);


}

void 
RigidObject3DTransformation::euler_2_quaternion(Quaternion<float>& quat,const CartesianCoordinate3D<float>& Euler_angles)		/* Euler angles to a Quaternion */
{
	float cx, cy, cz;
	float sx, sy, sz;

	//cx = ucos(Euler_angles[0]/2.0);  sx = usin(Euler_angles[0]/2.0);
	//cy = ucos(Euler_angles[1]/2.0);  sy = usin(Euler_angles[1]/2.0);
	//cz = ucos(Euler_angles[2]/2.0);  sz = usin(Euler_angles[2]/2.0);
	cx = ucos(Euler_angles.x()/2.0);  sx = usin(Euler_angles.x()/2.0);
	cy = ucos(Euler_angles.y()/2.0);  sy = usin(Euler_angles.y()/2.0);
	cz = ucos(Euler_angles.z()/2.0);  sz = usin(Euler_angles.z()/2.0);

	quat[1] = cx*cy*cz+sx*sy*sz;
	quat[2] = sx*cy*cz-cx*sy*sz;
	quat[3] = cx*sy*cz+sx*cy*sz;
	quat[4] = cx*cy*sz-sx*sy*cz;
	if (quat[1] < 0.0)
		quat.neg_quaternion();
}


END_NAMESPACE_STIR
