//
// $Id$
//
/*
    Copyright (C) 2000- $Date$ , Hammersmith Imanet Ltd
    For internal GE use only
*/
/*!
  \file
  \ingroup motion

  \brief Declaration of class stir::RigidObject3DTransformation

  \author  Sanida Mustafovic and Kris Thielemans
  $Date$
  $Revision$
*/

#ifndef __stir_RigidObject3DTransformation_H__
#define __stir_RigidObject3DTransformation_H__


#include "local/stir/Quaternion.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/Succeeded.h"
#include "stir/Array.h"
#include "stir/Bin.h"
#include "stir/ProjDataInfo.h"

START_NAMESPACE_STIR

/*! \ingroup  motion
  \brief Class to perform rigid object transformations in 3 dimensions

  Supported transformations include rotations and translations. Rotations are
  encoded using quaternions. The convention used is described in<br>
  B.K. Horn, <i>Closed-form solution of absolute orientation using 
  unit quaternions</i>,
  J. Opt. Soc. Am. A Vol.4 No. 6, (1987) p.629.

  \warning This class is tuned to work with the Polaris coordinate system.
  In particular, STIR uses a left-handed coordinate-system, while the Polaris uses
  a right-handed system. To solve this issue, STIR coordinates are x,y swapped before
  performing any actual transformation (and the result is swapped back of course).

  After the swapping, the transformation that is applied is as follows
  \f[ r' = \mathrm{conj}(q)(r-t)q \f]
  where the quaternion is specified as \f$[q0,qx,qy,qz]\f$, while the translation
  is initialised in the usual (in STIR) reverse order, e.g.
  \begin{verbatim}
  CartesianCoordinate3D<float> t(tz,ty,tx);
  \end{verbatim}
  \warning No swapping is performed on the translation. The whole transformation 
  has to be specified in the right-handed system.

  Note that this transformation is the inverse of Horn's.

  This class can transform coordinates and  Bin object belonging to some projection data.

  \warning The Euler angles are probably different from the ones used in the Shape3D hierarchy.
  \todo define Euler angles
*/
class RigidObject3DTransformation
{
public:
  
  RigidObject3DTransformation ();

  //! Constructor taking quaternion and translation info
  RigidObject3DTransformation (const Quaternion<float>& quat, const CartesianCoordinate3D<float>& translation);
  
  //! Compute the inverse transformation
  RigidObject3DTransformation inverse() const;
  //! Get quaternion
  Quaternion<float> get_quaternion() const;
  
  //! Get translation
  CartesianCoordinate3D<float> get_translation() const;
  
  //! Get Euler angles
  Coordinate3D<float> get_euler_angles() const;
  
  
  //Succeeded set_euler_angles();
  
  //! Transform point 
  CartesianCoordinate3D<float> transform_point(const CartesianCoordinate3D<float>& point) const;
  void transform_bin(Bin& bin,const ProjDataInfo& out_proj_data_info,
	             const ProjDataInfo& in_proj_data_info) const;
  //! Get relative transformation (not implemented at present)
  void get_relative_transformation(RigidObject3DTransformation& output, const RigidObject3DTransformation& reference);   
  
  static void quaternion_2_euler(Coordinate3D<float>& Euler_angles, const Quaternion<float>& quat);
  static void quaternion_2_m3(Array<2,float>& mat, const Quaternion<float>& quat);
  static void m3_2_euler(Coordinate3D<float>& Euler_angles, const Array<2,float>& mat); 
  static void euler_2_quaternion(Quaternion<float>& quat,const Coordinate3D<float>& Euler_angles);		/* Euler angles to a quaternion */

private:
  Quaternion<float> quat;
  CartesianCoordinate3D<float> translation;
  friend RigidObject3DTransformation compose ( const RigidObject3DTransformation& apply_last,
					       const RigidObject3DTransformation& apply_first);
};

//! Output to stream
/*! \ingroup motion
    Will be written as \verbatim { quaternion, translation } \endverbatim
*/
std::ostream&
operator<<(std::ostream& out,
	   const RigidObject3DTransformation& rigid_object_transformation);

END_NAMESPACE_STIR

#endif
