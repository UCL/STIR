//
// $Id: 
//
/*!
  \file
  \ingroup local_buildblock

  \brief Declaration of class RigidObject3DTransformation

  \author  Sanida Mustafovic and Kris Thielemans
  $Date$
  $Revision$
*/

/*
    Copyright (C) 2000- $Date$ , Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#ifndef __stir_RigidObject3DTransformation_H__
#define __stir_RigidObject3DTransformation_H__


#include "local/stir/Quaternion.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/Succeeded.h"
#include "stir/Bin.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"

START_NAMESPACE_STIR

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
  
  //! Set Euler angles
  //Succeeded set_euler_angles();
  
  //! Transform point 
  CartesianCoordinate3D<float> transform_point(const CartesianCoordinate3D<float>& point) const;
#ifndef NEW_ROT
  void transform_bin(Bin& bin,const ProjDataInfoCylindricalNoArcCorr& out_proj_data_info,
	             const ProjDataInfoCylindricalNoArcCorr& in_proj_data_info) const;
#else
  void transform_bin(Bin& bin,const ProjDataInfo& out_proj_data_info,
	             const ProjDataInfo& in_proj_data_info) const;
#endif  
  //! Get relative transformation
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

END_NAMESPACE_STIR

#endif
