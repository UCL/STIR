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
#include <algorithm>

START_NAMESPACE_STIR
class Succeeded;

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
  Use the functions right_handed_to_stir() and stir_to_right_handed() if you need to do
  this explicitly..

  After the swapping, the transformation that is applied is as follows
  \f[ r' = \mathrm{conj}(q)(r-t)q \f]
  where the quaternion is specified as \f${[}q0,qx,qy,qz{]}\f$, while the translation
  is initialised in the usual (in STIR) reverse order, e.g.
  \code
  CartesianCoordinate3D<float> t(tz,ty,tx);
  \endcode
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
  //! a function to convert a coordinate in a right-handed system to a left-handed system as used by STIR
  static inline 
    CartesianCoordinate3D<float>
    right_handed_to_stir(const CartesianCoordinate3D<float>& p)
    { return CartesianCoordinate3D<float>(p.z(), p.x(), p.y()); }
  //! a function to convert a coordinate in a left-handed system as used by STIR to a right-handed system
  static inline
    CartesianCoordinate3D<float>
    stir_to_right_handed(const CartesianCoordinate3D<float>& p)
    { return CartesianCoordinate3D<float>(p.z(), p.x(), p.y()); }
  template <class Iter1T, class Iter2T>
    static
    Succeeded
    find_closest_transformation(RigidObject3DTransformation& result,
				Iter1T start_orig_points,
				Iter1T end_orig_points,
				Iter2T start_transformed_points,
				const Quaternion<float>& initial_rotation);

  template <class Iter1T, class Iter2T>
    static  double
    RMS(const RigidObject3DTransformation& transformation,
	Iter1T start_orig_points,
	Iter1T end_orig_points,
	Iter2T start_transformed_points);

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
  //! Transform bin in from some projection data
  /*!  Finds 'closest' (in some sense) bin to the transformed LOR.

     if NEW_ROT is not #defined at compilation time, 
    it will throw an exception when arc-corrected data is used.*/
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

//! Composition of 2 transformations
/*! \ingroup motion
   This provides a way to perform 2 transformations after eachother.
   The following code will work
   \code
    RigidObject3DTransformation tf_1,tf_2; // initialise somehow
    const RigidObject3DTransformation tf_2_1 = compose(tf_2,tf_1));
      const CartesianCoordinate3D<float> point(1.F,-5.F,2.F);
      assert(norm(tf_2.transform_point(tf_1.transform_point(point)) -
                  tf_2_ 1.transform_point(point))
             < .01);
   \endcode
*/

RigidObject3DTransformation 
compose (const RigidObject3DTransformation& apply_last,
	 const RigidObject3DTransformation& apply_first);

END_NAMESPACE_STIR

#endif
