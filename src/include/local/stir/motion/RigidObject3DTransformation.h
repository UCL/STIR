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

  \warning STIR uses a left-handed coordinate-system. 

  The transformation that is applied is as follows
  \f[ r' = \mathrm{conj}(q)(r-t)q \f]
  where the quaternion is specified as \f${[}q0,qz,qy,qx{]}\f$, while the translation
  is initialised in the usual (in STIR) reverse order, e.g.
  \code
  CartesianCoordinate3D<float> t(tz,ty,tx);
  \endcode
  Note that this transformation is the inverse of Horn's.

  This implements a translation followed by a rotation. The quaternion
  is constructed by as follows:<br>
  A rotation around the axis
  with direction \f${[}n_z,n_y,n_x{]}\f$ and angle \f$\phi\f$ corresponds
  to a quaternion \f${[}cos(\phi),sin(\phi) n_z, sin(\phi) n_y, sin(\phi)n_x{]}\f$.
  \todo Document sign choice for rotation.

  This class can transform coordinates and  Bin object belonging to some projection data.

  \warning The Euler angles are probably different from the ones used in the Shape3D hierarchy.
  \todo define Euler angles (the code is derived from the Polaris manual)
*/
class RigidObject3DTransformation
{
public:
  /*! 
     \brief Find the rigid transformation that gives the closest match between 2 sets of points.

     Minimises the Mean Square Error, i.e. the sum of
     \code
     norm_squared(result.transform_point(orig_point) - transformed_point)
     \endcode

     The implementation uses Horn's algorithm.
 
     Horn's method needs to compute the maximum eigenvector of a matrix,
     which is done here using the Power method
     (see max_eigenvector_using_power_method()).

     \a initial_rotation will be used to initialise the Power method. So, a good choice will
     result in faster convergence, but would also avoid a problem when the default initial
     choice would correspond to another eigenvector of the matrix (giving a very bad match).
  */
  template <class Iter1T, class Iter2T>
    static
    Succeeded
    find_closest_transformation(RigidObject3DTransformation& result,
				Iter1T start_orig_points,
				Iter1T end_orig_points,
				Iter2T start_transformed_points,
				const Quaternion<float>& initial_rotation = Quaternion<float>(1.F,0.F,0.F,0.F));

  /*!
    \brief Compute Root Mean Square Error for 2 sets of points
  */
  template <class Iter1T, class Iter2T>
    static  double
    RMSE(const RigidObject3DTransformation& transformation,
	 Iter1T start_orig_points,
	 Iter1T end_orig_points,
	 Iter2T start_transformed_points);

  RigidObject3DTransformation ();

  //! Constructor taking quaternion and translation info
  /*! \see RigidObject3DTransformation class documentation for conventions */
  RigidObject3DTransformation (const Quaternion<float>& quat, const CartesianCoordinate3D<float>& translation);
  
  //! Compute the inverse transformation
  RigidObject3DTransformation inverse() const;
  //! Get quaternion
  Quaternion<float> get_quaternion() const;
  
  //! Get translation
  CartesianCoordinate3D<float> get_translation() const;

#if 0  
  // implementation probably only works for FIRSTROT
  //! Get Euler angles
  Coordinate3D<float> get_euler_angles() const;
    
  Succeeded set_euler_angles();
#endif

  //! Transform point 
  CartesianCoordinate3D<float> transform_point(const CartesianCoordinate3D<float>& point) const;

  //! Computes the jacobian for the transformation (which is always 1)
  float jacobian(const BasicCoordinate<3,float>& point) const
    { return 1; }

  //! Transform bin from some projection data
  /*!  Finds 'closest' (in some sense) bin to the transformed LOR.

     if NEW_ROT is not #defined at compilation time, 
    it will throw an exception when arc-corrected data is used.*/
  void transform_bin(Bin& bin,const ProjDataInfo& out_proj_data_info,
	             const ProjDataInfo& in_proj_data_info) const;
  //! Get relative transformation (not implemented at present)
  void get_relative_transformation(RigidObject3DTransformation& output, const RigidObject3DTransformation& reference);   
#if 0  
  //! \name conversion to other conventions for rotations
  /*! \warning Currently disabled Code probably only works when FIRSTROT is defined.
   */
  //@{
  static void quaternion_2_euler(Coordinate3D<float>& Euler_angles, const Quaternion<float>& quat);
  /*! Quaternion to 3x3  rotation matrix */
  static void quaternion_2_m3(Array<2,float>& mat, const Quaternion<float>& quat);
  static void m3_2_euler(Coordinate3D<float>& Euler_angles, const Array<2,float>& mat); 
  /*! Euler angles to a quaternion */
  static void euler_2_quaternion(Quaternion<float>& quat,const Coordinate3D<float>& Euler_angles);		
  //@}
#endif

private:
  Quaternion<float> quat;
  CartesianCoordinate3D<float> translation;
  friend RigidObject3DTransformation compose ( const RigidObject3DTransformation& apply_last,
					       const RigidObject3DTransformation& apply_first);
};

//! Output to (text) stream
/*! \ingroup motion
    Will be written as \verbatim { quaternion, translation } \endverbatim
*/
std::ostream&
operator<<(std::ostream& out,
	   const RigidObject3DTransformation& rigid_object_transformation);
//! Input from (text) stream
/*! \ingroup motion
    Should have format \verbatim { quaternion, translation } \endverbatim
*/
std::istream&
operator>>(std::istream& ,
	   RigidObject3DTransformation& rigid_object_transformation);

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
