//
//
/*
    Copyright (C) 2000- 2008, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup projection

  \brief stir::ProjMatrixByBinUsingInterpolation's definition 

  \author Kris Thielemans

*/
#ifndef __stir_recon_buildblock_ProjMatrixByBinUsingInterpolation__
#define __stir_recon_buildblock_ProjMatrixByBinUsingInterpolation__

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/shared_ptr.h"

 

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Bin;
/*!
  \ingroup projection
  \brief Computes projection matrix elements for VoxelsOnCartesianGrid images
  by using an interpolation model. 

  This class implements a projection model that interpolates in projection space.
  When used for back-projection, it should give the same results as 
  BackProjectorByByUsingInterpolation, but is probably much slower.

  The current implementation uses some quite generic code to handle symmetries, but
  is very very slow to compute the elements. Once they are cached, performance is 
  as usual of course.

  \warning Preliminary code, not tested to usual STIR standards.
*/

class ProjMatrixByBinUsingInterpolation : 
  public RegisteredParsingObject<
	      ProjMatrixByBinUsingInterpolation,
              ProjMatrixByBin,
              ProjMatrixByBin
	       >
{
public :
    //! Name which will be used when parsing a ProjMatrixByBin object
  static const char * const registered_name; 

  //! Default constructor (calls set_defaults())
  ProjMatrixByBinUsingInterpolation();

  //! Stores all necessary geometric info
  /*! Note that the density_info_ptr is not stored in this object. It's only used to get some info on sizes etc.
  */
  virtual void set_up(		 
		      const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    );

private:
  bool do_symmetry_90degrees_min_phi;
  bool do_symmetry_180degrees_min_phi;
  bool do_symmetry_swap_segment;
  bool do_symmetry_swap_s;
  bool do_symmetry_shift_z;

  // explicitly list necessary members for image details (should use an Info object instead)
  CartesianCoordinate3D<float> voxel_size;
  CartesianCoordinate3D<float> origin;  
  IndexRange<3> densel_range;


  shared_ptr<ProjDataInfo> proj_data_info_ptr;

  // for Jacobian
  const ProjDataInfoCylindrical&
    proj_data_info_cyl() const
    { return static_cast<const ProjDataInfoCylindrical&>(*proj_data_info_ptr); }
/*!
  \brief
  The next class is used
  to take geometric things
  into account. It also includes some normalisation. (internal use only).
  
  \internal 

  Use as follows:
  TODO incorrect (also in original)
  \code
    const JacobianForIntBP jacobian(*(segment.scanner));
    jacobian(segment.get_average_delta(), s+ 0.5);
  \endcode
 */


class JacobianForIntBP 
{
private:
  // store some scanner related data to avoid recomputation
  float R2;
  float ring_spacing2;
  bool arccor;
  // total normalisation of backprojection, 3 factors:
  //  (_Pi/scanner.num_views) for discretisation of integral over phi
  // scanner.ring_spacing for discretisation of integral over delta
  // normalisation of projection space integral: 1/(2 Pi)

  float backprojection_normalisation;

  bool use_exact_Jacobian_now;

public:
  // default constructor needed as now member of projector class (better to make set_up)
  JacobianForIntBP() {}
   explicit JacobianForIntBP(const ProjDataInfoCylindrical* proj_data_info_ptr, bool exact);
   // s in mm here!
   float operator()(const float delta, const float s) const
   {
     float tmp;
     if (use_exact_Jacobian_now)
       tmp = 4*(R2 - s*s);
     else
       tmp = 4*R2;
     if (!arccor)
       tmp *= sqrt(tmp);
     return 
       (arccor ? tmp : pow(tmp,1.5F)) /
       pow(tmp + ring_spacing2*delta*delta, 1.5F)* backprojection_normalisation;
   }
};     


  JacobianForIntBP jacobian;
  bool use_piecewise_linear_interpolation_now;
  bool use_exact_Jacobian_now;

  virtual void 
    calculate_proj_matrix_elems_for_one_bin(
                                            ProjMatrixElemsForOneBin&) const;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

   float
     get_element(const Bin& bin, 
		 const CartesianCoordinate3D<float>& densel_ctr) const;
 private:
   void 
     find_tang_ax_pos_diff(float& tang_pos_diff,
			   float& ax_pos_diff,
			   const Bin& bin,
			   const CartesianCoordinate3D<float>& point) const;
   
};

END_NAMESPACE_STIR

#endif



