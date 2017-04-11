//
//
/*!

  \file
  \ingroup projection

  \brief Declares class stir::BackProjectorByBinUsingInterpolation

  \author Kris Thielemans
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
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
#ifndef __BackProjectorByBinUsingInterpolation_h_
#define __BackProjectorByBinUsingInterpolation_h_

#include "stir/recon_buildblock/BackProjectorByBin.h" 
#include "stir/RegisteredParsingObject.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

template <typename elemT> class Viewgram;
template <typename elemT> class RelatedViewgrams;
template <typename elemT> class VoxelsOnCartesianGrid;
template <int num_dimensions, typename elemT> class Array;
class ProjDataInfo;
class ProjDataInfoCylindricalArcCorr;
class DataSymmetriesForBins_PET_CartesianGrid;
/*!
  \brief
  The next class is used in BackProjectorByBinUsingInterpolation 
  to take geometric things
  into account. It also includes some normalisation. (internal use only).
  
  \internal 

  Use as follows:
  \code
    const JacobianForIntBP jacobian(*(segment.scanner));
    jacobian(segment.get_average_delta(), s+ 0.5);
  \endcode
 */


class JacobianForIntBP 
{
private:
  // store some scanner related data to avoid recomputation
  const float R2;
  const float dxy2;
  const float ring_spacing2;
  // total normalisation of backprojection, 3 factors:
  //  (_Pi/scanner.num_views) for discretisation of integral over phi
  // scanner.ring_spacing for discretisation of integral over delta
  // normalisation of projection space integral: 1/(2 Pi)

  const float backprojection_normalisation;

  const bool use_exact_Jacobian_now;

public:
   explicit JacobianForIntBP(const ProjDataInfoCylindricalArcCorr* proj_data_info_ptr, bool exact);

   float operator()(const float delta, const float s) const
   {
     float tmp;
     if (use_exact_Jacobian_now)
       tmp = 4*(R2 - dxy2 * s*s);
     else
       tmp = 4*R2;
     return tmp / pow(tmp + ring_spacing2*delta*delta, 1.5F)* backprojection_normalisation;
   }
};     


/*!
  \ingroup projection
  \brief does backprojection by interpolating between the bins.

  This implementation uses incremental backprojection

  Two versions of interpolation are implemented:
  <ul>
  <li> ordinary linear interpolation
  <li> piecewise linear interpolation in the axial direction
  </ul>
  The former is an implementation of
  "Incremental beamwise backprojection using geometrical symmetries for 3D PET reconstruction 
  in a cylindrical scanner geometry"
  M L Egger, C Joseph, C Morel, Phys. Med. Biol. (1998) 43 3009-3024   
  http://dx.doi.org/10.1088/0031-9155/43/10/023

  For the latter, see the extended abstract for 3D99 
  "On various approximations for the projectors in iterative reconstruction algorithms for 
  3D-PET", K. Thielemans, M.W. Jacobson, D. Belluzzo. Available on the STIR web-site.

  The piecewise linear interpolation is only used when the axial voxel size is half the
  axial_sampling of the projection data (for the segment in question).

  \warning This implementation makes various assumptions (for optimal speed):
  <ul>
  <li> voxel_size.x() = voxel_size.y()
  <li> arc-corrected data 
  <li> voxel_size.z() is either equal to or half the axial_sampling of the projection data
  </ul>
  When the bin size is not equal to the voxel_size.x(), zoom_viewgrams() is first called to
  adjust the bin size, then the usual incremental backprojection is used.

  \bug Currently this implementation has problems on certain processors
  due to floating point rounding errors. Intel *86 and PowerPC give
  correct results, SunSparc has a problem at tangential_pos_num==0 (also HP
  stations give problems).
*/
class BackProjectorByBinUsingInterpolation : 
  public RegisteredParsingObject<BackProjectorByBinUsingInterpolation,
                                 BackProjectorByBin>

{ 
public:
  //! Name which will be used when parsing a BackProjectorByBin object
  static const char * const registered_name; 

  //! The constructor defaults to using piecewise linear interpolation and the exact Jacobian 
  explicit 
    BackProjectorByBinUsingInterpolation(
      const bool use_piecewise_linear_interpolation = true, 
      const bool use_exact_Jacobian = true);

  //! The constructor defaults to using piecewise linear interpolation and the exact Jacobian 
  /*! \deprecated Use set_up() instead */
  BackProjectorByBinUsingInterpolation(
    shared_ptr<ProjDataInfo>const&,
    shared_ptr<DiscretisedDensity<3,float> > const& image_info_ptr,
    const bool use_piecewise_linear_interpolation = true, const bool use_exact_Jacobian = true);

  //! Stores all necessary geometric info
  /*! Note that the density_info_ptr is not stored in this object. It's only used to get some info on sizes etc.
  */
  virtual void set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    );

  /*! \brief Gets the symmetries used by this backprojector

  \warning This BackProjectorByBin implementation requires that the 
  RelatedViewgrams data are constructed with symmetries corresponding
  to the current member. Using another DataSymmetriesForViewSegmentNumbers 
  object will likely crash the program.
  */
  const DataSymmetriesForViewSegmentNumbers * get_symmetries_used() const;
  /*! 
  \brief Use this to switch between the exact Jacobian and 
   an approximate Jacobian (valid for s << R).
   */
  void use_exact_Jacobian(const bool use_exact_Jacobian);


  /*!
  \brief Use this to switch between ordinary linear interpolation and
  piece-wise linear interpolation in the axial direction.
  */
  void use_piecewise_linear_interpolation(const bool use_piecewise_linear_interpolation);

  BackProjectorByBinUsingInterpolation* clone() const;

private:
 
  // KT 20/06/2001 changed type to enable use of more methods
  shared_ptr<DataSymmetriesForBins_PET_CartesianGrid> symmetries_ptr;
  //const DataSymmetriesForViewSegmentNumbers * symmetries_ptr;
  
  bool use_piecewise_linear_interpolation_now;

  bool use_exact_Jacobian_now;

  //! \name variables determining which symmetries will be used
  /*! \warning do NOT use. They are only here for testing purposes */
  //@{
  bool do_symmetry_90degrees_min_phi;
  bool do_symmetry_180degrees_min_phi;
  bool do_symmetry_swap_segment;
  bool do_symmetry_swap_s;
  bool do_symmetry_shift_z;
  //@}

#if 0
  // not used yet
struct ProjDataForIntBP
{ 
  float view__pos_s;
  float view__pos_sp1;
  float view__neg_s; 
  float view__neg_sp1;
  float min90__pos_s; 
  float min90__pos_sp1;
  float min90__neg_s; 
  float min90__neg_sp1;
  float plus90__pos_s; 
  float plus90__pos_sp1;
  float plus90__neg_s; 
  float plus90__neg_sp1;
  float min180__pos_s; 
  float min180__pos_sp1;
  float min180__neg_s; 
  float min180__neg_sp1;
};
#endif

 void actual_back_project(DiscretisedDensity<3,float>&,
                          const RelatedViewgrams<float>&,
		          const int min_axial_pos_num, const int max_axial_pos_num,
		          const int min_tangential_pos_num, const int max_tangential_pos_num);

 void actual_back_project(DiscretisedDensity<3,float>&,
                                  const Bin&);


  virtual void 
   back_project_all_symmetries(  VoxelsOnCartesianGrid<float>& image,
				 const Viewgram<float> & pos_view, 
				 const Viewgram<float> & neg_view, 
				 const Viewgram<float> & pos_plus90, 
				 const Viewgram<float> & neg_plus90, 
				 const Viewgram<float> & pos_min180, 
				 const Viewgram<float> & neg_min180, 
				 const Viewgram<float> & pos_min90, 
				 const Viewgram<float> & neg_min90,				 
				 const int min_axial_pos_num, const int max_axial_pos_num,
				 const int min_tangential_pos_num, const int max_tangential_pos_num);

  /*
    This function projects 4 viewgrams related by symmetry.
    It will be used for view=0 or 45 degrees 
    (or all others if the above version is not implemented in 
    the derived class)
    Here 0<=view < num_views/2 (= 90 degrees)
    */

  virtual void 
  back_project_view_plus_90_and_delta(VoxelsOnCartesianGrid<float>& image,
	                              const Viewgram<float> & pos_view, 
				      const Viewgram<float> & neg_view, 
				      const Viewgram<float> & pos_plus90, 
				      const Viewgram<float> & neg_plus90,					 
				      const int min_axial_pos_num, const int max_axial_pos_num,
				      const int min_tangential_pos_num, const int max_tangential_pos_num);
  /*
  void back_project_2D_view_plus_90(const PETSinogram<float> &sino, PETPlane &image, int view,
                               const int min_bin_num, const intmax_tangential_pos_num);
  void back_project_2D_all_symmetries(const PETSinogram<float> &sino, PETPlane &image, int view,
                                    const int min_bin_num, const intmax_tangential_pos_num);
*/


/* 

  These functions use a 3D version of Cho's algorithm for backprojecting incrementally.
  See M. Egger's thesis for details.
  In addition to the symmetries mentioned above, they also use s,-s symmetry 
  (while being careful when s=0 to avoid self-symmetric cases)
  */


 static void piecewise_linear_interpolation_backproj3D_Cho_view_viewplus90(Array<4, float > const & Projptr,
                                     VoxelsOnCartesianGrid<float>& image,				     
				     const ProjDataInfoCylindricalArcCorr* proj_data_info_ptr,
                                     float delta,
                                     const double cphi, const double sphi, int s, int ax_pos0, 
				     const int num_planes_per_axial_pos,
				     const float axial_pos_to_z_offset);

 static void piecewise_linear_interpolation_backproj3D_Cho_view_viewplus90_180minview_90minview(Array<4, float > const &Projptr,
                                                         VoxelsOnCartesianGrid<float>& image,							 
							 const ProjDataInfoCylindricalArcCorr* proj_data_info_ptr,
                                                         float delta,
                                                          const double cphi, const double sphi, int s, int ax_pos0,
                                                          const int num_planes_per_axial_pos,
							  const float axial_pos_to_z_offset);

  static void linear_interpolation_backproj3D_Cho_view_viewplus90(Array<4, float > const & Projptr,
                                     VoxelsOnCartesianGrid<float>& image,				     
				     const ProjDataInfoCylindricalArcCorr* proj_data_info_ptr,
                                     float delta,
                                     const double cphi, const double sphi, int s, int ax_pos0, 
				     const int num_planes_per_axial_pos,
				     const float axial_pos_to_z_offset);

 static void linear_interpolation_backproj3D_Cho_view_viewplus90_180minview_90minview(Array<4, float > const &Projptr,
                                                         VoxelsOnCartesianGrid<float>& image,							 
							 const ProjDataInfoCylindricalArcCorr* proj_data_info_ptr,
                                                         float delta,
                                                          const double cphi, const double sphi, int s, int ax_pos0,
                                                          const int num_planes_per_axial_pos,
							  const float axial_pos_to_z_offset);

  /*
static void   backproj2D_Cho_view_viewplus90( PETPlane & image,
                                     const ProjDataForIntBP &projs,
                                    const double cphi, const double sphi, int s);

  static void   backproj2D_Cho_view_viewplus90_180minview_90minview( PETPlane & image,
                                     const ProjDataForIntBP &projs,
                                    const double cphi, const double sphi, int s);

*/
  virtual void set_defaults();
  virtual void initialise_keymap();

};

END_NAMESPACE_STIR


#endif // __BackProjectorByBinUsingInterpolation_h_

