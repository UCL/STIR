//
// $Id$: $Date$
//
#ifndef __BackProjectorByBinUsingInterpolation_h_
#define __BackProjectorByBinUsingInterpolation_h_
/*!

  \file

  \brief Declares class BackProjectorByBinUsingInterpolation

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/

#include "recon_buildblock/BackProjectorByBin.h" 
#include "Array.h"
#include "DataSymmetriesForViewSegmentNumbers_PET_CartesianGrid.h"
#include "ProjDataInfoCylindricalArcCorr.h"



START_NAMESPACE_TOMO

template <typename elemT> class Segment;

/*!
  \brief
  The next class is used in backprojection to take geometric things
  into account. It also includes normalisation. (internal use only).
  */
  /*
  Use as follows:
  
    const JacobianForIntBP jacobian(*(segment.scanner));
    jacobian(segment.get_average_delta(), s+ 0.5);
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
   explicit JacobianForIntBP(const ProjDataInfoCylindricalArcCorr* proj_data_info_ptr, bool exact)
     
     : R2(square(proj_data_info_ptr->get_ring_radius())),
       dxy2(square(proj_data_info_ptr->get_tangential_sampling())),
       ring_spacing2 (square(proj_data_info_ptr->get_ring_spacing())),
       backprojection_normalisation 
      (proj_data_info_ptr->get_ring_spacing()/2/proj_data_info_ptr->get_num_views()),
      use_exact_Jacobian_now(exact)
      

   {}

   float operator()(const float delta, const float s) const
   {
     float tmp;
     if (use_exact_Jacobian_now)
       tmp = 4*(R2 - dxy2 * s*s);
     else
       tmp = 4*R2;
     return tmp / pow(tmp + ring_spacing2*delta*delta, 1.5)* backprojection_normalisation;
   }
};     


/*!
  \ingroup recon_buildblock
  \brief does backprojection by interpolating between the bins.

  This implementation uses incremental backprojection

  Two versions of interpolation are implemented:
  <ul>
  <li> ordinary linear interpolation
  <li> piecewise linear interpolation in the axial direction
  </ul>
  For the latter, see the extended abstract for 3D99 
  "On various approximations for the projectors in iterative reconstruction algorithms for 
  3D-PET", K. Thielemans, M.W. Jacobson, D. Belluzzo. Available at the
  PARAPET web-site http://www.brunel.ac.uk/~masrppet.

  The piecewise linear interpolation is only used when the axial voxel size is half the
  axial_sampling of the projection data (for the segment in question).

  \warning This implementation makes various assumptions (for optimal speed):
  <ul>
  <li> min_tangential_pos_num == -max_tangential_pos_num,
  <li> arc-corrected data, with bin_size = voxel_size.x() = voxel_size.y()
  <li> voxel_size.z() is either equal to or half the axial_sampling of the projection data
  </ul>

  \warning Currently this implementation has problems on certain processors
  due to floating point rounding errors. Intel *86 and PowerPC give
  correct results, SunSparc has a problem at tangential_pos_num==0.
*/
class BackProjectorByBinUsingInterpolation : public BackProjectorByBin
{ 
public:
  //! The constructor defaults to using picewise linear interpolation and the exact Jacobian 
  BackProjectorByBinUsingInterpolation(
    shared_ptr<ProjDataInfo>const&,
    shared_ptr<DiscretisedDensity<3,float> > const& image_info_ptr,
    const bool use_piecewise_linear_interpolation = true, const bool use_exact_Jacobian = true);

  /*! \brief This BackProjectorByBin implementation requires all symmetries.
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

private:
 
  //shared_ptr<DataSymmetriesForViewSegmentNumbers_PET_CartesianGrid> symmetries;
  DataSymmetriesForViewSegmentNumbers_PET_CartesianGrid symmetries;
  
  bool use_piecewise_linear_interpolation_now;

  bool use_exact_Jacobian_now;

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
                                     float cphi, float sphi, int s, int ring0, 
				     const int num_planes_per_virtual_ring,
				     const float virtual_ring_offset);

 static void piecewise_linear_interpolation_backproj3D_Cho_view_viewplus90_180minview_90minview(Array<4, float > const &Projptr,
                                                         VoxelsOnCartesianGrid<float>& image,							 
							 const ProjDataInfoCylindricalArcCorr* proj_data_info_ptr,
                                                         float delta,
                                                          float cphi, float sphi, int s, int ring0,
                                                          const int num_planes_per_virtual_ring,
							  const float virtual_ring_offset);

  static void linear_interpolation_backproj3D_Cho_view_viewplus90(Array<4, float > const & Projptr,
                                     VoxelsOnCartesianGrid<float>& image,				     
				     const ProjDataInfoCylindricalArcCorr* proj_data_info_ptr,
                                     float delta,
                                     float cphi, float sphi, int s, int ring0, 
				     const int num_planes_per_virtual_ring,
				     const float virtual_ring_offset);

 static void linear_interpolation_backproj3D_Cho_view_viewplus90_180minview_90minview(Array<4, float > const &Projptr,
                                                         VoxelsOnCartesianGrid<float>& image,							 
							 const ProjDataInfoCylindricalArcCorr* proj_data_info_ptr,
                                                         float delta,
                                                          float cphi, float sphi, int s, int ring0,
                                                          const int num_planes_per_virtual_ring,
							  const float virtual_ring_offset);

  /*
static void   backproj2D_Cho_view_viewplus90( PETPlane & image,
                                     const ProjDataForIntBP &projs,
                                    const double cphi, const double sphi, int s);

  static void   backproj2D_Cho_view_viewplus90_180minview_90minview( PETPlane & image,
                                     const ProjDataForIntBP &projs,
                                    const double cphi, const double sphi, int s);

*/

};

END_NAMESPACE_TOMO

//#include "recon_buildblock/BackProjectorByBinUsingInterpolation.inl"

#endif // __BackProjectorByBinUsingInterpolation_h_

