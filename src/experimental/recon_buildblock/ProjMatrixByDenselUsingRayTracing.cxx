//
//
/*!

  \file
  \ingroup recon_buildblock

  \brief non-inline implementations for ProjMatrixByDenselUsingRayTracing

  \author Kris Thielemans

*/
/*
    Copyright (C) 2000- 2012, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/



#include "stir_experimental/recon_buildblock/ProjMatrixByDenselUsingRayTracing.h"
#include "stir_experimental/recon_buildblock/DataSymmetriesForDensels_PET_CartesianGrid.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include <algorithm>
#include <math.h>

#ifndef STIR_NO_NAMESPACES
using std::max;
using std::min;
#endif

START_NAMESPACE_STIR


const char * const 
ProjMatrixByDenselUsingRayTracing::registered_name =
  "Ray Tracing";

ProjMatrixByDenselUsingRayTracing::
ProjMatrixByDenselUsingRayTracing()
{
  set_defaults();
}

void 
ProjMatrixByDenselUsingRayTracing::initialise_keymap()
{
  parser.add_start_key("Ray Tracing Matrix Parameters");
  //parser.add_key("restrict to cylindrical FOV", &restrict_to_cylindrical_FOV);
  parser.add_key("number of rays in tangential direction to trace for each bin",
                  &num_tangential_LORs);
  parser.add_key("use actual detector boundaries", &use_actual_detector_boundaries);
parser.add_stop_key("End Ray Tracing Matrix Parameters");
}


void
ProjMatrixByDenselUsingRayTracing::set_defaults()
{
  //ProjMatrixByDensel::set_defaults();
  //restrict_to_cylindrical_FOV = true;
  num_tangential_LORs = 1;
  use_actual_detector_boundaries = false;
}


bool
ProjMatrixByDenselUsingRayTracing::post_processing()
{
  //if (ProjMatrixByDensel::post_processing() == true)
  //  return true;
  if (num_tangential_LORs<1)
  {
    warning("ProjMatrixByDenselUsingRayTracing: num_tangential_LORs should be at least 1, but is %d\n",
            num_tangential_LORs);
    return true;
  }
  return false;
}

const DataSymmetriesForDensels*
ProjMatrixByDenselUsingRayTracing:: get_symmetries_ptr() const
{
  return  symmetries_ptr.get();
}

void
ProjMatrixByDenselUsingRayTracing::
set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    )
{
  base_type::set_up(proj_data_info_ptr_v, density_info_ptr);
  const VoxelsOnCartesianGrid<float> * image_info_ptr =
    dynamic_cast<const VoxelsOnCartesianGrid<float>*> (density_info_ptr.get());

  if (image_info_ptr == NULL)
    error("ProjMatrixByDenselUsingRayTracing initialised with a wrong type of DiscretisedDensity\n");

 
  voxel_size = image_info_ptr->get_voxel_size();
  origin = image_info_ptr->get_origin();
  image_info_ptr->get_regular_range(min_index, max_index);


  for (int segment_num = proj_data_info_ptr->get_min_segment_num();
       segment_num <= proj_data_info_ptr->get_max_segment_num();
       ++segment_num)
  {
     Bin bin (segment_num,0,0,0);
     if (fabs(proj_data_info_ptr->get_sampling_in_m(bin) / voxel_size.z() - 1)> .001)
       error("ProjMatrixByDenselUsingRayTracing used for pixel size (in z) which is "
       "not equal to the axial sampling (you're probably not using axially compressed data). I can't handle "
       "this yet. Sorry.\n");
  }

  symmetries_ptr
    .reset(new DataSymmetriesForDensels_PET_CartesianGrid(proj_data_info_ptr,
						   density_info_ptr));
  const float sampling_distance_of_adjacent_LORs_xy =
    proj_data_info_ptr->get_sampling_in_s(Bin(0,0,0,0));
  
  if(sampling_distance_of_adjacent_LORs_xy > voxel_size.x() + 1.E-3 ||
     sampling_distance_of_adjacent_LORs_xy > voxel_size.y() + 1.E-3)
     warning("WARNING: ProjMatrixByDenselUsingRayTracing used for pixel size (in x,y) "
             "that is smaller than the densel size.\n"
             "This matrix will completely miss some voxels for some (or all) views.\n");
  if(sampling_distance_of_adjacent_LORs_xy < voxel_size.x() - 1.E-3 ||
     sampling_distance_of_adjacent_LORs_xy < voxel_size.y() - 1.E-3)
     warning("WARNING: ProjMatrixByDenselUsingRayTracing used for pixel size (in x,y) "
             "that is larger than the densel size.\n"
             "Backprojecting with this matrix will have artefacts at views 0 and 90 degrees.\n");

  xhalfsize = voxel_size.x()/2;
  yhalfsize = voxel_size.y()/2;
  zhalfsize = voxel_size.z()/2;
};

#if 0
/* this is used when 
   (tantheta==0 && sampling_distance_of_adjacent_LORs_z==2*voxel_size.z())
  it adds two  adjacents z with their half value
  */
static void 
add_adjacent_z(ProjMatrixElemsForOneDensel& probs);

/* Complicated business to add the same values at z+1
   while taking care that the (x,y,z) coordinates remain unique in the LOR.
  (If you copy the LOR somewhere else, you can simply use 
   ProjMatrixElemsForOneDensel::merge())
*/         
static void merge_zplus1(ProjMatrixElemsForOneDensel& probs);
#endif
static inline int sign(const float x) 
{ return x>=0 ? 1 : - 1; }


// for positive halfsizes, this is valid for 0<=phi<=Pi/2 && 0<theta<Pi/2
static inline float 
  projection_of_voxel_help(const float xctr, const float yctr, const float zctr,
                      const float xhalfsize, const float yhalfsize, const float zhalfsize,
                      const float ctheta, const float tantheta, 
                      float cphi, float sphi,
                      const float m, const float s)
{
  const float epsilon = 1.E-4F;
  if (fabs(sphi)<epsilon)
    sphi=sign(sphi)*epsilon;
  else if (fabs(cphi)<epsilon)
    cphi=sign(cphi)*epsilon;
  const float zs = zctr - m; 
  const float ys = yctr - s*cphi;
  const float xs = xctr + s*sphi;
  return
    max((-max((zs - zhalfsize)/tantheta, 
            max((ys - yhalfsize)/sphi, (xs - xhalfsize)/cphi)) +
         min((zs + zhalfsize)/tantheta, 
            min((ys + yhalfsize)/sphi, (xs + xhalfsize)/cphi)))/ctheta,
     0.F);
}

// for positive halfsizes, this is valid for 0<=phi<=Pi/2 && 0==theta
static inline float 
  projection2D_of_voxel_help(const float xctr, const float yctr, const float zctr,
                      const float xhalfsize, const float yhalfsize, const float zhalfsize,
                      float cphi, float sphi,
                      const float m, const float s)
{
  const float epsilon = 1.E-4F;
  
  if (zhalfsize - fabs(zctr - m) <= 0)
    return 0.F;
#if 1
  if (fabs(sphi)<epsilon)
    sphi=sign(sphi)*epsilon;
  else if (fabs(cphi)<epsilon)
    cphi=sign(cphi)*epsilon;
#else
  // should work, but doesn't
  if (fabs(sphi)<epsilon)
    return (yhalfsize - fabs(yctr-s)) <= 0 ? 0 : 2*xhalfsize;
  if (fabs(cphi)<epsilon)
    return (xhalfsize - fabs(xctr-s)) <= 0 ? 0 : 2*yhalfsize;
#endif
  const float ys = yctr - s*cphi;
  const float xs = xctr + s*sphi;
  return
    max(-max((ys - yhalfsize)/sphi, (xs - xhalfsize)/cphi) +
         min((ys + yhalfsize)/sphi, (xs + xhalfsize)/cphi),
     0.F);
}


static inline float 
  projection_of_voxel(const CartesianCoordinate3D<float>& densel_ctr,
                      const float xhalfsize, const float yhalfsize, const float zhalfsize,
                      const float ctheta, const float tantheta, 
                      const float cphi, const float sphi,
                      const float m, const float s)
{
  // if you want to relax the next assertion, replace yhalfsize with sign(sphi)*yhalfsize below
  //assert(sphi>0);
  return
    fabs(tantheta)<1.E-4 ?
       projection2D_of_voxel_help(densel_ctr.x(), densel_ctr.y(), densel_ctr.z(),
                                  sign(cphi)*xhalfsize, sign(sphi)*yhalfsize, zhalfsize,
                                  cphi, sphi,
                                  m, s)
                                  :
       projection_of_voxel_help(densel_ctr.x(), densel_ctr.y(), densel_ctr.z(),
                                  sign(cphi)*xhalfsize, sign(sphi)*yhalfsize, sign(tantheta)*zhalfsize,
                                  ctheta, tantheta,
                                  cphi, sphi,
                                  m, s);
}

#if 0
static inline float 
  projection_of_voxel(const CartesianCoordinate3D<float>& densel_ctr,
                      const float xhalfsize, const float yhalfsize, const float zhalfsize,
                      const Bin& bin, const ProjDataInfo& proj_data_info)
#endif

float
ProjMatrixByDenselUsingRayTracing::
get_element(const Bin& bin, 
	    const CartesianCoordinate3D<float>& densel_ctr) const
{
  const ProjDataInfo& proj_data_info = *proj_data_info_ptr;

  const float tantheta = proj_data_info.get_tantheta(bin);
  const float costheta = 1/sqrt(1+square(tantheta));
  const float m = proj_data_info.get_t(bin)/costheta;

  float phi;
  float s_in_mm = proj_data_info_ptr->get_s(bin);
  /* Implementation note.
     KT initialised s_in_mm above instead of in the if because this meant
     that gcc 3.0.1 generated identical results to the previous version of this
file.
     Otherwise, some pixels at the boundary appear to be treated differently
     (probably due to different floating point rounding errors), at least
     on Linux on x86.
     A bit of a mistery that.
  */
  if (!use_actual_detector_boundaries)
  {
    phi = proj_data_info_ptr->get_phi(bin);
    //s_in_mm = proj_data_info_ptr->get_s(bin);
  }
  else
  {
    // can be static_cast later on
    const ProjDataInfoCylindricalNoArcCorr& proj_data_info_noarccor =
    dynamic_cast<const ProjDataInfoCylindricalNoArcCorr&>(*proj_data_info_ptr);
    // TODO check on 180 degrees for views
    const int num_detectors =
      proj_data_info_ptr->get_scanner_ptr()->get_num_detectors_per_ring();
    const float ring_radius =
      proj_data_info_ptr->get_scanner_ptr()->get_effective_ring_radius();

    int det_num1=0, det_num2=0;
    proj_data_info_noarccor.
      get_det_num_pair_for_view_tangential_pos_num(det_num1,
                                                 det_num2,
                                                 bin.view_num(),
                                                 bin.tangential_pos_num());
    phi = (det_num1+det_num2)*_PI/num_detectors-_PI/2;

    s_in_mm = ring_radius*sin((det_num1-det_num2)*_PI/num_detectors+_PI/2);
  }

  // phi in KT's Mathematica conventions
  const float phiKT = phi + _PI/2; 
  const float cphi = cos(phiKT);
  const float sphi = sin(phiKT);
 
  // KT TODOCHECK
  s_in_mm *= -1;
  float res = 0;
 
  if (num_tangential_LORs == 1)
  {
    res = 
    projection_of_voxel(densel_ctr,
                        xhalfsize, yhalfsize, zhalfsize,
                        costheta, tantheta,
                        cphi, sphi,
                        m, s_in_mm);
  }
  else
  {
    // get_sampling_in_s returns sampling in interleaved case
    // interleaved case has a sampling which is twice as high
    const float s_inc =
       (!use_actual_detector_boundaries ? 1 : 2) *
        proj_data_info_ptr->get_sampling_in_s(bin)/num_tangential_LORs;
    float current_s_in_mm =
        s_in_mm - s_inc*(num_tangential_LORs-1)/2.F;
    bool found_non_zero = false;
    for (int s_LOR_num=1; s_LOR_num<=num_tangential_LORs; ++s_LOR_num, current_s_in_mm+=s_inc) 
    {
      const float current_res =
        projection_of_voxel(densel_ctr,
                        xhalfsize, yhalfsize, zhalfsize,
                        costheta, tantheta,
                        cphi, sphi,
                        m, current_s_in_mm);
      if (current_res>0)
      {
       found_non_zero = true;
       res += current_res;
      }
      else if (found_non_zero)
      {
        // we've found non_zeroes, but this is one IS 0, so all the next ones
        // will be 0 as well. So, we get out.
        continue;
      }
   }
  }
  
  return 
     (res > xhalfsize/1000.) ? 
	res 
#ifndef NEWSCALE
                /voxel_size.x() // normalise to some kind of 'pixel units'
#endif

	: 0;
}

END_NAMESPACE_STIR

