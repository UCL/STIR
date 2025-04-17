/*
    Copyright (C) 2024 University College London

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup analytic
  \brief Implementation of class stir::SRT2DReconstruction

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans
*/

#include "stir/analytic/SRT2D/SRT2DReconstruction.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/RelatedViewgrams.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ArcCorrection.h"
#include "stir/SSRB.h"
#include "stir/ProjDataInMemory.h"
#include "stir/Bin.h"
#include "stir/round.h"
#include "stir/display.h"
#include <algorithm>
#include "stir/IO/interfile.h"
#include "stir/info.h"
#include "stir/format.h"

#ifdef STIR_OPENMP
#  include <omp.h>
#endif

#include <cmath> // For M_PI and other math functions
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

#ifdef STIR_OPENMP
#  include "stir/num_threads.h"
#endif

#include <vector>
#include <algorithm>

START_NAMESPACE_STIR

const char* const SRT2DReconstruction::registered_name = "SRT2D";

void
SRT2DReconstruction::set_defaults()
{
  base_type::set_defaults();
  num_segments_to_combine = -1;
}

void
SRT2DReconstruction::initialise_keymap()
{
  base_type::initialise_keymap();

  parser.add_start_key("SRT2DParameters");
  parser.add_stop_key("End");
  parser.add_key("num_segments_to_combine with SSRB", &num_segments_to_combine);
}

void
SRT2DReconstruction::ask_parameters()
{
  base_type::ask_parameters();
  num_segments_to_combine = ask_num("num_segments_to_combine (must be odd)", -1, 101, -1);
}

bool
SRT2DReconstruction::post_processing()
{
  return base_type::post_processing();
}

Succeeded
SRT2DReconstruction::set_up(shared_ptr<SRT2DReconstruction::TargetT> const& target_data_sptr)
{
  if (base_type::set_up(target_data_sptr) == Succeeded::no)
    return Succeeded::no;

  if (num_segments_to_combine >= 0 && num_segments_to_combine % 2 == 0)
    error(format("num_segments_to_combine has to be odd (or -1), but is {}", num_segments_to_combine));

  if (num_segments_to_combine == -1)
    {
      const shared_ptr<const ProjDataInfoCylindrical> proj_data_info_cyl_sptr
          = dynamic_pointer_cast<const ProjDataInfoCylindrical>(proj_data_ptr->get_proj_data_info_sptr());

      if (is_null_ptr(proj_data_info_cyl_sptr))
        num_segments_to_combine = 1; // cannot SSRB non-cylindrical data yet
      else
        {
          if (proj_data_info_cyl_sptr->get_min_ring_difference(0) != proj_data_info_cyl_sptr->get_max_ring_difference(0)
              || proj_data_info_cyl_sptr->get_num_segments() == 1)
            num_segments_to_combine = 1;
          else
            num_segments_to_combine = 3;
        }
    }
  return Succeeded::yes;
}

std::string
SRT2DReconstruction::method_info() const
{
  return "SRT2D";
}

SRT2DReconstruction::SRT2DReconstruction(const std::string& parameter_filename)
{
  initialise(parameter_filename);
  info(format("{}", parameter_info()));
}

SRT2DReconstruction::SRT2DReconstruction()
{
  set_defaults();
}

SRT2DReconstruction::SRT2DReconstruction(const shared_ptr<ProjData>& proj_data_ptr_v, const int num_segments_to_combine_v)
{
  set_defaults();
  proj_data_ptr = proj_data_ptr_v;
  num_segments_to_combine = num_segments_to_combine_v;
}

Succeeded
SRT2DReconstruction::actual_reconstruct(shared_ptr<DiscretisedDensity<3, float>> const& density_ptr)
{
  // In case of 3D data, use only direct sinograms
  // perform SSRB
  if (num_segments_to_combine > 1)
    {
      info(format("Performing SSRB with num_segments_to_combine = {}", num_segments_to_combine));
      const ProjDataInfoCylindrical& proj_data_info_cyl
          = dynamic_cast<const ProjDataInfoCylindrical&>(*proj_data_ptr->get_proj_data_info_sptr());

      shared_ptr<ProjDataInfo> ssrb_info_sptr(
          SSRB(proj_data_info_cyl, num_segments_to_combine, 1, 0, (num_segments_to_combine - 1) / 2));
      shared_ptr<ProjData> proj_data_to_SRT_ptr(new ProjDataInMemory(proj_data_ptr->get_exam_info_sptr(), ssrb_info_sptr));
      SSRB(*proj_data_to_SRT_ptr, *proj_data_ptr);
      proj_data_ptr = proj_data_to_SRT_ptr;
    }
  else
    {
      info("Using existing proj_data_ptr");
    }

  // check if segment 0 has direct sinograms
  {
    const float tan_theta = proj_data_ptr->get_proj_data_info_sptr()->get_tantheta(Bin(0, 0, 0, 0));
    if (fabs(tan_theta) > 1.E-4)
      {
        warning("SRT2D: segment 0 has non-zero tan(theta) %g", tan_theta);
        return Succeeded::no;
      }
  }

  float tangential_sampling;
  shared_ptr<const ProjDataInfo> arc_corrected_proj_data_info_sptr;

  // arc-correction if necessary
  ArcCorrection arc_correction;
  bool do_arc_correction = false;
  if (!is_null_ptr(dynamic_pointer_cast<const ProjDataInfoCylindricalArcCorr>(proj_data_ptr->get_proj_data_info_sptr())))
    {
      info("Data is already arc-corrected");
      arc_corrected_proj_data_info_sptr = proj_data_ptr->get_proj_data_info_sptr()->create_shared_clone();
      tangential_sampling = dynamic_cast<const ProjDataInfoCylindricalArcCorr&>(*proj_data_ptr->get_proj_data_info_sptr())
                                .get_tangential_sampling();
    }
  else
    {
      info("Performing arc-correction");
      if (arc_correction.set_up(proj_data_ptr->get_proj_data_info_sptr()->create_shared_clone()) == Succeeded::no)
        return Succeeded::no;
      do_arc_correction = true;
      arc_corrected_proj_data_info_sptr = arc_correction.get_arc_corrected_proj_data_info_sptr();
      tangential_sampling = arc_correction.get_arc_corrected_proj_data_info().get_tangential_sampling();
    }

  VoxelsOnCartesianGrid<float>& image = dynamic_cast<VoxelsOnCartesianGrid<float>&>(*density_ptr);
  Sinogram<float> sino = proj_data_ptr->get_empty_sinogram(0, 0);
  Viewgram<float> view = proj_data_ptr->get_viewgram(0, 0);
  if (do_arc_correction)
    {
      info("Arc-correcting viewgram");
      view = arc_correction.do_arc_correction(view);
    }
  Viewgram<float> view1 = proj_data_ptr->get_empty_viewgram(0, 0);
  Viewgram<float> view_th = proj_data_ptr->get_empty_viewgram(0, 0);
  Viewgram<float> view1_th = proj_data_ptr->get_empty_viewgram(0, 0);

  // Retrieve runtime-dependent sizes
  const int sp = view.get_num_tangential_poss();
  const int sth = proj_data_ptr->get_num_views();
  const int sa = proj_data_ptr->get_num_axial_poss(0);
  const int sx = image.get_x_size();
  const int sy = image.get_y_size();
  const int sx2 = ceil(sx / 2.0), sy2 = ceil(sy / 2.0);
  const int sth2 = ceil(sth / 2.0);
  const float c_int = -1 / (M_PI * sth * (sp - 1)) * 2; // instead of *2 we will write /zoom;

  int ia, image_pos;
  int ith, jth, ip, ix1, ix2, k1, k2;

  float x, aux;

  const int image_min_x = image.get_min_x();
  const int image_min_y = image.get_min_y();

  // Dynamic declarations using std::vector
  std::vector<float> th(sth);  // Theta values for each view angle.
  std::vector<float> p(sp);    // Tangential positions for projections.
  std::vector<float> p_ud(sp); // Tangential positions for up-down flipping.
  std::vector<float> x1(sx);   // X-coordinates for the reconstructed image grid.
  std::vector<float> x2(sy);   // Y-coordinates for the reconstructed image grid.

  std::vector<std::vector<float>> f(sth, std::vector<float>(sp, 0.0f));   // Projection data matrix.
  std::vector<std::vector<float>> ddf(sth, std::vector<float>(sp, 0.0f)); // Second derivative of projections.

  std::vector<std::vector<float>> f_ud(sth, std::vector<float>(sp, 0.0f));   // Flipped projection data.
  std::vector<std::vector<float>> ddf_ud(sth, std::vector<float>(sp, 0.0f)); // Second derivatives of flipped projections.

  std::vector<std::vector<float>> f1(sth, std::vector<float>(sp, 0.0f)); // Left-right mirrored projection data.
  std::vector<std::vector<float>> ddf1(sth,
                                       std::vector<float>(sp, 0.0f)); // Second derivatives of left-right mirrored projections.

  std::vector<std::vector<float>> f1_ud(sth, std::vector<float>(sp, 0.0f)); // Up-down flipped mirrored projection data.
  std::vector<std::vector<float>> ddf1_ud(
      sth, std::vector<float>(sp, 0.0f)); // Second derivatives of up-down flipped mirrored projections.

  std::vector<std::vector<float>> f_th(sth, std::vector<float>(sp, 0.0f));   // Projection data along theta views.
  std::vector<std::vector<float>> ddf_th(sth, std::vector<float>(sp, 0.0f)); // Second derivatives along theta views.

  std::vector<std::vector<float>> f_th_ud(sth, std::vector<float>(sp, 0.0f));   // Flipped data along theta views.
  std::vector<std::vector<float>> ddf_th_ud(sth, std::vector<float>(sp, 0.0f)); // Second derivatives of flipped data.

  std::vector<std::vector<float>> f1_th(sth, std::vector<float>(sp, 0.0f));   // Mirrored data along theta views.
  std::vector<std::vector<float>> ddf1_th(sth, std::vector<float>(sp, 0.0f)); // Second derivatives of mirrored data.

  std::vector<std::vector<float>> f1_th_ud(sth, std::vector<float>(sp, 0.0f));   // Flipped mirrored data along theta views.
  std::vector<std::vector<float>> ddf1_th_ud(sth, std::vector<float>(sp, 0.0f)); // Second derivatives of flipped mirrored data.

  std::vector<float> lg(sp);        // Logarithmic differences for interpolation.
  std::vector<float> termC(sth);    // Correction term for each view.
  std::vector<float> termC_th(sth); // Correction term for theta projections.

  const float dp6 = 6.0 / 4.0 * 2.0 / (sp - 1.0); // Integration constant for second derivatives.

#ifdef STIR_OPENMP
  set_num_threads();
#  pragma omp single
  info("Using OpenMP-version of SRT2D with " + std::to_string(omp_get_num_threads()) + " threads on "
       + std::to_string(omp_get_num_procs()) + " processors.");
#endif

  // Put theta and p in arrays.
  for (ith = 0; ith < sth; ith++)
    th[ith] = ith * M_PI / sth;

  for (ip = 0; ip < sp; ip++)
    p[ip] = -1.0 + 2.0 * ip / (sp - 1);
  for (ip = 0; ip < sp; ip++)
    p_ud[sp - ip - 1] = p[ip];

  //-- Creation of the grid
  for (k1 = 0; k1 < sx; k1++)
    x1[k1] = -1.0 * sx / (sp + 1) + 2.0 * sx / (sp + 1) * k1 / (sx - 1);
  //  x1[k1] = -1.0 * sx / ((sp + 1) * zoom) + k1 * 2.0 * sx / ((sp + 1) * zoom) / (sx - 1);
  for (k2 = 0; k2 < sx; k2++)
    x2[k2] = -1.0 * sx / (sp + 1) + 2.0 * sx / (sp + 1) * k2 / (sx - 1);
  //  x2[k2] = -1.0 * sx / ((sp + 1) * zoom) + k2 * 2.0 * sx / ((sp + 1) * zoom) / (sx - 1);

  // Starting calculations per view
  // 2D algorithm only

  // -----
  // special case of ith=0
  // -----
  for (ia = 0; ia < sa; ia++)
    {
      for (ip = 0; ip < sp; ip++)
        {
          f[ia][ip] = view[view.get_min_axial_pos_num() + ia][view.get_min_tangential_pos_num() + ip];
        }
      spline(p, f[ia], sp, ddf[ia]);
    }

  for (ia = 0; ia < sa; ia++)
    {
      termC[ia] = (ddf[ia][0] * (3 * p[1] - p[0]) + ddf[ia][sp - 1] * (p[sp - 1] - 3.0 * p[sp - 2])) / 4.0;
      for (ip = 0; ip < sp; ip++)
        {
          termC[ia] += dp6 * ddf[ia][ip];
        }
    }

  for (ix1 = 0; ix1 < sx2; ix1++)
    {
      for (ix2 = 0; ix2 < sy2; ix2++)
        {
          aux = sqrt(1.0 - x2[ix2] * x2[ix2]);
          if (fabs(x2[ix2]) >= 1.0 || fabs(x1[ix1]) >= aux)
            {
              continue;
            }
          x = x2[ix2] * cos(th[0]) - x1[ix1] * sin(th[0]);

          for (ip = 0; ip < sp; ip++)
            {
              double val = fabs(x - p[ip]);
              lg[ip] = val < 2e-6 ? 0. : std::log(val);
            }

          for (ia = 0; ia < sa; ia++)
            {
              int img_x = image_min_x + sx - ix1 - 1;
              int img_y = image_min_y + ix2;

              image[ia][img_x][img_y] = -hilbert_der(x, f[ia], ddf[ia], p, sp, lg, termC[ia]) / (M_PI * sth * (sp - 1));
            }
        }
    }

// jth, ia, ip, termC_th, ix2, ix1, aux, x, image_pos
#ifdef STIR_OPENMP
#  pragma omp parallel firstprivate(f,                                                                                           \
                                    ddf,                                                                                         \
                                    f_ud,                                                                                        \
                                    ddf_ud,                                                                                      \
                                    f1,                                                                                          \
                                    ddf1,                                                                                        \
                                    f1_ud,                                                                                       \
                                    ddf1_ud,                                                                                     \
                                    f_th,                                                                                        \
                                    ddf_th,                                                                                      \
                                    f_th_ud,                                                                                     \
                                    ddf_th_ud,                                                                                   \
                                    f1_th,                                                                                       \
                                    ddf1_th,                                                                                     \
                                    f1_th_ud,                                                                                    \
                                    ddf1_th_ud,                                                                                  \
                                    termC,                                                                                       \
                                    termC_th,                                                                                    \
                                    lg)                                                                                          \
      shared(view, view1, view_th, view1_th, do_arc_correction, arc_correction, p_ud, p, th, x1, x2, image) private(             \
          jth, ia, ip, ix2, ix1, aux, x, image_pos)
#  pragma omp for schedule(static) nowait
#endif
  for (ith = 1; ith < sth; ith++)
    {
      if (ith < sth2)
        {
          jth = sth2 - ith;
        }
      else if (ith > sth2)
        {
          jth = (int)ceil(3 * sth / 2.0) - ith; // MARK integer division
        }
      else
        {
          jth = sth2;
        }

        // Loading related viewgrams
#ifdef STIR_OPENMP
#  pragma omp critical
#endif
      {
        view = proj_data_ptr->get_viewgram(ith, 0);
        view1 = proj_data_ptr->get_viewgram(sth - ith, 0);
        view_th = proj_data_ptr->get_viewgram(jth, 0);
        view1_th = proj_data_ptr->get_viewgram(sth - jth, 0);
        if (do_arc_correction)
          {
            view = arc_correction.do_arc_correction(view);
            view1 = arc_correction.do_arc_correction(view1);
            view_th = arc_correction.do_arc_correction(view_th);
            view1_th = arc_correction.do_arc_correction(view1_th);
          }

        for (ia = 0; ia < sa; ia++)
          {
            for (ip = 0; ip < sp; ip++)
              {
                f[ia][ip] = view[view.get_min_axial_pos_num() + ia][view.get_min_tangential_pos_num() + ip];
                f_ud[ia][sp - ip - 1] = f[ia][ip];
                f1[ia][ip] = view1[view.get_min_axial_pos_num() + ia][view.get_min_tangential_pos_num() + ip];
                f1_ud[ia][sp - ip - 1] = f1[ia][ip];

                f_th[ia][ip] = view_th[view.get_min_axial_pos_num() + ia][view.get_min_tangential_pos_num() + ip];
                f_th_ud[ia][sp - ip - 1] = f_th[ia][ip];
                f1_th[ia][ip] = view1_th[view.get_min_axial_pos_num() + ia][view.get_min_tangential_pos_num() + ip];
                f1_th_ud[ia][sp - ip - 1] = f1_th[ia][ip];
              }
          }
      }

      // Calculation of second derivative by use of function spline
      for (ia = 0; ia < sa; ia++)
        {
          spline(p, f[ia], sp, ddf[ia]);
          spline(p, f1[ia], sp, ddf1[ia]);
          spline(p, f_th[ia], sp, ddf_th[ia]);
          spline(p, f1_th[ia], sp, ddf1_th[ia]);
          for (ip = 0; ip < sp; ip++)
            {
              ddf_ud[ia][sp - ip - 1] = ddf[ia][ip];
              ddf1_ud[ia][sp - ip - 1] = ddf1[ia][ip];
              ddf_th_ud[ia][sp - ip - 1] = ddf_th[ia][ip];
              ddf1_th_ud[ia][sp - ip - 1] = ddf1_th[ia][ip];
            }
        }

      for (ia = 0; ia < sa; ia++)
        {
          termC[ia] = (ddf[ia][0] * (3 * p[1] - p[0]) + ddf[ia][sp - 1] * (p[sp - 1] - 3.0 * p[sp - 2])) / 4.0;
          for (ip = 0; ip < sp; ip++)
            {
              termC[ia] += dp6 * ddf[ia][ip];
            }
          termC_th[ia] = (ddf_th[ia][0] * (3 * p[1] - p[0]) + ddf_th[ia][sp - 1] * (p[sp - 1] - 3.0 * p[sp - 2])) / 4.0;
          for (ip = 0; ip < sp; ip++)
            {
              termC_th[ia] += dp6 * ddf_th[ia][ip];
            }
        }

      // Starting the calculation of ff(x1,x2).
      for (ix1 = 0; ix1 < sx2; ix1++)
        {
          for (ix2 = 0; ix2 <= ix1; ix2++)
            {
              aux = sqrt(1.0 - x2[ix2] * x2[ix2]);
              if (fabs(x2[ix2]) >= 1.0 || fabs(x1[ix1]) >= aux)
                {
                  continue;
                }

              // Computation of h_rho
              x = x2[ix2] * cos(th[ith]) - x1[ix1] * sin(th[ith]);
              for (ip = 0; ip < sp; ip++)
                {
                  double val = fabs(x - p[ip]);
                  lg[ip] = val < 2e-6 ? 0. : std::log(val); // Using std::log to specify the namespace
                }

              for (ia = 0; ia < sa; ia++)
                {
                  image_pos = ia; // 2*ia;

                  image[image_pos][image_min_x + sx - ix1 - 1][image_min_y + ix2]
                      += hilbert_der(x, f[ia], ddf[ia], p, sp, lg, termC[ia]) * c_int; // bot-left
                  if (ix2 < sy2 - 1)
                    {
                      image[image_pos][image_min_x + sx - ix1 - 1][image_min_y + sy - ix2 - 1]
                          += hilbert_der(x, f1[ia], ddf1[ia], p, sp, lg, termC[ia]) * c_int; // bot-right
                    }
                  if (ix1 < sx2 - 1)
                    {
                      image[image_pos][image_min_x + ix1][image_min_y + ix2]
                          -= hilbert_der(-x, f1_ud[ia], ddf1_ud[ia], p_ud, sp, lg, termC[ia]) * c_int; // top-left
                    }
                  if ((ix1 < sx2 - 1) && (ix2 < sy2 - 1))
                    {
                      image[image_pos][image_min_x + ix1][image_min_y + sy - ix2 - 1]
                          -= hilbert_der(-x, f_ud[ia], ddf_ud[ia], p_ud, sp, lg, termC[ia]) * c_int; // top-right
                    }

                  if (ith <= sth2 && ix1 != ix2)
                    {
                      image[image_pos][image_min_x + sx - ix2 - 1][image_min_y + ix1]
                          -= hilbert_der(-x, f_th_ud[ia], ddf_th_ud[ia], p_ud, sp, lg, termC_th[ia]) * c_int; // bot-left
                      if (ix2 < sy2 - 1)
                        {
                          image[image_pos][image_min_x + ix2][image_min_y + ix1]
                              += hilbert_der(x, f1_th[ia], ddf1_th[ia], p, sp, lg, termC_th[ia]) * c_int; // bot-right
                        }
                      if (ix1 < sx2 - 1)
                        {
                          image[image_pos][image_min_x + sx - ix2 - 1][image_min_y + sx - ix1 - 1]
                              -= hilbert_der(-x, f1_th_ud[ia], ddf1_th_ud[ia], p_ud, sp, lg, termC_th[ia]) * c_int; // top-left
                        }
                      if ((ix1 < sx2 - 1) && (ix2 < sy2 - 1))
                        {
                          image[image_pos][image_min_x + ix2][image_min_y + sx - ix1 - 1]
                              += hilbert_der(x, f_th[ia], ddf_th[ia], p, sp, lg, termC_th[ia]) * c_int; // top-right
                        }
                    }
                  else if (ith > sth2 && ix1 != ix2)
                    {
                      image[image_pos][image_min_x + sx - ix2 - 1][image_min_y + ix1]
                          += hilbert_der(x, f_th[ia], ddf_th[ia], p, sp, lg, termC_th[ia]) * c_int; // bot-left
                      if (ix2 < sy2 - 1)
                        {
                          image[image_pos][image_min_x + ix2][image_min_y + ix1]
                              -= hilbert_der(-x, f1_th_ud[ia], ddf1_th_ud[ia], p_ud, sp, lg, termC_th[ia]) * c_int; // bot-right
                        }
                      if (ix1 < sx2 - 1)
                        {
                          image[image_pos][image_min_x + sx - ix2 - 1][image_min_y + sx - ix1 - 1]
                              += hilbert_der(x, f1_th[ia], ddf1_th[ia], p, sp, lg, termC_th[ia]) * c_int; // top-left
                        }
                      if ((ix1 < sx2 - 1) && (ix2 < sy2 - 1))
                        {
                          image[image_pos][image_min_x + ix2][image_min_y + sx - ix1 - 1]
                              -= hilbert_der(-x, f_th_ud[ia], ddf_th_ud[ia], p_ud, sp, lg, termC_th[ia]) * c_int; // top-right
                        }
                    }
                }
            }
        }
    }

  return Succeeded::yes;
}

float
SRT2DReconstruction::hilbert_der(float x,
                                 const std::vector<float>& f,
                                 const std::vector<float>& ddf,
                                 const std::vector<float>& p,
                                 int sp,
                                 const std::vector<float>& lg,
                                 float termC) const
{
  float term, trm0, termd0;

  const float d = p[1] - p[0];
  const float d_div_6 = d / 6.0;
  const float minus_half_div_d = -0.5 / d;

  term = 0.5 * (ddf[sp - 2] - ddf[0]) * x + termC;

  term += ((f[sp - 1] - f[sp - 2]) / d + ddf[sp - 2] * (d_div_6 + minus_half_div_d * (p[sp - 1] - x) * (p[sp - 1] - x))
           + ddf[sp - 1] * (-d_div_6 - minus_half_div_d * (p[sp - 2] - x) * (p[sp - 2] - x)))
          * lg[sp - 1];

  trm0 = d_div_6 + minus_half_div_d * (p[1] - x) * (p[1] - x);
  termd0 = (f[1] - f[0]) / d + ddf[0] * trm0 + ddf[1] * (-d_div_6 - minus_half_div_d * (p[0] - x) * (p[0] - x));

  term -= termd0 * lg[0];

  for (int ip = 0; ip < sp - 2; ip++)
    {
      float trm1 = d_div_6 + minus_half_div_d * (p[ip + 2] - x) * (p[ip + 2] - x);
      float termd = (f[ip + 2] - f[ip + 1]) / d + ddf[ip + 1] * trm1 - ddf[ip + 2] * trm0;
      term += (termd0 - termd) * lg[ip + 1];
      termd0 = termd;
      trm0 = trm1;
    }

  return term;
}

void
SRT2DReconstruction::spline(const std::vector<float>& x, const std::vector<float>& y, int n, std::vector<float>& y2) const
{
  int i, k;
  float qn, un;
  std::vector<float> u(n);
  y2[0] = 0.0;
  u[0] = 0.0;
  for (i = 1; i < n - 1; i++)
    {
      float sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
      float p = sig * y2[i - 1] + 2.0;
      y2[i] = (sig - 1.0) / p;
      u[i] = (6.0 * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1])) / (x[i + 1] - x[i - 1])
              - sig * u[i - 1])
             / p;
    }
  qn = 0.0;
  un = 0.0;
  y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0);
  for (k = n - 2; k >= 0; k--)
    y2[k] = y2[k] * y2[k + 1] + u[k];
  return;
}

float
SRT2DReconstruction::integ(float dist, int max, const std::vector<float>& ff) const
{
  int k, intg;
  intg = ff[0];
  for (k = 1; k < max; k++)
    {
      intg += ff[k];
    }
  return intg * dist / max;
}

END_NAMESPACE_STIR
