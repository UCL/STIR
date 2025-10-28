/*
    Copyright (C) 2024-2025, Dimitra Kyriakopoulou
    Copyright (C) 2025 University College London

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup GRD2D
  \brief Implementation of class stir::GRD2DReconstruction

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans

*/

#include "stir/analytic/GRD2D/GRD2DReconstruction.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfoCylindrical.h"

// kept for future work, i.e. for the commented out SSRB and arc-correction
//#include "stir/ProjDataInfoCylindricalArcCorr.h"
//#include "stir/ArcCorrection.h"
//#include "stir/SSRB.h"
//#include "stir/ProjDataInMemory.h"

#include "stir/Bin.h"
#include "stir/display.h"

#include "stir/Sinogram.h"
#include <complex>
#include "stir/numerics/fourier.h"
#include "stir/numerics/fftshift.h"
#include <boost/math/special_functions/bessel.hpp>
#include "stir/info.h"
#include "stir/format.h"
#include "stir/error.h"
#include "stir/warning.h"

// kept for future work
//#ifdef STIR_OPENMP
//#include <omp.h>
//#endif
//#include "stir/num_threads.h"

#include <vector>
#include <iostream>
#include <cmath> // For M_PI and other math functions
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

START_NAMESPACE_STIR

void
GRD2DReconstruction::set_defaults()
{
  base_type::set_defaults();

  display_level = 0; // no display
  num_segments_to_combine = -1;
  noise_filter = -1;
  alpha_gridding = 1;
  kappa_gridding = 4;
}

void
GRD2DReconstruction::initialise_keymap()
{
  base_type::initialise_keymap();

  parser.add_start_key("GRD2DParameters");
  parser.add_stop_key("End");
  parser.add_key("num_segments_to_combine with SSRB", &num_segments_to_combine);
  parser.add_key("Display level", &display_level);
  parser.add_key("noise filter", &noise_filter);
  parser.add_key("alpha for gridding", &alpha_gridding);
  parser.add_key("kappa for gridding", &kappa_gridding);
}

void
GRD2DReconstruction::ask_parameters()
{

  base_type::ask_parameters();

  num_segments_to_combine = ask_num("num_segments_to_combine (must be odd)", -1, 101, -1);
  display_level = ask_num("Which images would you like to display \n\t(0: None, 1: Final, 2: filtered viewgrams) ? ", 0, 2, 0);

  noise_filter = ask_num(" Noise filter (-1 to disable)", 0., 1., 1.);
  alpha_gridding = ask_num(" Alpha parameter for gridding ? ", 1., 2., 1.);
  kappa_gridding = ask_num(" Kappa parameter for gridding ? ", 2., 8., 4.);

#if 0
    // do not ask the user for the projectors to prevent them entering
    // silly things
  do 
    {
      back_projector_sptr =
	BackProjectorByBin::ask_type_and_parameters();
    }
#endif
}

bool
GRD2DReconstruction::post_processing()
{
  return base_type::post_processing();
}

Succeeded
GRD2DReconstruction::set_up(shared_ptr<GRD2DReconstruction::TargetT> const& target_data_sptr)
{
  if (base_type::set_up(target_data_sptr) == Succeeded::no)
    return Succeeded::no;

  if (noise_filter > 1)
    error(stir::format("Noise filter has to be between 0 and 1 but is {}", noise_filter));

  if (alpha_gridding < 1 || alpha_gridding > 2)
    error(stir::format("Alpha for gridding has to be between 1 and 2 but is {}", alpha_gridding));

  if (kappa_gridding < 2 || kappa_gridding > 8)
    error(stir::format("Kappa for gridding has to be between 2 and 8 but is {}", kappa_gridding));

  if (num_segments_to_combine >= 0 && num_segments_to_combine % 2 == 0)
    error(stir::format("num_segments_to_combine has to be odd (or -1), but is {}", num_segments_to_combine));

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
GRD2DReconstruction::method_info() const
{
  return "GRD2D";
}

GRD2DReconstruction::GRD2DReconstruction(const std::string& parameter_filename)
{
  initialise(parameter_filename);
  info(parameter_info());
}

GRD2DReconstruction::GRD2DReconstruction()
{
  set_defaults();
}

GRD2DReconstruction::GRD2DReconstruction(const shared_ptr<ProjData>& proj_data_ptr_v,
                                         const double noise_filter_v,
                                         const double alpha_gridding_v,
                                         const double kappa_gridding_v,
                                         const int num_segments_to_combine_v)
{
  set_defaults();

  noise_filter = noise_filter_v;
  alpha_gridding = alpha_gridding_v;
  kappa_gridding = kappa_gridding_v;
  num_segments_to_combine = num_segments_to_combine_v;
  proj_data_ptr = proj_data_ptr_v;
}

Succeeded
GRD2DReconstruction::actual_reconstruct(shared_ptr<DiscretisedDensity<3, float>> const& density_ptr)
{

  // perform SSRB
  /*  if (num_segments_to_combine>1)
      {
        const ProjDataInfoCylindrical& proj_data_info_cyl =
          dynamic_cast<const ProjDataInfoCylindrical&>
          (*proj_data_ptr->get_proj_data_info_sptr());

        //  full_log << "SSRB combining " << num_segments_to_combine
        //           << " segments in input file to a new segment 0\n" << std::endl;

        shared_ptr<ProjDataInfo>
          ssrb_info_sptr(SSRB(proj_data_info_cyl,
                              num_segments_to_combine,
                              1, 0,
                              (num_segments_to_combine-1)/2 ));
        shared_ptr<ProjData>
          proj_data_to_GRD_ptr(new ProjDataInMemory (proj_data_ptr->get_exam_info_sptr(), ssrb_info_sptr));
        SSRB(*proj_data_to_GRD_ptr, *proj_data_ptr);
        proj_data_ptr = proj_data_to_GRD_ptr;
      }
    else
      {
        // just use the proj_data_ptr we have already
      } */

  // check if segment 0 has direct sinograms
  {
    const float tan_theta = proj_data_ptr->get_proj_data_info_sptr()->get_tantheta(Bin(0, 0, 0, 0));
    if (fabs(tan_theta) > 1.E-4)
      {
        warning(stir::format("GRD2D: segment 0 has non-zero tan(theta) {}", tan_theta));
        return Succeeded::no;
      }
  }

  // unused warning
  float tangential_sampling;
  // TODO make next type shared_ptr<ProjDataInfoCylindricalArcCorr> once we moved to boost::shared_ptr
  // will enable us to get rid of a few of the ugly lines related to tangential_sampling below
  shared_ptr<const ProjDataInfo> arc_corrected_proj_data_info_sptr;

  // arc-correction if necessary
  /* ArcCorrection arc_correction;
   bool do_arc_correction = false;
   if (!is_null_ptr(dynamic_pointer_cast<const ProjDataInfoCylindricalArcCorr>
       (proj_data_ptr->get_proj_data_info_sptr())))
     {
       // it's already arc-corrected
       arc_corrected_proj_data_info_sptr =
         proj_data_ptr->get_proj_data_info_sptr()->create_shared_clone();
       tangential_sampling =
         dynamic_cast<const ProjDataInfoCylindricalArcCorr&>
         (*proj_data_ptr->get_proj_data_info_sptr()).get_tangential_sampling();
     }
   else
     {
       // TODO arc-correct to voxel_size
       if (arc_correction.set_up(proj_data_ptr->get_proj_data_info_sptr()->create_shared_clone()) ==
           Succeeded::no)
         return Succeeded::no;
       do_arc_correction = true;
       // TODO full_log
       warning("GRD2D will arc-correct data first");
       arc_corrected_proj_data_info_sptr =
         arc_correction.get_arc_corrected_proj_data_info_sptr();
       tangential_sampling =
         arc_correction.get_arc_corrected_proj_data_info().get_tangential_sampling();
     }*/
  // ProjDataInterfile ramp_filtered_proj_data(arc_corrected_proj_data_info_sptr,"ramp_filtered");

  VoxelsOnCartesianGrid<float>& image = dynamic_cast<VoxelsOnCartesianGrid<float>&>(*density_ptr);

  density_ptr->fill(0);

  Sinogram<float> sino = proj_data_ptr->get_empty_sinogram(0, 0);

  const int sp = sino.get_num_tangential_poss(), sth = sino.get_num_views();
  float dp = 2.0 / sp, dth = M_PI / sth; // dp := normalised detector spacing in p; dth := view angle step

  // pad to the next power of 2 for FFT
  const int sp1 = 2 * pow(2, ceil(log2(sp))); // sp1 := zero-padded FFT length in p (2×next power of two ≥ sp)

  const float alpha = alpha_gridding; // alpha := Kaiser–Bessel window width parameter (≈1..2)
  const float beta = 1.0 * sp1 / sp;  // beta := oversampling factor in p (FFT grid scaling sp1/sp)
  const float K = kappa_gridding;     // K := kernel support (grid radius in k-space, typical 4..8)

  // gridding
  std::vector<float> pn1(sp1); // pn1 := normalized frequency coordinate per FFT bin along p (length sp1, zero-padded)
  std::vector<float> thn(sth); // thn := view-angle grid (radians)
  std::vector<float> xn(sp);   // xn  := normalised image x-grid in [-1,1]
  std::vector<float> yn(sp);   // yn  := normalised image y-grid in [-1,1]
  for (int ith = 0; ith < sth; ith++)
    thn[ith] = ith * M_PI / sth;
  for (int ip = 0; ip < sp1; ip++)
    pn1[ip] = -sp / 4.0 + ip * (sp / 2.0) / (sp1 - 1);
  for (int ix = 0; ix < sp; ix++)
    xn[ix] = -1.0 + 2.0 * ix / (sp - 1);
  for (int iy = 0; iy < sp; iy++)
    yn[iy] = -1.0 + 2.0 * iy / (sp - 1);

  const int min_tang = sino.get_min_tangential_pos_num(); // min_tang := min tangential bin index in sinogram storage
  const int min_xy = image.get_min_x();                   // min_xy := min output image x-index (assumed square grid)

  IndexRange2D span(sino.get_num_views(), sp1); // span := (views × sp1) index range for polar frequency slices
  Array<2, std::complex<float>> P(span);        // P := per-view 1D-FFT spectra mapped into polar frequency bins
  Array<1, float> s(sp1);                       // s := zero-padded projection buffer (real, length sp1)
  Array<1, std::complex<float>> a;              // a := packed half-spectrum from real-FFT (fourier_1d_for_real_data)
  IndexRange2D span2(sp1, sp1);                 // span2 := Cartesian k-space grid size (sp1×sp1)
  Array<2, std::complex<float>> ff(span2);      // ff := Cartesian Fourier accumulator (post-gridding)
  std::complex<float> f1, f2;                   // f1,f2 := temporaries for complex math
  int l1a, l1b, l2a, l2b;                       // l1*,l2* := k-space index bounds for kernel support rectangles

  int p_cutoff; // p_cutoff := symmetric radial frequency taper (derived from noise_filter; 0 disables)
  if (noise_filter <= 0)
    {
      p_cutoff = 0;
    }
  else
    {
      noise_filter = noise_filter > 1 ? 1 : noise_filter;
      p_cutoff = floor(floor(sp1 / 2.0) * (1 - noise_filter));
      std::cout << "p_cutoff = " << 2 * p_cutoff << " of " << sp1 << std::endl;
    }

  float ar = alpha * sp * dp / 2.0f;     // ar := real-space KB parameter (window width scaling)
  float u = K / (2.0f * beta * sp * dp); // u  := half-support of KB kernel in k-space (per axis)

  std::vector<std::vector<float>> PGx(
      sth, std::vector<float>(sp1)); // PGx := Cartesian Fourier-plane x-coordinate (normalized) for each (view, p) polar sample
  std::vector<std::vector<float>> PGy(
      sth, std::vector<float>(sp1)); // PGy := Cartesian Fourier-plane y-coordinate (normalized) for each (view, p) polar sample
  for (int ith = 0; ith < sth; ith++)
    {
      for (int ip = 0; ip < sp1; ip++)
        {
          PGx[ith][ip] = pn1[ip] * cos(thn[ith]);
          PGy[ith][ip] = pn1[ip] * sin(thn[ith]);
        }
    }

  // TODO Bessel window and weights can be calculated only once and be used for every slice

  for (int iz = proj_data_ptr->get_min_axial_pos_num(0); iz <= proj_data_ptr->get_max_axial_pos_num(0); iz++)
    {

      std::cout << "Reconstructing slice " << (iz + 1) << " of " << proj_data_ptr->get_num_axial_poss(0) << "..." << std::endl;
      sino = proj_data_ptr->get_sinogram(iz, 0);

      for (int ith = 0; ith < sth; ith++)
        {
          s.fill(0);
          for (int ip = 0; ip < sp; ip++)
            {
              s[(sp1 - sp) / 2 + ip] = sino[ith][ip + min_tang];
            }

          fftshift(s, sp1);

          a = fourier_1d_for_real_data(s);

          for (int i = 0; i <= sp1 / 2; i++)
            P[ith][i] = a[sp1 / 2 - i];
          for (int i = 1; i < sp1 / 2; i++)
            P[ith][sp1 / 2 + i] = std::conj(a[i]);

          if (p_cutoff > 0)
            {
              for (int i = 0; i < p_cutoff; i++)
                P[ith][i] = 0;
              for (int i = sp1 - p_cutoff; i < sp1; i++)
                P[ith][i] = 0;
            }
        }

      ff.fill(0);
      const float sp1dp = sp1 * dp; // sp1dp := scale factor between normalised frequency units and FFT grid indices (used when
                                    // mapping k <-> array index)
      for (int ith = 0; ith < sth; ith++)
        {
          for (int ip = 0; ip < sp1; ip++)
            {
              l1a = ceil((-u + PGx[ith][ip]) * sp1dp);
              if (l1a < -sp1 / 2)
                l1a = -sp1 / 2;
              l1b = floor((u + PGx[ith][ip]) * sp1dp);
              if (l1b > sp1 / 2 - 1)
                l1b = sp1 / 2 - 1;

              l2a = ceil((-u + PGy[ith][ip]) * sp1dp);
              if (l2a < -sp1 / 2)
                l2a = -sp1 / 2;
              l2b = floor((u + PGy[ith][ip]) * sp1dp);
              if (l2b > sp1 / 2 - 1)
                l2b = sp1 / 2 - 1;

              for (int l1 = l1a; l1 <= l1b; l1++)
                {
                  float T1 = l1 / sp1dp - PGx[ith][ip];
                  float wKB1 = boost::math::cyl_bessel_i(0.0f, 2 * M_PI * ar * u * sqrt(1 - pow(T1 / u, 2))) / (2.0f * u);
                  for (int l2 = l2a; l2 <= l2b; l2++)
                    {
                      float T2 = l2 / sp1dp - PGy[ith][ip];
                      ff[l1 + sp1 / 2][l2 + sp1 / 2]
                          = ff[l1 + sp1 / 2][l2 + sp1 / 2]
                            + fabs(ip - sp1 / 2.0f) * wKB1
                                  * ((float)boost::math::cyl_bessel_i(0.0f, 2 * M_PI * ar * u * sqrt(1 - pow(T2 / u, 2))))
                                  / (2.0f * u) * P[ith][ip];
                    }
                }
            }
        }

      // gridding
      ff = ff * dth / (sp1 * dp);

      fftshift(ff, sp1);
      inverse_fourier(ff);
      fftshift(ff, sp1);

      // gridding
      int spf = 2 * floor(sp / 2);                                       // make dimensions even
                                                                         // float img[spf][spf];
      std::vector<std::vector<float>> img(spf, std::vector<float>(spf)); // img := temporary reconstructed slice on spf×spf grid
      for (int ix = -spf / 2; ix <= spf / 2 - 1; ix++)
        {
          for (int iy = -spf / 2; iy <= spf / 2 - 1; iy++)
            {
              img[ix + spf / 2][iy + spf / 2] = std::real(ff[ix + sp1 / 2][iy + sp1 / 2]
                                                          / ((float)(sinh(2 * M_PI * ar * u * sqrt(1 - pow(ix * dp / ar, 2)))
                                                                     / (2 * M_PI * ar * u * sqrt(1 - pow(ix * dp / ar, 2)))
                                                                     * sinh(2 * M_PI * ar * u * sqrt(1 - pow(iy * dp / ar, 2)))
                                                                     / (2 * M_PI * ar * u * sqrt(1 - pow(iy * dp / ar, 2))))));
            }
        }

      // --- Physical helpers (tangential sampling, voxel size [mm]) and FOV radius
      const auto vox = image.get_voxel_size(); // [z,y,x]
      const float vx = vox[3];
      const float vy = vox[2];
      const int sx_im = image.get_x_size();
      const int sy_im = image.get_y_size();
      const float Rmax = 0.5f * std::min((sx_im - 1) * vx, (sy_im - 1) * vy);

      if (image.get_x_size() != sp)
        {
          // perform bilinear interpolation
          if (iz == 0)
            info(stir::format(
                "Image dimension mismatch: tangential positions {}, xy output {} — interpolating...", sp, image.get_x_size()));

          int sx = image.get_x_size();
          int sy = sx; // sx,sy := output image size (assumed square)
          std::vector<float> xn1(sx);
          std::vector<float> yn1(sy);                             // xn1,yn1 := normalised target grids
          const float cx = 0.5f * (sx - 1), cy = 0.5f * (sy - 1); // pixel-centre coords
          float dx = 2. / (sp - 1), dy = 2. / (sp - 1);           // dx,dy  := source grid steps
          float val;                                              // val := interpolated sample

          // map output pixel centres -> mm -> normalised by Rmax (SRT-style change)
          for (int ix = 0; ix < sx; ++ix)
            {
              const float x_mm = (ix - cx) * vx;
              xn1[ix] = x_mm / Rmax;
            }
          for (int iy = 0; iy < sy; ++iy)
            {
              const float y_mm = (iy - cy) * vy;
              yn1[iy] = y_mm / Rmax;
            }

          for (int ix = 1; ix < sx - 1; ix++)
            {
              for (int iy = 1; iy < sy - 1; iy++)
                {
                  if (pow(xn1[ix], 2) + pow(yn1[iy], 2) > 1)
                    val = 0.;
                  else
                    {
                      // bilinear interpolation
                      int y0 = (int)((yn1[iy] - yn[0]) / dy);
                      int x0 = (int)((xn1[ix] - xn[0]) / dx);
                      float tx = (xn1[ix] - xn[0]) / dx - x0;
                      float ty = (yn1[iy] - yn[0]) / dy - y0;

                      float ya = img[y0][x0] * tx + img[y0][x0 + 1] * (1. - tx);
                      float yb = img[y0 + 1][x0] * tx + img[y0 + 1][x0 + 1] * (1. - tx);
                      val = ya * ty + yb * (1. - ty);
                    }
                  image[iz][sx - 1 + 1 - ix + min_xy][sy - iy + min_xy]
                      = val * pow(1. * sp / sp1, 2.)
                        * 17.97558; // 17.97558: Global empirical scaling factor applied so that the algorithm’s ROI mean matches
                                    // that of FBP2D; this uniformly rescales intensities and doesn’t alter relative contrast or
                                    // image structure.
                }
            }
        }
      else
        {
          const float cx2 = 0.5f * (sp - 1), cy2 = 0.5f * (sp - 1);
          for (int ix = 1; ix <= sp - 1; ++ix)
            {
              for (int iy = 1; iy <= sp - 1; ++iy)
                {
                  const float x_mm = (ix - cx2) * vx;
                  const float y_mm = (iy - cy2) * vy;
                  if (x_mm * x_mm + y_mm * y_mm > Rmax * Rmax)
                    image[iz][ix + min_xy][iy + min_xy] = 0;
                  else
                    image[iz][sp - 1 + 1 - ix + min_xy][sp - iy + min_xy]
                        = img[iy][ix] * sp / sp1
                          * 17.97558; // 17.97558: Global empirical scaling factor applied so that the algorithm’s ROI mean
                                      // matches that of FBP2D; this uniformly rescales intensities and doesn’t alter relative
                                      // contrast or image structure.
                }
            }
        }
    }

  if (display_level > 0)
    display(image, image.find_max(), "GRD image");

  return Succeeded::yes;
}

END_NAMESPACE_STIR
