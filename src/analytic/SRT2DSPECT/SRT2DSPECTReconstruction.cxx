/*
    Copyright (C) 2024 University College London

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup analytic
  \brief Implementation of class stir::SRT2DSPECTReconstruction

  \author Dimitra Kyriakopoulou
*/

#include "stir/analytic/SRT2DSPECT/SRT2DSPECTReconstruction.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/SSRB.h"
#include "stir/ProjDataInMemory.h"
#include "stir/Array.h"
#include <vector>
#include "stir/Sinogram.h"
#include "stir/Viewgram.h"
#include <math.h>
#include "stir/Bin.h"
#include "stir/round.h"
#include "stir/display.h"
#include <algorithm>
#include "stir/IO/interfile.h"
#include "stir/info.h"
#include <boost/format.hpp>

#include "stir/SegmentByView.h"
#include "stir/ArcCorrection.h"
#include "stir/shared_ptr.h"

#ifdef STIR_OPENMP
#  include <omp.h>
#endif
#include "stir/num_threads.h"
using std::cerr;
using std::endl;

#include <cmath> // For M_PI and other math functions
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

START_NAMESPACE_STIR

const char* const SRT2DSPECTReconstruction::registered_name = "SRT2DSPECT";

void
SRT2DSPECTReconstruction::set_defaults()
{
  base_type::set_defaults();
  attenuation_filename = "";
  num_segments_to_combine = -1;
  filter_wiener = 0;
  filter_median = 0;
  filter_gamma = 0;
}

void
SRT2DSPECTReconstruction::initialise_keymap()
{
  base_type::initialise_keymap();

  parser.add_start_key("SRT2DSPECTParameters");
  parser.add_stop_key("End");
  parser.add_key("num_segments_to_combine with SSRB", &num_segments_to_combine);
  parser.add_key("attenuation filename", &attenuation_filename);
  parser.add_key("wiener filter", &filter_wiener);
  parser.add_key("median filter", &filter_median);
  parser.add_key("gamma filter", &filter_gamma);
}

void
SRT2DSPECTReconstruction::ask_parameters()
{
  base_type::ask_parameters();
  num_segments_to_combine = ask_num("num_segments_to_combine (must be odd)", -1, 101, -1);
  attenuation_filename = ask_string("attenuation filename");
}

bool
SRT2DSPECTReconstruction::post_processing()
{
  return base_type::post_processing();
}

Succeeded
SRT2DSPECTReconstruction::set_up(shared_ptr<SRT2DSPECTReconstruction::TargetT> const& target_data_sptr)
{
  if (base_type::set_up(target_data_sptr) == Succeeded::no)
    return Succeeded::no;
  atten_data_ptr = ProjData::read_from_file(attenuation_filename);

  if (num_segments_to_combine >= 0 && num_segments_to_combine % 2 == 0)
    error(boost::format("num_segments_to_combine has to be odd (or -1), but is %d") % num_segments_to_combine);

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

string
SRT2DSPECTReconstruction::method_info() const
{
  return "SRT2DSPECT";
}

SRT2DSPECTReconstruction::SRT2DSPECTReconstruction(const std::string& parameter_filename)
{
  initialise(parameter_filename);
  info(boost::format("%1%") % parameter_info());
}

SRT2DSPECTReconstruction::SRT2DSPECTReconstruction()
{
  set_defaults();
}

SRT2DSPECTReconstruction::SRT2DSPECTReconstruction(const shared_ptr<ProjData>& proj_data_ptr_v,
                                                   const int num_segments_to_combine_v,
                                                   const int filter_wiener_v,
                                                   const int filter_median_v,
                                                   const int filter_gamma_v)
{
  set_defaults();
  proj_data_ptr = proj_data_ptr_v;
  num_segments_to_combine = num_segments_to_combine_v;
  filter_wiener = filter_wiener_v;
  filter_median = filter_median_v;
  filter_gamma = filter_gamma_v;
}

Succeeded
SRT2DSPECTReconstruction::actual_reconstruct(shared_ptr<DiscretisedDensity<3, float>> const& density_ptr)
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
          proj_data_to_SRT_ptr(new ProjDataInMemory (proj_data_ptr->get_exam_info_sptr(), ssrb_info_sptr));
        SSRB(*proj_data_to_SRT_ptr, *proj_data_ptr);
        proj_data_ptr = proj_data_to_SRT_ptr;
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
        warning("SRT2DSPECT: segment 0 has non-zero tan(theta) %g", tan_theta);
        return Succeeded::no;
      }
  }

  float tangential_sampling;
  // TODO make next type shared_ptr<ProjDataInfoCylindricalArcCorr> once we moved to boost::shared_ptr
  // will enable us to get rid of a few of the ugly lines related to tangential_sampling below
  shared_ptr<const ProjDataInfo> arc_corrected_proj_data_info_sptr;

  // arc-correction if necessary
  ArcCorrection arc_correction;
  bool do_arc_correction = false;
  if (!is_null_ptr(dynamic_pointer_cast<const ProjDataInfoCylindricalArcCorr>(proj_data_ptr->get_proj_data_info_sptr())))
    {
      // it's already arc-corrected
      arc_corrected_proj_data_info_sptr = proj_data_ptr->get_proj_data_info_sptr()->create_shared_clone();
      tangential_sampling = dynamic_cast<const ProjDataInfoCylindricalArcCorr&>(*proj_data_ptr->get_proj_data_info_sptr())
                                .get_tangential_sampling();
    }
  else
    {
      // TODO arc-correct to voxel_size
      if (arc_correction.set_up(proj_data_ptr->get_proj_data_info_sptr()->create_shared_clone()) == Succeeded::no)
        return Succeeded::no;
      do_arc_correction = true;
      // TODO full_log
      warning("SRT2DSPECT will arc-correct data first");
      arc_corrected_proj_data_info_sptr = arc_correction.get_arc_corrected_proj_data_info_sptr();
      tangential_sampling = arc_correction.get_arc_corrected_proj_data_info().get_tangential_sampling();
    }

  VoxelsOnCartesianGrid<float>& image = dynamic_cast<VoxelsOnCartesianGrid<float>&>(*density_ptr);
  density_ptr->fill(0);
  Sinogram<float> sino = proj_data_ptr->get_empty_sinogram(0, 0);
  Viewgram<float> view = proj_data_ptr->get_empty_viewgram(0, 0);
  Viewgram<float> view_atten = atten_data_ptr->get_empty_viewgram(0, 0);

  // Retrieve runtime-dependent sizes
  const int sp = proj_data_ptr->get_num_tangential_poss();
  const int sth = proj_data_ptr->get_num_views();
  const int sa = proj_data_ptr->get_num_axial_poss(0);

  //	RelatedViewgrams<float> viewgrams;

  const int sx = image.get_x_size();
  const int sy = image.get_y_size();

  // c ----------------------------------------------
  // c The rest of the variables used by the program.
  // c ----------------------------------------------
  int i, j, k1, k2;

  float aux, a, b, f_node;

  const int image_min_x = image.get_min_x();
  const int image_min_y = image.get_min_y();

  std::vector<float> th(sth, 0), p(sp, 0), x1(sx, 0), x2(sy, 0);
  std::vector<std::vector<float>> g(sa, std::vector<float>(sp, 0));
  std::vector<std::vector<float>> ddg(sa, std::vector<float>(sp, 0));

  const int Nt = 8, Nmul = sth / Nt;
  std::vector<float> lg(sp, 0);

  float dh1[Nt], dh2[Nt], t[Nt];

  std::vector<std::vector<float>> hilb(sa, std::vector<float>(sp, 0));
  std::vector<std::vector<float>> fcpe(sa, std::vector<float>(sp, 0));
  std::vector<std::vector<float>> fspe(sa, std::vector<float>(sp, 0));
  std::vector<std::vector<float>> fc(sa, std::vector<float>(sp, 0));
  std::vector<std::vector<float>> fs(sa, std::vector<float>(sp, 0));
  std::vector<std::vector<float>> ddfc(sa, std::vector<float>(sp, 0));
  std::vector<std::vector<float>> ddfs(sa, std::vector<float>(sp, 0));

  std::vector<std::vector<float>> f(sa, std::vector<float>(sp, 0));
  std::vector<std::vector<float>> ddf(sa, std::vector<float>(sp, 0));

  float rho, h, fcme_fin, fsme_fin, fc_fin, fs_fin, fcpe_fin, fspe_fin, hc_fin, hs_fin, I, Ft1, Ft2, rho1, rho2, tau, tau1, tau2,
      rx1, rx2;
  float gx, w, F;

  std::vector<std::vector<float>> lg1_cache(Nt / 2, std::vector<float>(sp - 1, 0));
  std::vector<std::vector<float>> lg2_cache(Nt / 2, std::vector<float>(sp - 1, 0));

  float*** rx1x2th = reinterpret_cast<float***>(malloc(sa * sizeof(float**)));
  for (int i = 0; i < sa; i++)
    {
      rx1x2th[i] = reinterpret_cast<float**>(malloc(sx * sizeof(float*)));
      for (int j = 0; j < sx; j++)
        {
          rx1x2th[i][j] = reinterpret_cast<float*>(malloc(sy * sizeof(float)));
          for (int k = 0; k < sy; k++)
            {
              rx1x2th[i][j][k] = 0.; // Initialize to zero
            }
        }
    }

  std::vector<std::vector<std::vector<float>>> f_cache(sa, std::vector<std::vector<float>>(Nt / 2, std::vector<float>(sp, 0)));
  std::vector<std::vector<std::vector<float>>> ddf_cache(sa, std::vector<std::vector<float>>(Nt / 2, std::vector<float>(sp, 0)));
  std::vector<std::vector<std::vector<float>>> f1_cache(sa, std::vector<std::vector<float>>(Nt / 2, std::vector<float>(sp, 0)));
  std::vector<std::vector<std::vector<float>>> ddf1_cache(sa, std::vector<std::vector<float>>(Nt / 2, std::vector<float>(sp, 0)));

#ifdef STIR_OPENMP
  if (getenv("OMP_NUM_THREADS") == NULL)
    {
      omp_set_num_threads(omp_get_num_procs());
      if (omp_get_num_procs() == 1)
        warning("Using OpenMP with #processors=1 produces parallel overhead. You should compile without using USE_OPENMP=TRUE.");
      cerr << "Using OpenMP-version of SRT2DSPECT with thread-count = processor-count (=" << omp_get_num_procs() << ")." << endl;
    }
  else
    {
      cerr << "Using OpenMP-version of SRT2DSPECT with " << getenv("OMP_NUM_THREADS") << " threads on " << omp_get_num_procs()
           << " processors." << endl;
      if (atoi(getenv("OMP_NUM_THREADS")) == 1)
        warning("Using OpenMP with OMP_NUM_THREADS=1 produces parallel overhead. Use more threads or compile without using "
                "USE_OPENMP=TRUE.");
    }
// cerr<<"Define number of threads by setting OMP_NUM_THREADS environment variable, i.e. \"export
// OMP_NUM_THREADS=<num_threads>\""<<endl; shared_ptr<DiscretisedDensity<3,float> > empty_density_ptr(density_ptr->clone());
#endif

  // c --------------------------
  // c Put theta and p in arrays.
  // c --------------------------
  for (i = 0; i < sth; i++)
    th[i] = i * 2 * M_PI / sth;
  for (int it = 0; it < Nt; it++)
    t[it] = it * 2 * M_PI / Nt;
  for (j = 0; j < sp; j++)
    p[j] = -1.0 + 2.0 * j / (sp - 1);

  // c ------------------------
  // c Put x1 and x2 in arrays.
  // c ------------------------
  for (k1 = 0; k1 < sx; k1++)
    x1[k1] = -1.0 + 2.0 * k1 / (sx - 1);
  for (k2 = 0; k2 < sx; k2++)
    x2[k2] = -1.0 + 2.0 * k2 / (sx - 1);

  for (int it = 0; it < Nt / 2; it++)
    {
      view_atten = atten_data_ptr->get_viewgram(Nmul * it, 0);
      if (do_arc_correction)
        {
          view_atten = arc_correction.do_arc_correction(view_atten);
        }
      for (int ia = 0; ia < sa; ia++)
        {
          for (int ip = 0; ip < sp; ip++)
            {
              f_cache[ia][it][ip]
                  = view_atten[view_atten.get_min_axial_pos_num() + ia][view_atten.get_min_tangential_pos_num() + ip]; //*.15;
            }
        }
      for (int ia = 0; ia < sa; ia++)
        {
          spline(p, f_cache[ia][it], sp, ddf_cache[ia][it]);
        }

      for (int ia = 0; ia < sa; ia++)
        {
          for (int ip = 0; ip < sp; ip++)
            {
              f1_cache[ia][it][sp - ip - 1] = f_cache[ia][it][ip];
            }
        }
      for (int ia = 0; ia < sa; ia++)
        {
          for (int ip = 0; ip < sp; ip++)
            {
              ddf1_cache[ia][it][sp - ip - 1] = ddf_cache[ia][it][ip];
            }
        }
    }

//-- Starting calculations per view
// 2D algorithm only
#ifdef STIR_OPENMP
#  pragma omp parallel shared(view, do_arc_correction, arc_correction, p, th, x1, x2, f_cache, ddf_cache, image) private(        \
      g,                                                                                                                         \
      f,                                                                                                                         \
      ddg,                                                                                                                       \
      ddf,                                                                                                                       \
      hilb,                                                                                                                      \
      fcpe,                                                                                                                      \
      fspe,                                                                                                                      \
      fc,                                                                                                                        \
      fs,                                                                                                                        \
      ddfc,                                                                                                                      \
      ddfs,                                                                                                                      \
      aux,                                                                                                                       \
      rho,                                                                                                                       \
      lg,                                                                                                                        \
      tau,                                                                                                                       \
      a,                                                                                                                         \
      b,                                                                                                                         \
      tau1,                                                                                                                      \
      tau2,                                                                                                                      \
      w,                                                                                                                         \
      rho1,                                                                                                                      \
      rho2,                                                                                                                      \
      lg1_cache,                                                                                                                 \
      lg2_cache,                                                                                                                 \
      f_node,                                                                                                                    \
      h,                                                                                                                         \
      fcme_fin,                                                                                                                  \
      fsme_fin,                                                                                                                  \
      fcpe_fin,                                                                                                                  \
      fspe_fin,                                                                                                                  \
      gx,                                                                                                                        \
      fc_fin,                                                                                                                    \
      fs_fin,                                                                                                                    \
      hc_fin,                                                                                                                    \
      hs_fin,                                                                                                                    \
      rx1x2th,                                                                                                                   \
      dh1,                                                                                                                       \
      dh2,                                                                                                                       \
      Ft1,                                                                                                                       \
      Ft2,                                                                                                                       \
      F,                                                                                                                         \
      I,                                                                                                                         \
      rx1,                                                                                                                       \
      rx2)
#  pragma omp for schedule(dynamic) nowait
#endif
  for (int ith = 0; ith < sth; ith++)
    {

      info(boost::format("View %d of %d") % ith % sth);

//-- Loading the viewgram
#ifdef STIR_OPENMP
#  pragma omp critical
#endif
      {
        view = proj_data_ptr->get_viewgram(ith, 0);
        view_atten = atten_data_ptr->get_viewgram(ith, 0);
        if (do_arc_correction)
          {
            view = arc_correction.do_arc_correction(view);
            view_atten = arc_correction.do_arc_correction(view_atten);
          }
        for (int ia = 0; ia < sa; ia++)
          {
            for (int ip = 0; ip < sp; ip++)
              {
                g[ia][ip] = view[view.get_min_axial_pos_num() + ia][view.get_min_tangential_pos_num() + ip];
                f[ia][ip]
                    = view_atten[view_atten.get_min_axial_pos_num() + ia][view_atten.get_min_tangential_pos_num() + ip] * .15;
              }
          }
      }
      //-- Calculation of second derivative by use of function spline
      for (int ia = 0; ia < sa; ia++)
        {
          spline(p, g[ia], sp, ddg[ia]);
          spline(p, f[ia], sp, ddf[ia]);
        }

      //---- calculate h(rho,theta) for all rho, theta
      for (int ia = 0; ia < sa; ia++)
        {
          for (int ip = 0; ip < sp; ip++)
            {
              hilb[ia][ip] = hilbert_node(p[ip], f[ia], ddf[ia], p, sp, f[ia][ip]);

              fcpe[ia][ip] = exp(0.5 * f[ia][ip]) * cos(hilb[ia][ip] / (2 * M_PI));
              fspe[ia][ip] = exp(0.5 * f[ia][ip]) * sin(hilb[ia][ip] / (2 * M_PI));

              fc[ia][ip] = fcpe[ia][ip] * g[ia][ip];
              fs[ia][ip] = fspe[ia][ip] * g[ia][ip];
            }
          //-- calculate ddfc, ddfs for all \rho, \theta
          spline(p, fc[ia], sp, ddfc[ia]);
          spline(p, fs[ia], sp, ddfs[ia]);
        }

      //---- calculate r(x1, x2, theta)
      for (int ix1 = 0; ix1 < sx; ix1++)
        {
          for (int ix2 = 0; ix2 < sy; ix2++)
            {
              aux = sqrt(1.0 - x2[ix2] * x2[ix2]);
              if (fabs(x2[ix2]) >= 1.0 || fabs(x1[ix1]) >= aux)
                continue;

              rho = x2[ix2] * cos(th[ith]) - x1[ix1] * sin(th[ith]);

              int i = floor((rho + 1) * (sp - 1) / 2);
              float p1 = p[i];
              float p2 = p[i + 1];
              float A = (p2 - rho) / (p2 - p1);
              float B = 1 - A;
              float C = 1.0 / 6 * (A * A * A - A) * (p2 - p1) * (p2 - p1);
              float D = 1.0 / 6 * (B * B * B - B) * (p2 - p1) * (p2 - p1);

              for (int ip = 0; ip < sp; ip++)
                {
                  double val = fabs(rho - p[ip]);
                  lg[ip] = val < 2e-6 ? 0. : std::log(val); // Using std::log to specify the namespace
                }

              // calculate I
              tau = x2[ix2] * sin(th[ith]) + x1[ix1] * cos(th[ith]);
              if (tau >= 0)
                {
                  a = tau;
                  b = sqrt(1 - rho * rho);
                }
              else
                {
                  a = -sqrt(1 - rho * rho);
                  b = tau;
                }

              tau1 = a + (b - a) * (1.0 / 2 - sqrt(3.0) / 6);
              tau2 = a + (b - a) * (1.0 / 2 + sqrt(3.0) / 6);
              w = 1.0 / 2 * (b - a);

              for (int it = 0; it < Nt / 2; it++)
                {
                  rho1 = tau1 * sin(th[ith] - t[it]) + rho * cos(th[ith] - t[it]);
                  rho2 = tau2 * sin(th[ith] - t[it]) + rho * cos(th[ith] - t[it]);

                  for (int ip = 0; ip < sp - 1; ip++)
                    {
                      lg1_cache[it][ip] = log(fabs((p[ip + 1] - rho1) / (p[ip] - rho1)));
                      if (fabs(p[ip + 1] - rho1) < 2e-6 || fabs(p[ip] - rho1) < 2e-6)
                        lg1_cache[it][ip] = 0.;
                      lg2_cache[it][ip] = log(fabs((p[ip + 1] - rho2) / (p[ip] - rho2)));
                      if (fabs(p[ip + 1] - rho2) < 2e-6 || fabs(p[ip] - rho2) < 2e-6)
                        lg2_cache[it][ip] = 0.;
                    }
                }

              for (int ia = 0; ia < sa; ia++)
                {
                  // if(ia!=31 && ia!=70 && ia!=71 &&ia!=81 && ia!=100) continue;

                  f_node = A * f[ia][i] + B * f[ia][i + 1] + C * ddf[ia][i] + D * ddf[ia][i + 1];

                  // calculate fcme, fsme, fc, fs, hc, hs

                  h = hilbert(rho, f[ia], ddf[ia], p, sp, lg);
                  fcme_fin = exp(-0.5 * f_node) * cos(h / (2 * M_PI));
                  fsme_fin = exp(-0.5 * f_node) * sin(h / (2 * M_PI));

                  fcpe_fin = exp(0.5 * f_node) * cos(h / (2 * M_PI));
                  fspe_fin = exp(0.5 * f_node) * sin(h / (2 * M_PI));

                  gx = splint(p, g[ia], ddg[ia], sp, rho);

                  fc_fin = fcpe_fin * gx;
                  fs_fin = fspe_fin * gx;

                  hc_fin = hilbert(rho, fc[ia], ddfc[ia], p, sp, lg);
                  hs_fin = hilbert(rho, fs[ia], ddfs[ia], p, sp, lg);

                  rx1x2th[ia][ix1][ix2]
                      = fcme_fin * (1.0 / M_PI * hc_fin + 2.0 * fs_fin) + fsme_fin * (1.0 / M_PI * hs_fin - 2.0 * fc_fin);

                  // calculate I
                  for (int it = 0; it < Nt / 2; it++)
                    {
                      rho1 = tau1 * sin(th[ith] - t[it]) + rho * cos(th[ith] - t[it]);
                      rho2 = tau2 * sin(th[ith] - t[it]) + rho * cos(th[ith] - t[it]);
                      hilbert_der_double(rho1,
                                         f_cache[ia][it],
                                         ddf_cache[ia][it],
                                         f1_cache[ia][it],
                                         ddf1_cache[ia][it],
                                         p,
                                         sp,
                                         &dh1[it],
                                         &dh1[it + Nt / 2],
                                         lg1_cache[it]);
                      hilbert_der_double(rho2,
                                         f_cache[ia][it],
                                         ddf_cache[ia][it],
                                         f1_cache[ia][it],
                                         ddf1_cache[ia][it],
                                         p,
                                         sp,
                                         &dh2[it],
                                         &dh2[it + Nt / 2],
                                         lg2_cache[it]);
                    }

                  Ft1 = -1.0 / (4.0 * M_PI * M_PI) * integ(2 * M_PI, Nt, dh1);
                  Ft2 = -1.0 / (4.0 * M_PI * M_PI) * integ(2 * M_PI, Nt, dh2);
                  F = w * Ft1 + w * Ft2;

                  // if(tau>=0) {
                  //	I = exp(F);
                  // } else {
                  I = exp(f_node - F);
                  //}

                  // calculate r
                  rx1x2th[ia][ix1][ix2] = I * rx1x2th[ia][ix1][ix2];
                }
            }
        }

      //---- calculate g(x1, x2)
      for (int ia = 0; ia < sa; ia++)
        {
          for (int ix1 = 0; ix1 < sx; ix1++)
            {
              for (int ix2 = 0; ix2 < sy; ix2++)
                {
                  aux = sqrt(1.0 - x2[ix2] * x2[ix2]);
                  if (fabs(x2[ix2]) >= 1.0 || fabs(x1[ix1]) >= aux)
                    {
                      continue;
                    }

                  if (x1[ix1] < 0)
                    {
                      rx1 = (-3.0 * rx1x2th[ia][ix1][ix2] + 4.0 * rx1x2th[ia][ix1 + 1][ix2] - rx1x2th[ia][ix1 + 2][ix2])
                            / (2.0 * (2.0 / (sx - 1)));
                    }
                  else
                    {
                      rx1 = (3.0 * rx1x2th[ia][ix1][ix2] - 4.0 * rx1x2th[ia][ix1 - 1][ix2] + rx1x2th[ia][ix1 - 2][ix2])
                            / (2.0 * (2.0 / (sx - 1)));
                    }

                  if (x2[ix2] < 0)
                    {
                      rx2 = (-3.0 * rx1x2th[ia][ix1][ix2] + 4.0 * rx1x2th[ia][ix1][ix2 + 1] - rx1x2th[ia][ix1][ix2 + 2])
                            / (2.0 * (2.0 / (sy - 1)));
                    }
                  else
                    {
                      rx2 = (3.0 * rx1x2th[ia][ix1][ix2] - 4.0 * rx1x2th[ia][ix1][ix2 - 1] + rx1x2th[ia][ix1][ix2 - 2])
                            / (2.0 * (2.0 / (sy - 1)));
                    }

#ifdef STIR_OPENMP
#  pragma omp critical
#endif
                  {
                    image[ia][image_min_x + sx - ix1 - 1][image_min_y + ix2]
                        += 1.0 / (4.0 * M_PI) * (rx1 * sin(th[ith]) - rx2 * cos(th[ith])) * (2.0 * M_PI / sth);
                  }
                }
            }
        }
    } // slice

  // apply Wiener filter
  if (filter_wiener != 0)
    wiener(image, sx, sy, image.get_z_size());
  // apply median filter
  if (filter_median != 0)
    median(image, sx, sy, image.get_z_size());
  // adjust gamma
  if (filter_gamma != 0)
    gamma(image, sx, sy, image.get_z_size());

  return Succeeded::yes;
}

void
SRT2DSPECTReconstruction::wiener(VoxelsOnCartesianGrid<float>& image, int sx, int sy, int sa)
{

  const int min_x = image.get_min_x();
  const int min_y = image.get_min_y();
  const int ws = 9;

  for (int ia = 0; ia < sa; ia++)
    {
      std::vector<std::vector<float>> localMean(sx, std::vector<float>(sy, 0));
      std::vector<std::vector<float>> localVar(sx, std::vector<float>(sy, 0));

      float noise = 0.;

      for (int i = 0 + 1; i < sx - 1; i++)
        {
          for (int j = 0 + 1; j < sy - 1; j++)
            {
              localMean[i][j] = 0;
              localVar[i][j] = 0;

              for (int k = -1; k <= 1; k++)
                for (int l = -1; l <= 1; l++)
                  localMean[i][j] += image[ia][min_x + i + k][min_y + j + l] * 1.;
              localMean[i][j] /= ws;

              for (int k = -1; k <= 1; k++)
                for (int l = -1; l <= 1; l++)
                  localVar[i][j] += image[ia][min_x + i + k][min_y + j + l] * image[ia][min_x + i + k][min_y + j + l];
              localVar[i][j] = localVar[i][j] / ws - localMean[i][j] * localMean[i][j];

              noise += localVar[i][j];
            }
        }
      noise /= sx * sy;

      for (int i = 0 + 1; i < sx - 1; i++)
        for (int j = 0 + 1; j < sy - 1; j++)
          image[ia][min_x + i][min_y + j] = (image[ia][min_x + i][min_y + j] - localMean[i][j]) / std::max(localVar[i][j], noise)
                                                * std::max(localVar[i][j] - noise, 0.f)
                                            + localMean[i][j];
    }
  return;
}

void
SRT2DSPECTReconstruction::median(VoxelsOnCartesianGrid<float>& image, int sx, int sy, int sa)
{
  const int min_x = image.get_min_x();
  const int min_y = image.get_min_y();
  const int filter_size = 3;
  const int offset = filter_size / 2;
  const int len = 4;
  // double neighbors[9];
  std::vector<double> neighbors(filter_size * filter_size, 0);

  for (int ia = 0; ia < sa; ia++)
    {
      for (int i = 0; i < 9; i++)
        neighbors[i] = 0;

      for (int i = 0; i < sx; i++)
        {
          for (int j = 0; j < sy; j++)
            {
              if (i == 0 || i == sx - 1 || j == 0 || j == sy - 1)
                continue;
              for (int k = -offset; k <= offset; k++)
                {
                  for (int l = -offset; l <= offset; l++)
                    {
                      neighbors[(k + offset) * filter_size + l + offset]
                          = image[ia][min_x + i + (k + i < sx ? k : 0)][min_y + j + (j + l < sy ? l : 0)];
                    }
                }
              std::sort(neighbors.begin(), neighbors.end());
              image[ia][min_x + i][min_y + j] = neighbors[len];
            }
        }
    }
  return;
}

void
SRT2DSPECTReconstruction::gamma(VoxelsOnCartesianGrid<float>& image, int sx, int sy, int sa)
{

  const int min_x = image.get_min_x();
  const int min_y = image.get_min_y();
  float targetAverage = .25; // Desired average pixel value

  for (int ia = 0; ia < sa; ia++)
    {

      // normalize image
      float min_val = INFINITY, max_val = -INFINITY;
      for (int i = 0; i < sx; i++)
        {
          for (int j = 0; j < sy; j++)
            {
              min_val = std::min(image[ia][min_x + i][min_y + j], min_val);
              max_val = std::max(image[ia][min_x + i][min_y + j], max_val);
            }
        }
      for (int i = 0; i < sx; i++)
        for (int j = 0; j < sy; j++)
          image[ia][min_x + i][min_y + j] = (image[ia][min_x + i][min_y + j] - min_val) / (max_val - min_val);

      // averagePixelValue = mean(img(abs(img)>.1));
      int count = 0;
      float averagePixelValue = 0.;
      for (int i = 0; i < sx; i++)
        {
          for (int j = 0; j < sy; j++)
            {
              if (std::abs(image[ia][min_x + i][min_y + j]) > 0.1)
                {
                  count++;
                  averagePixelValue += image[ia][min_x + i][min_y + j];
                }
            }
        }
      averagePixelValue /= count;

      float gamma_val = 1.;
      if (averagePixelValue > 0.)
        gamma_val = std::log(targetAverage) / std::log(averagePixelValue);
      // img = img.^gamma;
      for (int i = 0; i < sx; i++)
        for (int j = 0; j < sy; j++)
          image[ia][min_x + i][min_y + j] = std::abs(image[ia][min_x + i][min_y + j]) > 1e-6
                                                ? std::pow(image[ia][min_x + i][min_y + j], gamma_val)
                                                : image[ia][min_x + i][min_y + j];

      // denormalize image
      for (int i = 0; i < sx; i++)
        for (int j = 0; j < sy; j++)
          image[ia][min_x + i][min_y + j] = image[ia][min_x + i][min_y + j] * (max_val - min_val) + min_val;
    }
  return;
}

float
SRT2DSPECTReconstruction::hilbert_node(
    float x, const std::vector<float>& f, const std::vector<float>& ddf, const std::vector<float>& p, int sp, float fn)
{
  float dh;

  dh = 0;
  for (int i = 0; i < sp - 1; i++)
    {
      dh = dh - f[i] + f[i + 1]
           + 1.0 / 36
                 * (4 * p[i] * p[i] - 5 * p[i] * p[i + 1] - 5 * p[i + 1] * p[i + 1] - 3 * (p[i] - 5 * p[i + 1]) * x - 6 * x * x)
                 * ddf[i]
           + 1.0 / 36
                 * (5 * p[i] * p[i] + 5 * p[i] * p[i + 1] - 4 * p[i + 1] * p[i + 1] - 3 * (5 * p[i] - p[i + 1]) * x + 6 * x * x)
                 * ddf[i + 1];
    }

  if (fabs(x) == 1)
    {
      dh = 2 / (sp - 1) * dh;
    }
  else
    {
      dh = fn * log((1 - x) / (1 + x)) + 2 / (sp - 1) * dh;
    }

  return dh;
}

float
SRT2DSPECTReconstruction::hilbert(float x,
                                  const std::vector<float>& f,
                                  const std::vector<float>& ddf,
                                  const std::vector<float>& p,
                                  int sp,
                                  std::vector<float>& lg)
{
  float dh, Di;
  int i;

  i = 0;
  Di = -1.0 / (p[i] - p[i + 1])
       * ((p[i + 1] - x) * f[i] - (p[i] - x) * f[i + 1]
          - 1.0 / 6 * (p[i] - x) * (p[i + 1] - x)
                * ((p[i] - 2 * p[i + 1] + x) * ddf[i] + (2 * p[i] - p[i + 1] - x) * ddf[i + 1]));
  dh = -f[i] + f[i + 1]
       + 1.0 / 36 * (4 * p[i] * p[i] - 5 * p[i] * p[i + 1] - 5 * p[i + 1] * p[i + 1] - 3 * (p[i] - 5 * p[i + 1]) * x - 6 * x * x)
             * ddf[i]
       + 1.0 / 36 * (5 * p[i] * p[i] + 5 * p[i] * p[i + 1] - 4 * p[i + 1] * p[i + 1] - 3 * (5 * p[i] - p[i + 1]) * x + 6 * x * x)
             * ddf[i + 1]
       - Di * lg[i];

  for (i = 1; i < sp - 2; i++)
    {

      float Di1 = -1.0 / (p[i] - p[i + 1])
                  * ((p[i + 1] - x) * f[i] - (p[i] - x) * f[i + 1]
                     - 1.0 / 6 * (p[i] - x) * (p[i + 1] - x)
                           * ((p[i] - 2 * p[i + 1] + x) * ddf[i] + (2 * p[i] - p[i + 1] - x) * ddf[i + 1]));

      dh = dh - f[i] + f[i + 1]
           + 1.0 / 36
                 * (4 * p[i] * p[i] - 5 * p[i] * p[i + 1] - 5 * p[i + 1] * p[i + 1] - 3 * (p[i] - 5 * p[i + 1]) * x - 6 * x * x)
                 * ddf[i]
           + 1.0 / 36
                 * (5 * p[i] * p[i] + 5 * p[i] * p[i + 1] - 4 * p[i + 1] * p[i + 1] - 3 * (5 * p[i] - p[i + 1]) * x + 6 * x * x)
                 * ddf[i + 1]
           + (Di - Di1) * lg[i + 1];

      Di = Di1;
    }

  i = sp - 2;
  Di = -1.0 / (p[i] - p[i + 1])
       * ((p[i + 1] - x) * f[i] - (p[i] - x) * f[i + 1]
          - 1.0 / 6 * (p[i] - x) * (p[i + 1] - x)
                * ((p[i] - 2 * p[i + 1] + x) * ddf[i] + (2 * p[i] - p[i + 1] - x) * ddf[i + 1]));
  dh = dh - f[i] + f[i + 1]
       + 1.0 / 36 * (4 * p[i] * p[i] - 5 * p[i] * p[i + 1] - 5 * p[i + 1] * p[i + 1] - 3 * (p[i] - 5 * p[i + 1]) * x - 6 * x * x)
             * ddf[i]
       + 1.0 / 36 * (5 * p[i] * p[i] + 5 * p[i] * p[i + 1] - 4 * p[i + 1] * p[i + 1] - 3 * (5 * p[i] - p[i + 1]) * x + 6 * x * x)
             * ddf[i + 1]
       + Di * lg[sp - 1];

  dh = 2.0 / (sp - 1) * dh;

  return dh;
}

void
SRT2DSPECTReconstruction::hilbert_der_double(float x,
                                             const std::vector<float>& f,
                                             const std::vector<float>& ddf,
                                             const std::vector<float>& f1,
                                             const std::vector<float>& ddf1,
                                             const std::vector<float>& p,
                                             int sp,
                                             float* dhp,
                                             float* dh1p,
                                             const std::vector<float>& lg)
{
  float dh, dh1, dp;
  dh = 0;
  dh1 = 0;
  dp = p[1] - p[2]; // float pix = 0., pi1x = 0.;
  for (int i = 0; i < sp - 1; i++)
    {
      float pix = fabs(p[i] - x) > 2e-6 ? f[i] / (p[i] - x) : 0.;
      float pi1x = fabs(p[i + 1] - x) > 2e-6 ? f[i + 1] / (p[i + 1] - x) : 0.;
      dh = dh + pix - pi1x - 1.0 / 4 * (p[i] - 3 * p[i + 1] + 2 * x) * ddf[i]
           - 1.0 / 4 * (3 * p[i] - p[i + 1] - 2 * x) * ddf[i + 1]
           + ((f[i] - f[i + 1]) / dp - 1.0 / 6 * (p[i] - p[i + 1] - (3 * (p[i + 1] - x) * (p[i + 1] - x)) / dp) * ddf[i]
              + 1.0 / 6 * (p[i] - p[i + 1] - (3 * (p[i] - x) * (p[i] - x)) / dp) * ddf[i + 1])
                 * lg[i];
      pix = fabs(p[i] - x) > 2e-6 ? f1[i] / (p[i] - x) : 0.;
      pi1x = fabs(p[i + 1] - x) > 2e-6 ? f1[i + 1] / (p[i + 1] - x) : 0.;
      dh1 = dh1 + pix - pi1x - 1.0 / 4 * (p[i] - 3 * p[i + 1] + 2 * x) * ddf1[i]
            - 1.0 / 4 * (3 * p[i] - p[i + 1] - 2 * x) * ddf1[i + 1]
            + ((f1[i] - f1[i + 1]) / dp - 1.0 / 6 * (p[i] - p[i + 1] - (3 * (p[i + 1] - x) * (p[i + 1] - x)) / dp) * ddf1[i]
               + 1.0 / 6 * (p[i] - p[i + 1] - (3 * (p[i] - x) * (p[i] - x)) / dp) * ddf1[i + 1])
                  * lg[i];
    }
  dh = 2.0 / (sp - 1) * dh;
  dh1 = 2.0 / (sp - 1) * dh1;
  *dhp = dh;
  *dh1p = dh1;
}

float
SRT2DSPECTReconstruction::splint(
    const std::vector<float>& xa, const std::vector<float>& ya, const std::vector<float>& y2a, int n, float x)
{
  int klo, khi;
  float h, a, b, y;

  klo = 1;
  khi = n;
  while (khi - klo > 1)
    {
      int k = floor((khi + klo) / 2.0);
      if (xa[k] > x)
        {
          khi = k;
        }
      else
        {
          klo = k;
        }
    }

  h = xa[khi] - xa[klo];
  /* if(h == 0) {
          error('bad xa input in splint');
  } */
  a = (xa[khi] - x) / h;
  b = (x - xa[klo]) / h;
  y = a * ya[klo] + b * ya[khi] + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * (h * h) / 6.0;

  return y;
}

void
SRT2DSPECTReconstruction::spline(const std::vector<float>& x, const std::vector<float>& y, int n, std::vector<float>& y2)
{
  // function for nanural qubic spline.
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
SRT2DSPECTReconstruction::integ(float dist, int max, float ff[])
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
