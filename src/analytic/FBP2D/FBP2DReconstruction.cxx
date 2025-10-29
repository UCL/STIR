/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2012-01-09, Hammersmith Imanet Ltd
    Copyright (C) 2013, 2020 University College London

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup FBP2D
  \brief Implementation of class stir::FBP2DReconstruction

  \author Kris Thielemans
  \author Claire Labbe
  \author PARAPET project
*/

#include "stir/analytic/FBP2D/FBP2DReconstruction.h"
#include "stir/recon_buildblock/find_basic_vs_nums_in_subsets.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/RelatedViewgrams.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ArcCorrection.h"
#include "stir/analytic/FBP2D/RampFilter.h"
#include "stir/SSRB.h"
#include "stir/ProjDataInMemory.h"
// #include "stir/ProjDataInterfile.h"
#include "stir/Bin.h"
#include "stir/round.h"
#include "stir/display.h"
#include <algorithm>
#include "stir/IO/interfile.h"
#include "stir/info.h"
#include "stir/format.h"
#include "stir/warning.h"
#include "stir/error.h"

#ifdef STIR_OPENMP
#  include <omp.h>
#endif
#include "stir/num_threads.h"

START_NAMESPACE_STIR

const char* const FBP2DReconstruction::registered_name = "FBP2D";

void
FBP2DReconstruction::set_defaults()
{
  base_type::set_defaults();

  alpha_ramp = 1;
  fc_ramp = 0.5;
  pad_in_s = 1;
  display_level = 0; // no display
  num_segments_to_combine = -1;
  back_projector_sptr.reset(new BackProjectorByBinUsingInterpolation(
      /*use_piecewise_linear_interpolation = */ true,
      /*use_exact_Jacobian = */ false));
}

void
FBP2DReconstruction::initialise_keymap()
{
  base_type::initialise_keymap();

  parser.add_start_key("FBP2DParameters");
  parser.add_stop_key("End");
  parser.add_key("num_segments_to_combine with SSRB", &num_segments_to_combine);
  parser.add_key("Alpha parameter for Ramp filter", &alpha_ramp);
  parser.add_key("Cut-off for Ramp filter (in cycles)", &fc_ramp);
  parser.add_key("Transaxial extension for FFT", &pad_in_s);
  parser.add_key("Display level", &display_level);

  parser.add_parsing_key("Back projector type", &back_projector_sptr);
}

void
FBP2DReconstruction::ask_parameters()
{

  base_type::ask_parameters();

  num_segments_to_combine = ask_num("num_segments_to_combine (must be odd)", -1, 101, -1);
  alpha_ramp = ask_num(" Alpha parameter for Ramp filter ? ", 0., 1., 1.);
  fc_ramp = ask_num(" Cut-off frequency for Ramp filter ? ", 0., .5, 0.5);
  pad_in_s = ask_num("  Transaxial extension for FFT : ", 0, 1, 1);
  display_level = ask_num("Which images would you like to display \n\t(0: None, 1: Final, 2: filtered viewgrams) ? ", 0, 2, 0);

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
FBP2DReconstruction::post_processing()
{
  return base_type::post_processing();
}

Succeeded
FBP2DReconstruction::set_up(shared_ptr<FBP2DReconstruction::TargetT> const& target_data_sptr)
{
  if (base_type::set_up(target_data_sptr) == Succeeded::no)
    return Succeeded::no;

  if (fc_ramp <= 0 || fc_ramp > .5000000001)
    error(format("Cut-off frequency has to be between 0 and .5 but is {}", fc_ramp));

  if (alpha_ramp <= 0 || alpha_ramp > 1.000000001)
    error(format("Alpha parameter for ramp has to be between 0 and 1 but is {}", alpha_ramp));

  if (pad_in_s < 0 || pad_in_s > 2)
    error(format("padding factor has to be between 0 and 2 but is {}", pad_in_s));

  if (pad_in_s < 1)
    warning("Transaxial extension for FFT:=0 should ONLY be used when the non-zero data\n"
            "occupy only half of the FOV. Otherwise aliasing will occur!");

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

  if (is_null_ptr(back_projector_sptr))
    error("Back projector not set.");

  return Succeeded::yes;
}

std::string
FBP2DReconstruction::method_info() const
{
  return "FBP2D";
}

FBP2DReconstruction::FBP2DReconstruction(const std::string& parameter_filename)
{
  initialise(parameter_filename);
  info(format("{}", parameter_info()));
}

FBP2DReconstruction::FBP2DReconstruction()
{
  set_defaults();
}

FBP2DReconstruction::FBP2DReconstruction(const shared_ptr<ProjData>& proj_data_ptr_v,
                                         const double alpha_ramp_v,
                                         const double fc_ramp_v,
                                         const int pad_in_s_v,
                                         const int num_segments_to_combine_v)
{
  set_defaults();

  alpha_ramp = alpha_ramp_v;
  fc_ramp = fc_ramp_v;
  pad_in_s = pad_in_s_v;
  num_segments_to_combine = num_segments_to_combine_v;
  proj_data_ptr = proj_data_ptr_v;
}

Succeeded
FBP2DReconstruction::actual_reconstruct(shared_ptr<DiscretisedDensity<3, float>> const& density_ptr)
{

  // perform SSRB
  if (num_segments_to_combine > 1)
    {
      const ProjDataInfoCylindrical& proj_data_info_cyl
          = dynamic_cast<const ProjDataInfoCylindrical&>(*proj_data_ptr->get_proj_data_info_sptr());

      //  full_log << "SSRB combining " << num_segments_to_combine
      //           << " segments in input file to a new segment 0\n" << std::endl;

      shared_ptr<ProjDataInfo> ssrb_info_sptr(
          SSRB(proj_data_info_cyl, num_segments_to_combine, 1, 0, (num_segments_to_combine - 1) / 2));
      shared_ptr<ProjData> proj_data_to_FBP_ptr(new ProjDataInMemory(proj_data_ptr->get_exam_info_sptr(), ssrb_info_sptr));
      SSRB(*proj_data_to_FBP_ptr, *proj_data_ptr);
      proj_data_ptr = proj_data_to_FBP_ptr;
    }
  else
    {
      // just use the proj_data_ptr we have already
    }

  // check if segment 0 has direct sinograms
  {
    const float tan_theta = proj_data_ptr->get_proj_data_info_sptr()->get_tantheta(Bin(0, 0, 0, 0));
    if (fabs(tan_theta) > 1.E-4)
      {
        warning("FBP2D: segment 0 has non-zero tan(theta) %g", tan_theta);
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
      warning("FBP2D will arc-correct data first");
      arc_corrected_proj_data_info_sptr = arc_correction.get_arc_corrected_proj_data_info_sptr();
      tangential_sampling = arc_correction.get_arc_corrected_proj_data_info().get_tangential_sampling();
    }
  // ProjDataInterfile ramp_filtered_proj_data(arc_corrected_proj_data_info_sptr,"ramp_filtered");

  VoxelsOnCartesianGrid<float>& image = dynamic_cast<VoxelsOnCartesianGrid<float>&>(*density_ptr);

  // set projector to be used for the calculations
  back_projector_sptr->set_up(arc_corrected_proj_data_info_sptr, density_ptr);

  // set ramp filter with appropriate sizes
  const int fft_size = round(
      pow(2., ceil(log((double)(pad_in_s + 1) * arc_corrected_proj_data_info_sptr->get_num_tangential_poss()) / log(2.))));

  RampFilter filter(tangential_sampling, fft_size, float(alpha_ramp), float(fc_ramp));

  back_projector_sptr->start_accumulating_in_new_target();

  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr(back_projector_sptr->get_symmetries_used()->clone());

  const std::vector<ViewSegmentNumbers> vs_nums_to_process
      = detail::find_basic_vs_nums_in_subset(*proj_data_ptr->get_proj_data_info_sptr(),
                                             *symmetries_sptr,
                                             0,
                                             0, // only segment zero
                                             0,
                                             1); // project everything, therefore subset 0 of 1 subsets

#ifdef STIR_OPENMP
#  pragma omp parallel
#endif
  {
#ifdef STIR_OPENMP
#  pragma omp for schedule(dynamic)
#endif
    // note: older versions of openmp need an int as loop
    for (int i = 0; i < static_cast<int>(vs_nums_to_process.size()); ++i)
      {
        const ViewSegmentNumbers vs = vs_nums_to_process[i];
#ifdef STIR_OPENMP
        RelatedViewgrams<float> viewgrams;
#  pragma omp critical(FBP2D_get_viewgrams)
        viewgrams = proj_data_ptr->get_related_viewgrams(vs, symmetries_sptr);
#else
        RelatedViewgrams<float> viewgrams = proj_data_ptr->get_related_viewgrams(vs, symmetries_sptr);
#endif

        if (do_arc_correction)
          viewgrams = arc_correction.do_arc_correction(viewgrams);

        // now filter
        for (RelatedViewgrams<float>::iterator viewgram_iter = viewgrams.begin(); viewgram_iter != viewgrams.end();
             ++viewgram_iter)
          {
#ifdef NRFFT
            filter.apply(*viewgram_iter);
#else
            std::for_each(viewgram_iter->begin(), viewgram_iter->end(), filter);
#endif
          }

        info(format("Processing view {} of segment {}", vs.view_num(), vs.segment_num()), 2);
        back_projector_sptr->back_project(viewgrams);
      }
  }
  back_projector_sptr->get_output(*density_ptr);

  // Normalise the image
  const ProjDataInfoCylindrical& proj_data_info_cyl
      = dynamic_cast<const ProjDataInfoCylindrical&>(*proj_data_ptr->get_proj_data_info_sptr());

  float magic_number = 1.F;
  if (dynamic_cast<BackProjectorByBinUsingInterpolation const*>(back_projector_sptr.get()) != 0)
    {
      // KT & Darren Hogg 17/05/2000 finally found the scale factor!
      // TODO remove magic, is a scale factor in the backprojector
      magic_number
          = 2 * proj_data_info_cyl.get_ring_radius() * proj_data_info_cyl.get_num_views() / proj_data_info_cyl.get_ring_spacing();
    }
  else
    {
      if (proj_data_info_cyl.get_min_ring_difference(0) != proj_data_info_cyl.get_max_ring_difference(0))
        {
          magic_number = .5F;
        }
    }
#ifdef NEWSCALE
  // added binsize etc here to get units ok
  // only do this when the forward projector units are appropriate
  image *= magic_number / proj_data_ptr->get_num_views() * tangential_sampling
           / (image.get_voxel_size().x() * image.get_voxel_size().y());
#else
  image *= magic_number / proj_data_ptr->get_num_views();
#endif

  if (display_level > 0)
    display(image, image.find_max(), "FBP image");

  return Succeeded::yes;
}

END_NAMESPACE_STIR
