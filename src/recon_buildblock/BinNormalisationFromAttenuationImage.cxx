//
//
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup normalisation

  \brief Implementation for class stir::BinNormalisationFromAttenuationImage

  \author Kris Thielemans
*/

#include "stir/recon_buildblock/BinNormalisationFromAttenuationImage.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h" // used for rescaling attenuation image
#include "stir/RelatedViewgrams.h"
#include "stir/ArrayFunction.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/read_from_file.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/format.h"

START_NAMESPACE_STIR

const char* const BinNormalisationFromAttenuationImage::registered_name = "From Attenuation Image";

void
BinNormalisationFromAttenuationImage::set_defaults()
{
  base_type::set_defaults();
  attenuation_image_ptr.reset();
  forward_projector_ptr.reset();
  attenuation_image_filename = "";
}

void
BinNormalisationFromAttenuationImage::initialise_keymap()
{
  base_type::initialise_keymap();
  parser.add_start_key("Bin Normalisation From Attenuation Image");
  parser.add_key("attenuation_image_filename", &attenuation_image_filename);
  parser.add_parsing_key("forward projector type", &forward_projector_ptr);
  parser.add_stop_key("End Bin Normalisation From Attenuation Image");
}

bool
BinNormalisationFromAttenuationImage::post_processing()
{
  // read attenuation_image
  // we do this only when it isn't initialised yet, as this function can be called from a constructor
  if (is_null_ptr(attenuation_image_ptr))
    attenuation_image_ptr = read_from_file<DiscretisedDensity<3, float>>(attenuation_image_filename);
  if (is_null_ptr(attenuation_image_ptr))
    {
      warning("BinNormalisationFromAttenuationImage could not read attenuation image %s\n", attenuation_image_filename.c_str());
      return true;
    }
  if (is_null_ptr(forward_projector_ptr))
    forward_projector_ptr.reset(new ForwardProjectorByBinUsingRayTracing());

  {
    const float amax = attenuation_image_ptr->find_max();
    if ((amax < .08F) || (amax > .2F))
      warning(format("BinNormalisationFromAttenuationImage:\n"
                     "\tattenuation image data are supposed to be in units cm^-1\n"
                     "\tReference: water has mu .096 cm^-1\n"
                     "\tMax in attenuation image: {}\n"
                     "\tContinuing as you might know what you are doing.",
                     amax));
  }
#ifndef NEWSCALE
  /*
  cerr << "WARNING: multiplying attenuation image by x-voxel size "
  << " to correct for scale factor in forward projectors...\n";
*/
  // projectors work in pixel units, so convert attenuation data
  // from cm^-1 to pixel_units^-1
  const float rescale
      = dynamic_cast<DiscretisedDensityOnCartesianGrid<3, float> const&>(*attenuation_image_ptr).get_grid_spacing()[3] / 10;
#else
  const float rescale = 0.1F;
#endif
  shared_ptr<DiscretisedDensity<3, float>> new_sptr(attenuation_image_ptr->clone());
  *new_sptr *= rescale;
  attenuation_image_ptr = new_sptr;

  return base_type::post_processing();
}

BinNormalisationFromAttenuationImage::BinNormalisationFromAttenuationImage()
{
  set_defaults();
}

BinNormalisationFromAttenuationImage::BinNormalisationFromAttenuationImage(
    const std::string& filename, shared_ptr<ForwardProjectorByBin> const& forward_projector_ptr)
    : forward_projector_ptr(forward_projector_ptr),
      attenuation_image_filename(filename)
{
  attenuation_image_ptr.reset();
  post_processing();
}

BinNormalisationFromAttenuationImage::BinNormalisationFromAttenuationImage(
    shared_ptr<const DiscretisedDensity<3, float>> const& attenuation_image_ptr_v,
    shared_ptr<ForwardProjectorByBin> const& forward_projector_ptr)
    : attenuation_image_ptr(
        attenuation_image_ptr_v->clone()), // need a clone as it guarantees we won't be affected by the caller, and vice versa
      forward_projector_ptr(forward_projector_ptr)
{
  post_processing();
}

Succeeded
BinNormalisationFromAttenuationImage::set_up(const shared_ptr<const ExamInfo>& exam_info_sptr,
                                             const shared_ptr<const ProjDataInfo>& proj_data_info_ptr)
{
  if (proj_data_info_ptr->get_num_tof_poss() > 1)
    error("BinNormalisationFromAttenuationImage limitation: currently can only handle non_TOF data.\n"
          "You currently have to follow a 2 step procedure:\n"
          "   1) compute ACF factors without TOF\n"
          "   2) use this as input for BinNormalisationFromProjData");
  base_type::set_up(exam_info_sptr, proj_data_info_ptr);
  forward_projector_ptr->set_up(proj_data_info_ptr, attenuation_image_ptr);
  forward_projector_ptr->set_input(*attenuation_image_ptr);
  return Succeeded::yes;
}

void
BinNormalisationFromAttenuationImage::apply(RelatedViewgrams<float>& viewgrams) const
{
  this->check(*viewgrams.get_proj_data_info_sptr());
  RelatedViewgrams<float> attenuation_viewgrams = viewgrams.get_empty_copy();
  forward_projector_ptr->forward_project(attenuation_viewgrams);

  // TODO cannot use std::transform ?
  for (RelatedViewgrams<float>::iterator viewgrams_iter = attenuation_viewgrams.begin();
       viewgrams_iter != attenuation_viewgrams.end();
       ++viewgrams_iter)
    {
      in_place_exp(*viewgrams_iter);
    }
  viewgrams *= attenuation_viewgrams;
}

void
BinNormalisationFromAttenuationImage::undo(RelatedViewgrams<float>& viewgrams) const
{
  this->check(*viewgrams.get_proj_data_info_sptr());
  RelatedViewgrams<float> attenuation_viewgrams = viewgrams.get_empty_copy();
  forward_projector_ptr->forward_project(attenuation_viewgrams);

  // TODO cannot use std::transform ?
  for (RelatedViewgrams<float>::iterator viewgrams_iter = attenuation_viewgrams.begin();
       viewgrams_iter != attenuation_viewgrams.end();
       ++viewgrams_iter)
    {
      in_place_exp(*viewgrams_iter);
    }
  viewgrams /= attenuation_viewgrams;
}

float
BinNormalisationFromAttenuationImage::get_bin_efficiency(const Bin& bin) const
{
  // TODO
  error("BinNormalisationFromAttenuationImage::get_bin_efficiency is not implemented");
  return 1;
}

END_NAMESPACE_STIR
