//
//
/*!
  \file
  \ingroup projection
  \ingroup SPECTGPU

  \brief non-inline implementations for stir::BackProjectorByBinSPECTGPU

  \author Daniel Deidda

*/
/*
    Copyright (C) 2026, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/SPECTGPU_projector/BackProjectorByBinSPECTGPU.h"
#include "stir/DiscretisedDensity.h"
#include "stir/RelatedViewgrams.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_array_functions.h"
#include "stir/error.h"
#include "stir/format.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUBackwardProjectorCUDA.h"
#include "stir/IO/read_from_file.h"
//#include "stir/cuda_utilities.h"

START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char* const BackProjectorByBinSPECTGPU::registered_name = "SPECTGPU";

BackProjectorByBinSPECTGPU::BackProjectorByBinSPECTGPU()
    : _cuda_device(0),
      _cuda_verbosity(true),
      _use_truncation(false),
      _slope(-1),
      _sigma0(-1),
      dev_image(nullptr),
      dev_umap(nullptr)
{
  this->_already_set_up = false;
}

BackProjectorByBinSPECTGPU::~BackProjectorByBinSPECTGPU()
{}

void
BackProjectorByBinSPECTGPU::initialise_keymap()
{
  parser.add_start_key("Back Projector Using SPECTGPU Parameters");
  parser.add_stop_key("End Back Projector Using SPECTGPU Parameters");
  parser.add_key("CUDA device", &_cuda_device);
  parser.add_key("collimator slope", &_slope);
  parser.add_key("collimator sigma 0(cm)", &_sigma0);
  parser.add_key("attenuation image filename", &_att_filename);
  parser.add_key("verbosity", &_cuda_verbosity);
}

void
BackProjectorByBinSPECTGPU::set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr,
                                   const shared_ptr<const DiscretisedDensity<3, float>>& density_info_sptr)
{
  BackProjectorByBin::set_up(proj_data_info_sptr, density_info_sptr);
  check(*proj_data_info_sptr, *_density_sptr);
  _symmetries_sptr.reset(new TrivialDataSymmetriesForBins(proj_data_info_sptr));

  if (_att_filename.empty())
    {
      warning("No attenuation is being used"); // no attenuation map
      _do_atten = false;
    }
  else
    {
      // read attenuation map
      _att_coeff_sptr = read_from_file<DiscretisedDensity<3, float>>(_att_filename);
      _do_atten = true;
      const auto o1 = _density_sptr->get_origin();
      const auto o2 = _att_coeff_sptr->get_origin();

      if (o1 != o2)
        {
          const auto& r1 = _density_sptr->get_index_range();
          const auto& r2 = _att_coeff_sptr->get_index_range();

          error(format("SPECTGPU:: Attenuation coefficient image expected to match characteristics of Activity image\n"
                       "Image origin:\n"
                       "  x={} y={} z={} \n"
                       "Attenuation origin:\n"
                       "  x={} y={} z={} \n"
                       "Image index range:\n"
                       "  x=({},{}) y=({},{}) z=({},{}) \n"
                       "Attenuation: index range\n"
                       "  x=({},{}) y=({},{}) z=({},{}) \n",
                       o1.x(),
                       o1.y(),
                       o1.z(),
                       o2.x(),
                       o2.y(),
                       o2.z(),
                       r1[1].get_min_index(),
                       r1[1].get_max_index(),
                       r1[2].get_min_index(),
                       r1[2].get_max_index(),
                       r1[3].get_min_index(),
                       r1[3].get_max_index(),
                       r2[1].get_min_index(),
                       r2[1].get_max_index(),
                       r2[2].get_min_index(),
                       r2[2].get_max_index(),
                       r2[3].get_min_index(),
                       r2[3].get_max_index()));
        }
    }

  auto& density_cast = dynamic_cast<const VoxelsOnCartesianGrid<float>&>(*_density_sptr);

  auto sizes = density_cast.get_lengths();

  this->spacing_x = std::abs(density_cast.get_grid_spacing().at(3));
  this->spacing_y = std::abs(density_cast.get_grid_spacing().at(2));
  this->spacing_z = std::abs(density_cast.get_grid_spacing().at(1));

  this->num_views = proj_data_info_sptr->get_num_views();

  this->origin_x = density_info_sptr->get_origin().x();
  this->origin_y = density_info_sptr->get_origin().y();
  this->origin_z = density_info_sptr->get_origin().z();

  this->dim_z = sizes[1];
  this->dim_y = sizes[2];
  this->dim_x = sizes[3];

  int dim_ax = proj_data_info_sptr->get_num_axial_poss(0);
  int dim_tg = proj_data_info_sptr->get_num_tangential_poss();
  float tg_spacing = proj_data_info_sptr->get_scanner_sptr()->get_default_bin_size();
  float ax_spacing = proj_data_info_sptr->get_scanner_sptr()->get_ring_spacing();
  if (dim_ax != this->dim_z || std::fabs(this->spacing_z - ax_spacing) > 1e-5f || dim_tg != this->dim_x
      || std::fabs(this->spacing_x - tg_spacing > 1e-5f))
    {
      error(format("SPECTGPU:: expected axial and tangential dimensions/spacings to match image z/x.\n"
                   "Image:\n"
                   "  dim_x={} dim_z={} spacing_x={} spacing_z={}\n"
                   "\n"
                   "Projection:\n"
                   "  dim_tg={} dim_ax={} spacing_tg={} spacing_ax={}\n"
                   "\n"
                   "Consider:\n"
                   "  zoom_image <output> <input> {} {} 0 0 {} {} 0\n",
                   this->dim_x,
                   this->dim_z,
                   this->spacing_x,
                   this->spacing_z,
                   dim_tg,
                   dim_ax,
                   tg_spacing,
                   ax_spacing,
                   dim_tg,
                   tg_spacing / this->spacing_x,
                   dim_ax,
                   ax_spacing / this->spacing_z));
    }
  // Set the thread block and grid dimensions using std::tuple
  this->min_z = density_info_sptr->get_min_index();
  this->min_y = density_cast[0][0].get_min_index();
  this->min_x = density_cast[0].get_min_index();

  //    std::cout<<"min_ind z = "<<min_z<<std::endl;
  //    std::cout<<"min_ind x = "<<min_x<<std::endl;
  this->grid_dim.x = (this->dim_x + this->block_dim.x - 1) / this->block_dim.x;
  this->grid_dim.y = (this->dim_y + this->block_dim.y - 1) / this->block_dim.y;
  this->grid_dim.z = (this->dim_z + this->block_dim.z - 1) / this->block_dim.z;
}

void
BackProjectorByBinSPECTGPU::get_output(DiscretisedDensity<3, float>& density) const
{
  copy_im_to_stir(density, dev_image);

  free_im_buffers(dev_image, dev_umap, _do_atten);
}

void
BackProjectorByBinSPECTGPU::start_accumulating_in_new_target()
{
  // Call base level
  BackProjectorByBin::start_accumulating_in_new_target();

  allocate_im_buffers(this->dev_image, this->dev_umap, *_density_sptr, *_att_coeff_sptr, _do_atten);

  if (_do_atten)
    copy_stir_im_to_dev(dev_umap, *_att_coeff_sptr);
}

void
BackProjectorByBinSPECTGPU::actual_back_project(DiscretisedDensity<3, float>& stir_image,
                                                const RelatedViewgrams<float>& stir_sino,
                                                const int,
                                                const int,
                                                const int,
                                                const int)
{
  run_backward_projection_cuda(dev_image,
                               stir_sino,
                               dev_umap,
                               _do_atten,
                               _sigma0,
                               _slope,
                               this->num_views,
                               this->block_dim.x,
                               this->block_dim.y,
                               this->block_dim.z,
                               this->grid_dim.x,
                               this->grid_dim.y,
                               this->grid_dim.z,
                               this->spacing_x,
                               this->spacing_y,
                               this->spacing_z,
                               this->origin_x,
                               this->origin_y,
                               this->origin_z,
                               this->dim_x,
                               this->dim_y,
                               this->dim_z,
                               this->min_z,
                               this->min_y,
                               this->min_x);
}

void
BackProjectorByBinSPECTGPU::actual_back_project(const RelatedViewgrams<float>& viewgrams,
                                                const int min_axial_pos_num,
                                                const int max_axial_pos_num,
                                                const int min_tangential_pos_num,
                                                const int max_tangential_pos_num)
{
  if (is_null_ptr(_density_sptr))
    error("BackProjectorByBinSPECTGPU: no input image set.");

  this->actual_back_project(
      *_density_sptr, viewgrams, min_axial_pos_num, max_axial_pos_num, min_tangential_pos_num, max_tangential_pos_num);
}
END_NAMESPACE_STIR
