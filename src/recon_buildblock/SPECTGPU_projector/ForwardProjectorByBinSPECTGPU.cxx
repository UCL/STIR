//
//
/*!

  \file
  \ingroup projection
  \ingroup SPECTGPU

  \brief non-inline implementations for stir::ForwardProjectorByBinSPECTGPU

  \author Daniel Deidda


*/
/*
    Copyright (C) 2026, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/SPECTGPU_projector/ForwardProjectorByBinSPECTGPU.h"
#include "stir/ProjDataInMemory.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/recon_array_functions.h"
#include "stir/error.h"
#include "stir/format.h"
#include "stir/IO/read_from_file.h"

#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUForwardProjectorCUDA.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUBackwardProjectorCUDA.h"

START_NAMESPACE_STIR

//////////////////////////////////////////////////////////
const char* const ForwardProjectorByBinSPECTGPU::registered_name = "SPECTGPU";

ForwardProjectorByBinSPECTGPU::ForwardProjectorByBinSPECTGPU()
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

void
ForwardProjectorByBinSPECTGPU::initialise_keymap()
{
  parser.add_start_key("Forward Projector Using SPECTGPU Parameters");
  parser.add_stop_key("End Forward Projector Using SPECTGPU Parameters");
  parser.add_key("CUDA device", &_cuda_device);
  parser.add_key("collimator slope", &_slope);
  parser.add_key("collimator sigma 0(cm)", &_sigma0);
  parser.add_key("attenuation image filename", &_att_filename);
  parser.add_key("verbosity", &_cuda_verbosity);
}

void
ForwardProjectorByBinSPECTGPU::set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr,
                                      const shared_ptr<const DiscretisedDensity<3, float>>& density_info_sptr)
{
  ForwardProjectorByBin::set_up(proj_data_info_sptr, density_info_sptr);
  check(*proj_data_info_sptr, *_density_sptr);
  _symmetries_sptr.reset(new TrivialDataSymmetriesForBins(proj_data_info_sptr));

  if (_att_filename.empty())
    {
      warning("SPECTGPU: No attenuation is being used"); // no attenuation map
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

  //  float spacing_x, spacing_y, spacing_z;
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
  this->block_dim.x = 8;
  this->block_dim.y = 8;
  this->block_dim.z = 8;

  this->min_z = density_info_sptr->get_min_index();
  this->min_y = density_cast[0][0].get_min_index();
  this->min_x = density_cast[0].get_min_index();

  //    std::cout<<"min_ind z = "<<min_z<<std::endl;
  //    std::cout<<"min_ind x = "<<min_x<<std::endl;
  this->grid_dim.x = (this->dim_x + this->block_dim.x - 1) / this->block_dim.x;
  this->grid_dim.y = (this->dim_y + this->block_dim.y - 1) / this->block_dim.y;
  this->grid_dim.z = (this->dim_z + this->block_dim.z - 1) / this->block_dim.z;

  // Initialise projected_data_sptr from this->_proj_data_info_sptr
  _projected_data_sptr.reset(new ProjDataInMemory(this->_density_sptr->get_exam_info_sptr(), proj_data_info_sptr));

  // Set up the SPECTGPU binary helper
  //    _helper.set_scanner_type(proj_data_info_sptr->get_scanner_ptr()->get_type());
  //    _helper.set_cuda_device_id(_cuda_device);
  //    _helper.set_verbose(_cuda_verbosity);
  //    _helper.set_up();
}

void
ForwardProjectorByBinSPECTGPU::actual_forward_project(RelatedViewgrams<float>& stir_sino,
                                                      const DiscretisedDensity<3, float>& stir_image,
                                                      const int min_ax,
                                                      const int max_ax,
                                                      const int min_tg,
                                                      const int max_tg)
{
  run_forward_projection_cuda(stir_sino,
                              this->dev_image,
                              this->dev_umap,
                              _do_atten,
                              _sigma0,
                              _slope,
                              this->num_views,
                              min_ax,
                              max_ax,
                              min_tg,
                              max_tg,
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
ForwardProjectorByBinSPECTGPU::actual_forward_project(
    RelatedViewgrams<float>& viewgrams, const int min_ax, const int max_ax, const int min_tg, const int max_tg)
{
  if (is_null_ptr(_density_sptr))
    error("ForwardProjectorByBinSPECTGPU: no input image set.");

  this->actual_forward_project(viewgrams, *_density_sptr, min_ax, max_ax, min_tg, max_tg);
}

void
ForwardProjectorByBinSPECTGPU::set_input(const DiscretisedDensity<3, float>& density)
{
  ForwardProjectorByBin::set_input(density);

  if (dev_image)
    {
      free_im_buffers(this->dev_image, this->dev_umap, this->_do_atten);
      dev_image = nullptr;
      dev_umap = nullptr;
    }

  allocate_im_buffers(this->dev_image, this->dev_umap, *_density_sptr, *_att_coeff_sptr, _do_atten);

  if (_do_atten)
    copy_stir_im_to_dev(dev_umap, *_att_coeff_sptr);
  copy_stir_im_to_dev(dev_image, density);

  // Before forward projection, we enforce a truncation outside of the FOV.
  // This is because the SPECTGPU FOV is smaller than the STIR FOV and this
  // could cause some voxel values to spiral out of control.
  //  if (_use_truncation)
  //    truncate_rim(*_density_sptr, 17);

  // --------------------------------------------------------------- //
  //   STIR -> SPECTGPU image data conversion
  // --------------------------------------------------------------- //

  //  std::vector<float> np_vec = _helper.create_SPECTGPU_image();
  //  _helper.convert_image_stir_to_SPECTGPU(np_vec, *_density_sptr);

  // --------------------------------------------------------------- //
  //   Forward projection
  // --------------------------------------------------------------- //
}

ForwardProjectorByBinSPECTGPU::~ForwardProjectorByBinSPECTGPU()
{
  if (dev_image)
    free_im_buffers(this->dev_image, this->dev_umap, this->_do_atten);
}
END_NAMESPACE_STIR
