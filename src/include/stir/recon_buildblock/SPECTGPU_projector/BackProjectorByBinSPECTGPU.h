//
//
/*!
  \file
  \ingroup projection
  \ingroup SPECTGPU

  \brief Back projection class using SPECTGPU's GPU implementation.

  \author Daniel Deidda

  \todo SPECTGPU limitations - 

  \todo STIR wrapper limitations - 
*/
/*
    Copyright (C) 2026, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_gpu_BackProjectorByBinSPECTGPU_h__
#define __stir_gpu_BackProjectorByBinSPECTGPU_h__

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUHelper.h"
#include "stir/cuda_utilities.h"

START_NAMESPACE_STIR

class DataSymmetriesForViewSegmentNumbers;

/*!
  \ingroup projection
  \brief Class for SPECTGPU's GPU back projector
*/
class BackProjectorByBinSPECTGPU : public RegisteredParsingObject<BackProjectorByBinSPECTGPU, BackProjectorByBin>
{
public:
  //! Name which will be used when parsing a BackProjectorByBin object
  static const char* const registered_name;

  //! Default constructor calls reset_timers()
  BackProjectorByBinSPECTGPU();

  virtual ~BackProjectorByBinSPECTGPU();

  /// Keymap
  virtual void initialise_keymap() override;

  //! Stores all necessary geometric info
  /*!
   If necessary, set_up() can be called more than once.
   */
  virtual void set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
                      const shared_ptr<const DiscretisedDensity<3, float>>& density_info_sptr // TODO should be Info only
  ) override;

  /// Back project
//  void back_project(const ProjData&, int subset_num = 0, int num_subsets = 1) override;

  /// Get output
//  virtual void get_output(DiscretisedDensity<3, float>&) const override;

  /*! \brief tell the back projector to start accumulating into a new target.
    This function has to be called before any back-projection is initiated.*/
//  virtual void start_accumulating_in_new_target() override;

  /// Set verbosity
  void set_verbosity(const bool verbosity) { _cuda_verbosity = verbosity; }

  /// Set use truncation - truncate before forward
  /// projection and after back projection
  void set_use_truncation(const bool use_truncation) { _use_truncation = use_truncation; }

  virtual BackProjectorByBin* clone() const override
  {
      return new BackProjectorByBinSPECTGPU(*this);
  }

  virtual const DataSymmetriesForViewSegmentNumbers*
  get_symmetries_used() const override
  {
      return _symmetries_sptr.get();
  }

protected:
  virtual void actual_back_project(const RelatedViewgrams<float>&,
                                   const int min_axial_pos_num,
                                   const int max_axial_pos_num,
                                   const int min_tangential_pos_num,
                                   const int max_tangential_pos_num) override;

  virtual void actual_back_project(DiscretisedDensity<3, float>& stir_image,
                                   const RelatedViewgrams<float>&,
                                   const int min_axial_pos_num,
                                   const int max_axial_pos_num,
                                   const int min_tangential_pos_num,
                                   const int max_tangential_pos_num) override;
protected:

  int dim_z, dim_y, dim_x;
  cuda_dim3 block_dim;
  cuda_dim3 grid_dim;
  float spacing_x;
  float spacing_y;
  float spacing_z;
  float origin_x;
  float origin_y;
  float origin_z;
  int min_z;
  int min_y;
  int min_x;
  int num_views;
private:
  shared_ptr<DataSymmetriesForViewSegmentNumbers> _symmetries_sptr;
  SPECTGPUHelper _helper;
  int _cuda_device;
  bool _cuda_verbosity;
  std::vector<float> _np_sino;
  bool _use_truncation;
};

END_NAMESPACE_STIR

#endif // __stir_gpu_BackProjectorByBinSPECTGPU_h__
