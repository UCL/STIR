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
  virtual void initialise_keymap();

  //! Stores all necessary geometric info
  /*!
   If necessary, set_up() can be called more than once.
   */
  virtual void set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
                      const shared_ptr<const DiscretisedDensity<3, float>>& density_info_sptr // TODO should be Info only
  );

  /// Back project
  void back_project(const ProjData&, int subset_num = 0, int num_subsets = 1);

  /// Get output
  virtual void get_output(DiscretisedDensity<3, float>&) const;

  /*! \brief tell the back projector to start accumulating into a new target.
    This function has to be called before any back-projection is initiated.*/
  virtual void start_accumulating_in_new_target();

  /// Set verbosity
  void set_verbosity(const bool verbosity) { _cuda_verbosity = verbosity; }

  /// Set use truncation - truncate before forward
  /// projection and after back projection
  void set_use_truncation(const bool use_truncation) { _use_truncation = use_truncation; }

protected:
  virtual void actual_back_project(const RelatedViewgrams<float>&,
                                   const int min_axial_pos_num,
                                   const int max_axial_pos_num,
                                   const int min_tangential_pos_num,
                                   const int max_tangential_pos_num);

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
