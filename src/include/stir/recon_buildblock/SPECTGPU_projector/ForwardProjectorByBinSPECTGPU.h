//
//

#ifndef __stir_gpu_ForwardProjectorByBinSPECTGPU_h__
#define __stir_gpu_ForwardProjectorByBinSPECTGPU_h__
/*!
  \file
  \ingroup projection
  \ingroup SPECTGPU

  \brief Forward projection class using SPECTGPU's GPU implementation.

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

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/SPECTGPU_projector/SPECTGPUHelper.h"
#include "stir/cuda_utilities.h"

START_NAMESPACE_STIR

class ProjDataInMemory;
class DataSymmetriesForViewSegmentNumbers;

/*!
  \ingroup projection
  \brief Class for SPECTGPU's GPU forward projector.
*/
class ForwardProjectorByBinSPECTGPU : public RegisteredParsingObject<ForwardProjectorByBinSPECTGPU, ForwardProjectorByBin>
{
public:
  //! Name which will be used when parsing a ForwardProjectorByBin object
  static const char* const registered_name;

  //! Default constructor calls reset_timers()
  // inline
  ForwardProjectorByBinSPECTGPU();

  /// Constructor
  virtual ~ForwardProjectorByBinSPECTGPU();

  /// Keymap
  virtual void initialise_keymap() override;

//  virtual ForwardProjectorByBin* clone() const override
//  {
//      return new ForwardProjectorByBinSPECTGPU(*this);
//  }

  virtual const DataSymmetriesForViewSegmentNumbers*
  get_symmetries_used() const override
  {
      return _symmetries_sptr.get();
  }

  //! Stores all necessary geometric info
  /*!
   If necessary, set_up() can be called more than once.

   Derived classes can assume that forward_project()  will be called
   with input corresponding to the arguments of the last call to set_up().

   \warning there is currently no check on this.
   \warning Derived classes have to call set_up from the base class.
   */
  virtual void set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
                      const shared_ptr<const DiscretisedDensity<3, float>>& density_info_sptr // TODO should be Info only
  ) override;

  /// Set input
  virtual void set_input(const DiscretisedDensity<3, float>&) override;

  /// Set verbosity
  void set_verbosity(const bool verbosity) { _cuda_verbosity = verbosity; }

  /// Set use truncation - truncate before forward
  /// projection and after back projection
  void set_use_truncation(const bool use_truncation) { _use_truncation = use_truncation; }

protected:
  //! This virtual function has to be implemented by the derived class.
  virtual void actual_forward_project(RelatedViewgrams<float>&,
                                      const DiscretisedDensity<3, float>&,
                                      const int min_axial_pos_num,
                                      const int max_axial_pos_num,
                                      const int min_tangential_pos_num,
                                      const int max_tangential_pos_num) override;

  virtual void actual_forward_project(RelatedViewgrams<float>& viewgrams,
                                      const int min_ax,
                                      const int max_ax,
                                      const int min_tg,
                                      const int max_tg) override;

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
  shared_ptr<ProjDataInMemory> _projected_data_sptr;
  SPECTGPUHelper _helper;
  int _cuda_device;
  bool _cuda_verbosity;
  bool _use_truncation;
};

END_NAMESPACE_STIR

#endif // __stir_gpu_ForwardProjectorByBinSPECTGPU_h__
