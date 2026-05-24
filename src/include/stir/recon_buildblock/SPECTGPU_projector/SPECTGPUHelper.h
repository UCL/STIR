//
//

#ifndef __stir_gpu_SPECTGPUHelper_h__
#define __stir_gpu_SPECTGPUHelper_h__
/*!
  \file
  \ingroup projection
  \ingroup SPECTGPU

  \brief Helper class for SPECTGPU's GPU implementation.

  \author Daniel Deidda

  Helper class for SPECTGPU's GPU functionality. Wrapped
  functionality includes projection.

  \todo SPECTGPU limitations - .

  \todo STIR wrapper limitations - 
/*
    Copyright (C) 2026, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/shared_ptr.h"
#include "stir/Scanner.h"

// Forward declarations
struct Cnst;
struct txLUTs;
struct axialLUT;

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT>
class DiscretisedDensity;
class ProjData;
template <typename elemT>
class Viewgram;
template <typename elemT>
class VoxelsOnCartesianGrid;

/*!
  \ingroup projection
  \brief Helper class for the wrapped SPECTGPU projectors.
*/
class SPECTGPUHelper
{
public:
  /// Default constructor
  SPECTGPUHelper()
      : _already_set_up(false),
        _devid(0),
        _att(-1),
        _scanner_type(Scanner::Unknown_scanner)
  {}

  /// Destructor
  virtual ~SPECTGPUHelper();

  /// Set CUDA device ID
  void set_cuda_device_id(const int devid) { _devid = char(devid); }


  /// Set emission (0) or transmission (1) - whether to exp{-result} for attenuation maps
  void set_att(const char att) { _att = att; }

  /// Set verbosity level for CUDA output
  void set_verbose(const bool verbose) { _verbose = verbose; }

  /// Set scanner type
  void set_scanner_type(const Scanner::Type scanner_type) { _scanner_type = scanner_type; }

  /// Set up
  void set_up();

  /// Create SPECTGPU image
  static std::vector<float> create_SPECTGPU_image();

  /// Create STIR image with template dim
  static shared_ptr<VoxelsOnCartesianGrid<float>> create_stir_im();

  /// Create SPECTGPU singram. Forward project into this
  std::vector<float> create_SPECTGPU_sinogram() const;

  /// Convert STIR image to SPECTGPU image
  static void convert_image_stir_to_SPECTGPU(std::vector<float>& np, const DiscretisedDensity<3, float>& stir);

  /// Convert SPECTGPU image to STIR image
  static void convert_image_SPECTGPU_to_stir(DiscretisedDensity<3, float>& stir, const std::vector<float>& np_vec);

  /// Convert STIR proj data to SPECTGPU proj data
  void convert_proj_data_stir_to_SPECTGPU(std::vector<float>& np_vec, const ProjData& stir) const;

  /// Convert STIR viewgram to SPECTGPU
  void convert_viewgram_stir_to_SPECTGPU(std::vector<float>& np_vec, const Viewgram<float>& viewgram) const;

  /// Convert SPECTGPU proj data to STIR proj data
  void convert_proj_data_SPECTGPU_to_stir(ProjData& stir_sptr, const std::vector<float>& np_vec) const;

  /// Back project. Do some unavoidable const_casting as the wrapped methods don't use const
  void back_project(std::vector<float>& image, const std::vector<float>& sino_no_gaps) const;

  /// Forward project, returns sinogram without gaps. Do some unavoidable const_casting as the wrapped methods don't use const
  void forward_project(std::vector<float>& sino_no_gaps, const std::vector<float>& image) const;

  /// Create a STIR sinogram
  static shared_ptr<ProjData> create_stir_sino();

private:
  /// Check that set up has been run before returning data
  void check_set_up() const;

  /// Permute the data
  void permute(std::vector<float>& output_array,
               const std::vector<float>& orig_array,
               const unsigned output_dims[3],
               const unsigned* permute_order) const;

  /// Convert 3d SPECTGPU proj data index to 1d
  unsigned convert_SPECTGPU_proj_3d_to_1d_idx(const unsigned ang, const unsigned bins, const unsigned sino) const;

  /// Convert 1d SPECTGPU proj data index to 3d
  void convert_SPECTGPU_proj_1d_to_3d_idx(unsigned& ang, unsigned& bins, unsigned& sino, const unsigned idx) const;

  bool _already_set_up;
  char _devid;
  shared_ptr<Cnst> _cnt_sptr;
  int _nsinos;
  char _att;
  std::vector<int> _isub;
  bool _verbose;
  Scanner::Type _scanner_type;


  std::vector<float> _crs;
  std::vector<short> _s2c;

};

END_NAMESPACE_STIR

#endif // __stir_gpu_SPECTGPUHelper_h__
