/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup recon_buildblock

  \brief Standalone utilities for DFM-style 3D missing-data filling.

*/

#ifndef __stir_recon_buildblock_missing_data_MissingDataReprojection3D_H__
#define __stir_recon_buildblock_missing_data_MissingDataReprojection3D_H__

#include "stir/Array_complex_numbers.h"
#include "stir/common.h"
#include <iosfwd>
#include <vector>

START_NAMESPACE_STIR

class ProjData;
struct ArtificialScanner3DLayout;

/*!
  \brief Dense 4D storage for per-segment sinograms, including artificial missing-data bins.

  Layout indexes are \c (segment_index, view, axial, tangential).
*/
class MissingDataSinogram4D
{
public:
  MissingDataSinogram4D() = default;
  MissingDataSinogram4D(int num_segments, int num_views, int num_tangential_poss, const std::vector<int>& axial_counts);

  void
  resize(int num_segments, int num_views, int num_tangential_poss, const std::vector<int>& axial_counts);

  inline int get_num_segments() const { return static_cast<int>(this->axial_counts_.size()); }
  inline int get_num_views() const { return this->num_views_; }
  inline int get_num_tangential_poss() const { return this->num_tangential_poss_; }
  inline int get_num_axial_poss(const int segment_index) const { return this->axial_counts_[segment_index]; }

  float&
  at(int segment_index, int view, int axial, int tangential);
  const float&
  at(int segment_index, int view, int axial, int tangential) const;

private:
  std::size_t
  compute_offset(int segment_index, int view, int axial, int tangential) const;

  std::vector<int> axial_counts_;
  std::vector<std::size_t> segment_offsets_;
  int num_views_ = 0;
  int num_tangential_poss_ = 0;
  std::vector<float> data_;
};

/*!
  \brief Copy measured projection data into a missing-data container according to artificial-scanner layout.
*/
void
embed_measured_viewgrams_into_missing_data_sinogram(
    MissingDataSinogram4D& destination, const ProjData& proj_data, const ArtificialScanner3DLayout& layout);

/*!
  \brief Fill only missing bins with DFM-style trilinear reprojection from \c image.

  Measured bins are left untouched, defined by \c layout.measured_offsets and
  \c layout.measured_axial_counts.
*/
void
fill_missing_data_by_trilinear_reprojection(MissingDataSinogram4D& destination,
                                            const Array<3, std::complex<float>>& image,
                                            const std::vector<float>& pn,
                                            const std::vector<float>& an,
                                            const std::vector<float>& thn,
                                            const std::vector<float>& phin,
                                            const ArtificialScanner3DLayout& layout,
                                            std::ostream* log_stream = nullptr);

END_NAMESPACE_STIR

#endif
