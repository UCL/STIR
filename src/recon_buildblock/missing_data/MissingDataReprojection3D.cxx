/*
    Copyright (C) 2026 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup recon_buildblock

  \brief Implementation of standalone DFM-style 3D missing-data filling utilities.

*/

#include "stir/recon_buildblock/missing_data/MissingDataReprojection3D.h"
#include "stir/recon_buildblock/ArtificialScanner3D.h"
#include "stir/ProjData.h"
#include "stir/ProjDataInfo.h"
#include "stir/Viewgram.h"
#include "stir/error.h"
#include <cmath>
#include <complex>
#include <iostream>

START_NAMESPACE_STIR

MissingDataSinogram4D::MissingDataSinogram4D(
    int num_segments, int num_views, int num_tangential_poss, const std::vector<int>& axial_counts)
{
  this->resize(num_segments, num_views, num_tangential_poss, axial_counts);
}

void
MissingDataSinogram4D::resize(
    const int num_segments, const int num_views, const int num_tangential_poss, const std::vector<int>& axial_counts)
{
  if (num_segments <= 0)
    error("MissingDataSinogram4D: num_segments must be positive (%d)", num_segments);
  if (num_views <= 0)
    error("MissingDataSinogram4D: num_views must be positive (%d)", num_views);
  if (num_tangential_poss <= 0)
    error("MissingDataSinogram4D: num_tangential_poss must be positive (%d)", num_tangential_poss);
  if (static_cast<int>(axial_counts.size()) != num_segments)
    error("MissingDataSinogram4D: axial_counts size mismatch (%d vs %d)", static_cast<int>(axial_counts.size()), num_segments);

  this->axial_counts_ = axial_counts;
  this->num_views_ = num_views;
  this->num_tangential_poss_ = num_tangential_poss;

  this->segment_offsets_.assign(static_cast<std::size_t>(num_segments) + 1U, 0U);
  for (int iphi = 0; iphi < num_segments; ++iphi)
    {
      if (this->axial_counts_[iphi] <= 0)
        error("MissingDataSinogram4D: non-positive axial count %d at segment index %d", this->axial_counts_[iphi], iphi);

      const std::size_t segment_size = static_cast<std::size_t>(num_views) * static_cast<std::size_t>(this->axial_counts_[iphi])
                                       * static_cast<std::size_t>(num_tangential_poss);
      this->segment_offsets_[static_cast<std::size_t>(iphi) + 1U]
          = this->segment_offsets_[static_cast<std::size_t>(iphi)] + segment_size;
    }

  this->data_.assign(this->segment_offsets_.back(), 0.F);
}

std::size_t
MissingDataSinogram4D::compute_offset(const int segment_index, const int view, const int axial, const int tangential) const
{
  if (segment_index < 0 || segment_index >= this->get_num_segments())
    error("MissingDataSinogram4D: segment index out of range (%d)", segment_index);
  if (view < 0 || view >= this->num_views_)
    error("MissingDataSinogram4D: view index out of range (%d)", view);
  if (axial < 0 || axial >= this->axial_counts_[segment_index])
    error("MissingDataSinogram4D: axial index out of range (%d)", axial);
  if (tangential < 0 || tangential >= this->num_tangential_poss_)
    error("MissingDataSinogram4D: tangential index out of range (%d)", tangential);

  return this->segment_offsets_[static_cast<std::size_t>(segment_index)]
         + (static_cast<std::size_t>(view) * static_cast<std::size_t>(this->axial_counts_[segment_index])
            + static_cast<std::size_t>(axial))
               * static_cast<std::size_t>(this->num_tangential_poss_)
         + static_cast<std::size_t>(tangential);
}

float&
MissingDataSinogram4D::at(const int segment_index, const int view, const int axial, const int tangential)
{
  return this->data_[this->compute_offset(segment_index, view, axial, tangential)];
}

const float&
MissingDataSinogram4D::at(const int segment_index, const int view, const int axial, const int tangential) const
{
  return this->data_[this->compute_offset(segment_index, view, axial, tangential)];
}

void
embed_measured_viewgrams_into_missing_data_sinogram(
    MissingDataSinogram4D& destination, const ProjData& proj_data, const ArtificialScanner3DLayout& layout)
{
  const int sphi = static_cast<int>(layout.segment_numbers.size());
  if (destination.get_num_segments() != sphi)
    error("embed_measured_viewgrams_into_missing_data_sinogram: segment mismatch (%d vs %d)",
          destination.get_num_segments(),
          sphi);
  if (static_cast<int>(layout.measured_axial_counts.size()) != sphi || static_cast<int>(layout.measured_offsets.size()) != sphi)
    error("embed_measured_viewgrams_into_missing_data_sinogram: inconsistent layout vectors");

  const int min_tangential = proj_data.get_proj_data_info_sptr()->get_min_tangential_pos_num();

  for (int iphi = 0; iphi < sphi; ++iphi)
    {
      const int seg = layout.segment_numbers[iphi];
      const int measured_axial = layout.measured_axial_counts[iphi];
      const int axial_offset = layout.measured_offsets[iphi];

      for (int ith = 0; ith < destination.get_num_views(); ++ith)
        {
          const Viewgram<float> view = proj_data.get_viewgram(ith, seg);
          const int min_axial = view.get_min_axial_pos_num();

          for (int ia = 0; ia < measured_axial; ++ia)
            {
              for (int ip = 0; ip < destination.get_num_tangential_poss(); ++ip)
                {
                  destination.at(iphi, ith, ia + axial_offset, ip) = view[ia + min_axial][ip + min_tangential];
                }
            }
        }
    }
}

void
fill_missing_data_by_trilinear_reprojection(MissingDataSinogram4D& destination,
                                            const Array<3, std::complex<float>>& image,
                                            const std::vector<float>& pn,
                                            const std::vector<float>& an,
                                            const std::vector<float>& thn,
                                            const std::vector<float>& phin,
                                            const ArtificialScanner3DLayout& layout,
                                            std::ostream* log_stream)
{
  const int sphi = static_cast<int>(layout.segment_numbers.size());
  if (destination.get_num_segments() != sphi)
    error("fill_missing_data_by_trilinear_reprojection: segment mismatch (%d vs %d)", destination.get_num_segments(), sphi);
  if (static_cast<int>(layout.target_axial_counts.size()) != sphi || static_cast<int>(layout.measured_offsets.size()) != sphi
      || static_cast<int>(layout.measured_axial_counts.size()) != sphi)
    error("fill_missing_data_by_trilinear_reprojection: inconsistent layout vectors");

  const int sp = static_cast<int>(pn.size());
  const int sa = static_cast<int>(an.size());
  const int sth = static_cast<int>(thn.size());
  if (sp < 2 || sa < 2)
    error("fill_missing_data_by_trilinear_reprojection: pn/an sizes must be >=2 (sp=%d sa=%d)", sp, sa);
  if (destination.get_num_tangential_poss() != sp)
    error("fill_missing_data_by_trilinear_reprojection: tangential size mismatch (%d vs %d)",
          destination.get_num_tangential_poss(),
          sp);
  if (destination.get_num_views() != sth)
    error("fill_missing_data_by_trilinear_reprojection: view size mismatch (%d vs %d)", destination.get_num_views(), sth);
  if (static_cast<int>(phin.size()) != sphi)
    error("fill_missing_data_by_trilinear_reprojection: phin size mismatch (%d vs %d)", static_cast<int>(phin.size()), sphi);

  const float dp = pn[1] - pn[0];
  const float da = an[1] - an[0];
  if (dp == 0.F || da == 0.F)
    error("fill_missing_data_by_trilinear_reprojection: zero sampling step (dp=%g da=%g)", dp, da);

  if (log_stream)
    *log_stream << "Reprojecting...   [segment_num(num_axial_poss)]" << std::endl;

  for (int iphi = 0; iphi < sphi; ++iphi)
    {
      const int seg = layout.segment_numbers[iphi];
      const int sa1 = layout.target_axial_counts[iphi];
      const int known_begin = layout.measured_offsets[iphi];
      const int known_end = known_begin + layout.measured_axial_counts[iphi];

      if (destination.get_num_axial_poss(iphi) != sa1)
        error("fill_missing_data_by_trilinear_reprojection: axial size mismatch at seg index %d (%d vs %d)",
              iphi,
              destination.get_num_axial_poss(iphi),
              sa1);

      if (log_stream)
        *log_stream << seg << "(" << layout.measured_axial_counts[iphi] << "->" << sa1 << "), " << std::flush;

      std::vector<float> an1(sa1, 0.F);
      for (int ia = 0; ia < sa1; ++ia)
        an1[ia] = std::cos(phin[iphi]) * (-(sa1 - 1.0F) / 2.0F * da + ia * da);

      for (int ith = 0; ith < sth; ++ith)
        {
          for (int ia = 0; ia < sa1; ++ia)
            {
              if (ia >= known_begin && ia < known_end)
                continue;

              for (int ip = 0; ip < sp; ++ip)
                {
                  for (int is = 0; is < sp; ++is)
                    {
                      const float x = pn[is] * std::cos(thn[ith]) * std::cos(phin[iphi]) - pn[ip] * std::sin(thn[ith])
                                      - an1[ia] * std::cos(thn[ith]) * std::sin(phin[iphi]);
                      const float y = pn[is] * std::sin(thn[ith]) * std::cos(phin[iphi]) + pn[ip] * std::cos(thn[ith])
                                      - an1[ia] * std::sin(thn[ith]) * std::sin(phin[iphi]);
                      const float z = pn[is] * std::sin(phin[iphi]) + an1[ia] * std::cos(phin[iphi]);

                      if (x < -1.F || x > 1.F || y < -1.F || y > 1.F || z < an.front() || z > an.back())
                        continue;

                      int i = static_cast<int>(std::floor((sp - 1) * (x + 1.0F) / 2.0F));
                      if (x > 1.0F - 2.0e-6F)
                        i = sp - 2;

                      int j = static_cast<int>(std::floor((sp - 1) * (y + 1.0F) / 2.0F));
                      if (y > 1.0F - 2.0e-6F)
                        j = sp - 2;

                      int k = static_cast<int>(std::floor((sa - 1) * (z + an.back()) / (an.back() - an.front())));
                      if (z > an.back() - 2.0e-6F)
                        k = sa - 2;

                      const float xd = (x - pn[i]) / dp;
                      const float yd = (y - pn[j]) / dp;
                      const float zd = (z - an[k]) / da;

                      const float f00 = std::real(image[k][i][j]) * (1.0F - xd) + std::real(image[k][i + 1][j]) * xd;
                      const float f01 = std::real(image[k + 1][i][j]) * (1.0F - xd) + std::real(image[k + 1][i + 1][j]) * xd;
                      const float f10 = std::real(image[k][i][j + 1]) * (1.0F - xd) + std::real(image[k][i + 1][j + 1]) * xd;
                      const float f11
                          = std::real(image[k + 1][i][j + 1]) * (1.0F - xd) + std::real(image[k + 1][i + 1][j + 1]) * xd;

                      const float f0 = f00 * (1.0F - yd) + f10 * yd;
                      const float f1 = f01 * (1.0F - yd) + f11 * yd;
                      const float f = f0 * (1.0F - zd) + f1 * zd;

                      float& out_bin = destination.at(iphi, ith, ia, ip);
                      out_bin += f;
                      if (is == 0 || is == sp - 1)
                        out_bin -= 0.5F * f;
                    }
                }
            }
        }
    }
}

END_NAMESPACE_STIR
