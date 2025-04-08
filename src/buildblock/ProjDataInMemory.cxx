/*!

  \file
  \ingroup projdata
  \brief Implementations for non-inline functions of class stir::ProjDataInMemory

  \author Nikos Efthimiou
  \author Kris Thielemans
  \author Nikos Efthimiou
  \author Gemma Fardell
*/
/*
 *  Copyright (C) 2016, UCL
    Copyright (C) 2002 - 2011-02-23, Hammersmith Imanet Ltd
    Copyright (C) 2011, Kris Thielemans
    Copyright (C) 2016, University of Hull
    Copyright (C) 2016, 2019, 2020, 2023, 2024, UCL
    Copyright (C) 2020,  Rutherford Appleton Laboratory STFC
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInMemory.h"
#include "stir/copy_fill.h"
#include "stir/shared_ptr.h"
#include "stir/Succeeded.h"
#include "stir/SegmentByView.h"
#include "stir/Bin.h"
#include "stir/is_null_ptr.h"
#include "stir/numerics/norm.h"
#include <iostream>
#include <cstring>
#include <algorithm>

using std::string;
using std::streamoff;

START_NAMESPACE_STIR

ProjDataInMemory::~ProjDataInMemory()
{}

ProjDataInMemory::ProjDataInMemory(shared_ptr<const ExamInfo> const& exam_info_sptr,
                                   shared_ptr<const ProjDataInfo> const& proj_data_info_ptr,
                                   const bool initialise_with_0)
    : ProjData(exam_info_sptr, proj_data_info_ptr),
      segment_sequence(ProjData::standard_segment_sequence(*proj_data_info_ptr))
{
  this->create_buffer(initialise_with_0);

  int sum = 0;
  for (int segment_num = proj_data_info_sptr->get_min_segment_num(); segment_num <= proj_data_info_sptr->get_max_segment_num();
       ++segment_num)
    {
      sum += get_num_axial_poss(segment_num) * get_num_views() * get_num_tangential_poss();
    }

  offset_3d_data = static_cast<streamoff>(sum);

  timing_poss_sequence.resize(proj_data_info_sptr->get_num_tof_poss());

  for (int i = 0, timing_pos_num = proj_data_info_sptr->get_min_tof_pos_num();
       timing_pos_num <= proj_data_info_sptr->get_max_tof_pos_num();
       ++i, ++timing_pos_num)
    {
      timing_poss_sequence[i] = timing_pos_num;
    }
}

void
ProjDataInMemory::create_buffer(const bool initialise_with_0)
{
  this->buffer.resize(0, this->size_all() - 1, initialise_with_0);
}

///////////////// /set functions

namespace detail
{
template <int num_dimensions>
void
copy_data_from_buffer(const Array<1, float>& buffer, Array<num_dimensions, float>& array, std::streamoff offset)
{
#ifdef STIR_OPENMP
#  pragma omp critical(PROJDATAINMEMORYCOPY)
#endif
  {
    const float* ptr = buffer.get_const_data_ptr() + offset;
    fill_from(array, ptr, ptr + array.size_all());
    buffer.release_const_data_ptr();
  }
}

template <int num_dimensions>
void
copy_data_to_buffer(Array<1, float>& buffer, const Array<num_dimensions, float>& array, std::streamoff offset)
{
#ifdef STIR_OPENMP
#  pragma omp critical(PROJDATAINMEMORYCOPY)
#endif
  {
    float* ptr = buffer.get_data_ptr() + offset;
    copy_to(array, ptr);
    buffer.release_data_ptr();
  }
}
} // namespace detail

Viewgram<float>
ProjDataInMemory::get_viewgram(const int view_num,
                               const int segment_num,
                               const bool make_num_tangential_poss_odd,
                               const int timing_pos) const
{
  Bin bin(segment_num, view_num, this->get_min_axial_pos_num(segment_num), this->get_min_tangential_pos_num(), timing_pos);
  Viewgram<float> viewgram(proj_data_info_sptr, bin);

  for (bin.axial_pos_num() = get_min_axial_pos_num(segment_num); bin.axial_pos_num() <= get_max_axial_pos_num(segment_num);
       bin.axial_pos_num()++)
    {
      detail::copy_data_from_buffer(this->buffer, viewgram[bin.axial_pos_num()], this->get_index(bin));
    }

  if (make_num_tangential_poss_odd && (get_num_tangential_poss() % 2 == 0))

    {
      const int new_max_tangential_pos = get_max_tangential_pos_num() + 1;

      viewgram.grow(IndexRange2D(get_min_axial_pos_num(segment_num),
                                 get_max_axial_pos_num(segment_num),
                                 get_min_tangential_pos_num(),
                                 new_max_tangential_pos));
    }
  return viewgram;
}

Succeeded
ProjDataInMemory::set_viewgram(const Viewgram<float>& v)
{
  if (*get_proj_data_info_sptr() != *(v.get_proj_data_info_sptr()))
    {
      warning("ProjDataInMemory::set_viewgram: viewgram has incompatible ProjDataInfo member\n"
              "Original ProjDataInfo: %s\n"
              "ProjDataInfo From viewgram: %s",
              this->get_proj_data_info_sptr()->parameter_info().c_str(),
              v.get_proj_data_info_sptr()->parameter_info().c_str());

      return Succeeded::no;
    }
  const int segment_num = v.get_segment_num();
  const int view_num = v.get_view_num();
  const int timing_pos = v.get_timing_pos_num();

  Bin bin(segment_num, view_num, this->get_min_axial_pos_num(segment_num), this->get_min_tangential_pos_num(), timing_pos);

  for (bin.axial_pos_num() = get_min_axial_pos_num(segment_num); bin.axial_pos_num() <= get_max_axial_pos_num(segment_num);
       bin.axial_pos_num()++)
    {
      detail::copy_data_to_buffer(this->buffer, v[bin.axial_pos_num()], this->get_index(bin));
    }

  return Succeeded::yes;
}

std::streamoff
ProjDataInMemory::get_index(const Bin& this_bin) const
{
  if (!(this_bin.segment_num() >= get_min_segment_num() && this_bin.segment_num() <= get_max_segment_num()))
    error("ProjDataInMemory::this->get_index: segment_num out of range : %d", this_bin.segment_num());

  if (!(this_bin.axial_pos_num() >= get_min_axial_pos_num(this_bin.segment_num())
        && this_bin.axial_pos_num() <= get_max_axial_pos_num(this_bin.segment_num())))
    error("ProjDataInMemory::this->get_index: axial_pos_num out of range : %d", this_bin.axial_pos_num());
  if (!(this_bin.timing_pos_num() >= get_min_tof_pos_num() && this_bin.timing_pos_num() <= get_max_tof_pos_num()))
    error("ProjDataInMemory::this->get_index: timing_pos_num out of range : %d", this_bin.timing_pos_num());

  const int index = static_cast<int>(std::find(segment_sequence.begin(), segment_sequence.end(), this_bin.segment_num())
                                     - segment_sequence.begin());

  streamoff num_axial_pos_offset = 0;

  for (int i = 0; i < index; i++)
    num_axial_pos_offset += get_num_axial_poss(segment_sequence[i]);

  streamoff segment_offset = static_cast<streamoff>(num_axial_pos_offset * get_num_tangential_poss() * get_num_views());

  // Now we are just in front of  the correct segment
  {
    if (proj_data_info_sptr->get_num_tof_poss() > 1)
      {
        const int timing_index
            = static_cast<int>(std::find(timing_poss_sequence.begin(), timing_poss_sequence.end(), this_bin.timing_pos_num())
                               - timing_poss_sequence.begin());

        assert(offset_3d_data > 0);
        segment_offset += static_cast<streamoff>(timing_index) * offset_3d_data;
      }
    // skip axial positions
    const streamoff ax_pos_offset = (this_bin.axial_pos_num() - get_min_axial_pos_num(this_bin.segment_num())) * get_num_views()
                                    * get_num_tangential_poss();

    // sinogram location

    // find view
    const streamoff view_offset = (this_bin.view_num() - get_min_view_num()) * get_num_tangential_poss();

    // find tang pos
    const streamoff tang_offset = (this_bin.tangential_pos_num() - get_min_tangential_pos_num());

    return segment_offset + ax_pos_offset + view_offset + tang_offset;
  }
}

Sinogram<float>
ProjDataInMemory::get_sinogram(const int ax_pos_num,
                               const int segment_num,
                               const bool make_num_tangential_poss_odd,
                               const int timing_pos) const
{
  Sinogram<float> sinogram(proj_data_info_sptr, ax_pos_num, segment_num, timing_pos);
  Bin bin(segment_num, this->get_min_view_num(), ax_pos_num, this->get_min_tangential_pos_num(), timing_pos);

  detail::copy_data_from_buffer(this->buffer, sinogram, this->get_index(bin));

  if (make_num_tangential_poss_odd && (get_num_tangential_poss() % 2 == 0))
    {
      int new_max_tangential_pos = get_max_tangential_pos_num() + 1;

      sinogram.grow(IndexRange2D(get_min_view_num(), get_max_view_num(), get_min_tangential_pos_num(), new_max_tangential_pos));
    }

  return sinogram;
}

Succeeded
ProjDataInMemory::set_sinogram(const Sinogram<float>& s)
{
  if (*get_proj_data_info_sptr() != *(s.get_proj_data_info_sptr()))
    {
      warning("ProjDataInMemory::set_sinogram: Sinogram<float> has incompatible ProjDataInfo member.\n"
              "Original ProjDataInfo: %s\n"
              "ProjDataInfo from sinogram: %s",
              this->get_proj_data_info_sptr()->parameter_info().c_str(),
              s.get_proj_data_info_sptr()->parameter_info().c_str());

      return Succeeded::no;
    }
  int segment_num = s.get_segment_num();
  int ax_pos_num = s.get_axial_pos_num();
  int timing_pos = s.get_timing_pos_num();
  Bin bin(segment_num, this->get_min_view_num(), ax_pos_num, this->get_min_tangential_pos_num(), timing_pos);

  detail::copy_data_to_buffer(this->buffer, s, this->get_index(bin));
  return Succeeded::yes;
}

SegmentBySinogram<float>
ProjDataInMemory::get_segment_by_sinogram(const int segment_num, const int timing_pos_num) const
{
  const Bin bin(segment_num,
                this->get_min_view_num(),
                this->get_min_axial_pos_num(segment_num),
                this->get_min_tangential_pos_num(),
                timing_pos_num);
  SegmentBySinogram<float> segment(proj_data_info_sptr, bin);
  detail::copy_data_from_buffer(this->buffer, segment, this->get_index(bin));
  return segment;
}

SegmentByView<float>
ProjDataInMemory::get_segment_by_view(const int segment_num, const int timing_pos) const
{
  // TODO rewrite in terms of get_sinogram as this doubles memory temporarily
  return SegmentByView<float>(get_segment_by_sinogram(segment_num, timing_pos));
}

Succeeded
ProjDataInMemory::set_segment(const SegmentBySinogram<float>& segmentbysinogram_v)
{
  if (get_num_tangential_poss() != segmentbysinogram_v.get_num_tangential_poss())
    {
      warning("ProjDataInMemory::set_segmen: num_bins is not correct\n");
      return Succeeded::no;
    }
  if (get_num_views() != segmentbysinogram_v.get_num_views())
    {
      warning("ProjDataInMemory::set_segment: num_views is not correct\n");
      return Succeeded::no;
    }

  const int segment_num = segmentbysinogram_v.get_segment_num();
  const Bin bin(segment_num,
                this->get_min_view_num(),
                this->get_min_axial_pos_num(segment_num),
                this->get_min_tangential_pos_num(),
                segmentbysinogram_v.get_timing_pos_num());

  detail::copy_data_to_buffer(this->buffer, segmentbysinogram_v, this->get_index(bin));
  return Succeeded::yes;
}

Succeeded
ProjDataInMemory::set_segment(const SegmentByView<float>& segmentbyview_v)
{
  // TODO rewrite in terms of set_sinogram
  const SegmentBySinogram<float> segmentbysinogram(segmentbyview_v);
  return set_segment(segmentbysinogram);
}

/////////////////  other functions
void
ProjDataInMemory::fill(const float value)
{
  std::fill(begin_all(), end_all(), value);
}

void
ProjDataInMemory::fill(const ProjData& proj_data)
{
  auto pdm_ptr = dynamic_cast<ProjDataInMemory const*>(&proj_data);
  if (!is_null_ptr(pdm_ptr) && (*this->get_proj_data_info_sptr()) == (*proj_data.get_proj_data_info_sptr()))
    {
      std::copy(pdm_ptr->begin_all(), pdm_ptr->end_all(), begin_all());
    }
  else
    {
      return ProjData::fill(proj_data);
    }
}

ProjDataInMemory::ProjDataInMemory(const ProjData& proj_data)
    : ProjDataInMemory(proj_data.get_exam_info_sptr(), proj_data.get_proj_data_info_sptr()->create_shared_clone(), false)
{
  this->fill(proj_data);
}

ProjDataInMemory::ProjDataInMemory(const ProjDataInMemory& proj_data)
    : ProjDataInMemory(proj_data.get_exam_info_sptr(), proj_data.get_proj_data_info_sptr()->create_shared_clone(), false)
{
  std::copy(proj_data.begin_all(), proj_data.end_all(), this->begin_all());
}

shared_ptr<ProjDataInMemory>
ProjDataInMemory::read_from_file(const std::string& filename)
{
  return std::make_shared<ProjDataInMemory>(*ProjData::read_from_file(filename));
}

float
ProjDataInMemory::get_bin_value(Bin& bin)
{
  return buffer[this->get_index(bin)];
}

void
ProjDataInMemory::set_bin_value(const Bin& bin)
{
  buffer[this->get_index(bin)] = bin.get_bin_value();
}

float
ProjDataInMemory::sum() const
{
  return buffer.sum();
}

float
ProjDataInMemory::find_max() const
{
  return buffer.find_max();
}

float
ProjDataInMemory::find_min() const
{
  return buffer.find_min();
}

double
ProjDataInMemory::norm() const
{
  return stir::norm(this->buffer);
}

double
ProjDataInMemory::norm_squared() const
{
  return stir::norm_squared(this->buffer);
}

ProjDataInMemory&
ProjDataInMemory::operator+=(const base_type& v)
{
  if (auto vp = dynamic_cast<const ProjDataInMemory*>(&v))
    this->buffer += vp->buffer;
  else
    base_type::operator+=(v);

  return *this;
}

ProjDataInMemory&
ProjDataInMemory::operator-=(const base_type& v)
{
  if (auto vp = dynamic_cast<const ProjDataInMemory*>(&v))
    this->buffer -= vp->buffer;
  else
    base_type::operator-=(v);
  return *this;
}

ProjDataInMemory&
ProjDataInMemory::operator*=(const base_type& v)
{
  if (auto vp = dynamic_cast<const ProjDataInMemory*>(&v))
    this->buffer *= vp->buffer;
  else
    base_type::operator*=(v);
  return *this;
}

ProjDataInMemory&
ProjDataInMemory::operator/=(const base_type& v)
{
  if (auto vp = dynamic_cast<const ProjDataInMemory*>(&v))
    this->buffer /= vp->buffer;
  else
    base_type::operator/=(v);

  return *this;
}

ProjDataInMemory&
ProjDataInMemory::operator+=(const float v)
{
  this->buffer += v;
  return *this;
}

ProjDataInMemory&
ProjDataInMemory::operator-=(const float v)
{
  this->buffer -= v;
  return *this;
}

ProjDataInMemory&
ProjDataInMemory::operator*=(const float v)
{
  this->buffer *= v;
  return *this;
}

ProjDataInMemory&
ProjDataInMemory::operator/=(const float v)
{
  this->buffer /= v;
  return *this;
}

ProjDataInMemory
ProjDataInMemory::operator+(const ProjDataInMemory& iv) const
{
  ProjDataInMemory c(*this);
  return c += iv;
}

ProjDataInMemory
ProjDataInMemory::operator-(const ProjDataInMemory& iv) const
{
  ProjDataInMemory c(*this);
  return c -= iv;
}

ProjDataInMemory
ProjDataInMemory::operator*(const ProjDataInMemory& iv) const
{
  ProjDataInMemory c(*this);
  return c *= iv;
}

ProjDataInMemory
ProjDataInMemory::operator/(const ProjDataInMemory& iv) const
{
  ProjDataInMemory c(*this);
  return c /= iv;
}

ProjDataInMemory
ProjDataInMemory::operator+(const float a) const
{
  ProjDataInMemory c(*this);
  return c += a;
}

ProjDataInMemory
ProjDataInMemory::operator-(const float a) const
{
  ProjDataInMemory c(*this);
  return c -= a;
}

ProjDataInMemory
ProjDataInMemory::operator*(const float a) const
{
  ProjDataInMemory c(*this);
  return c *= a;
}

ProjDataInMemory
ProjDataInMemory::operator/(const float a) const
{
  ProjDataInMemory c(*this);
  return c /= a;
}

void
ProjDataInMemory::axpby(const float a, const ProjData& x, const float b, const ProjData& y)
{
  xapyb(x, a, y, b);
}

void
ProjDataInMemory::xapyb(const ProjData& x, const float a, const ProjData& y, const float b)
{
  // To use this method, we require that all three proj data be ProjDataInMemory
  // So cast them. If any null pointers, fall back to default functionality
  const ProjDataInMemory* x_pdm = dynamic_cast<const ProjDataInMemory*>(&x);
  const ProjDataInMemory* y_pdm = dynamic_cast<const ProjDataInMemory*>(&y);
  // At least one is not ProjDataInMemory, fall back to default
  if (is_null_ptr(x_pdm) || is_null_ptr(y_pdm))
    {
      ProjData::xapyb(x, a, y, b);
      return;
    }

  // Else, all are ProjDataInMemory

  // First check that info match
  if (*get_proj_data_info_sptr() != *x.get_proj_data_info_sptr() || *get_proj_data_info_sptr() != *y.get_proj_data_info_sptr())
    error("ProjDataInMemory::xapyb: ProjDataInfo don't match");

#if 0
    // Get number of elements
    const std::size_t numel = size_all();

    float *buffer = this->buffer.get();
    const float *x_buffer = x_pdm->buffer.get();
    const float *y_buffer = y_pdm->buffer.get();

    for (unsigned i=0; i<numel; ++i)
        buffer[i] = a*x_buffer[i] + b*y_buffer[i];
#else
  this->buffer.xapyb(x_pdm->buffer, a, y_pdm->buffer, b);
#endif
}

void
ProjDataInMemory::xapyb(const ProjData& x, const ProjData& a, const ProjData& y, const ProjData& b)
{
  // To use this method, we require that all three proj data be ProjDataInMemory
  // So cast them. If any null pointers, fall back to default functionality
  const ProjDataInMemory* x_pdm = dynamic_cast<const ProjDataInMemory*>(&x);
  const ProjDataInMemory* y_pdm = dynamic_cast<const ProjDataInMemory*>(&y);
  const ProjDataInMemory* a_pdm = dynamic_cast<const ProjDataInMemory*>(&a);
  const ProjDataInMemory* b_pdm = dynamic_cast<const ProjDataInMemory*>(&b);

  // At least one is not ProjDataInMemory, fall back to default
  if (is_null_ptr(x_pdm) || is_null_ptr(y_pdm) || is_null_ptr(a_pdm) || is_null_ptr(b_pdm))
    {
      ProjData::xapyb(x, a, y, b);
      return;
    }

  // Else, all are ProjDataInMemory

  // First check that info match
  if (*get_proj_data_info_sptr() != *x.get_proj_data_info_sptr() || *get_proj_data_info_sptr() != *y.get_proj_data_info_sptr()
      || *get_proj_data_info_sptr() != *a.get_proj_data_info_sptr() || *get_proj_data_info_sptr() != *b.get_proj_data_info_sptr())
    error("ProjDataInMemory::xapyb: ProjDataInfo don't match");

#if 0
    // Get number of elements
    const std::size_t numel = size_all();

    float *buffer = this->buffer.get();
    const float *x_buffer = x_pdm->buffer.get();
    const float *y_buffer = y_pdm->buffer.get();
    const float *a_buffer = a_pdm->buffer.get();
    const float *b_buffer = b_pdm->buffer.get();

    for (unsigned i=0; i<numel; ++i)
        buffer[i] = a_buffer[i]*x_buffer[i] + b_buffer[i]*y_buffer[i];
#else
  this->buffer.xapyb(x_pdm->buffer, a_pdm->buffer, y_pdm->buffer, b_pdm->buffer);
#endif
}

void
ProjDataInMemory::sapyb(const float a, const ProjData& y, const float b)
{
  this->xapyb(*this, a, y, b);
}

void
ProjDataInMemory::sapyb(const ProjData& a, const ProjData& y, const ProjData& b)
{
  this->xapyb(*this, a, y, b);
}

END_NAMESPACE_STIR
