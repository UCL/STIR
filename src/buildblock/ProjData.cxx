/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2010-10-15, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 -2013, Kris Thielemans
    Copyright (C) 2016, University of Hull
    Copyright (C) 2015, 2020, 2022, 2023 University College London
    Copyright (C) 2021-2022, Commonwealth Scientific and Industrial Research Organisation
    Copyright (C) 2021, Rutherford Appleton Laboratory STFC
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup projdata

  \brief Implementations for non-inline functions of class stir::ProjData

  \author Nikos Efthimiou
  \author Kris Thielemans
  \author Ashley Gillman
  \author Evgueni Ovtchinnikov
  \author Gemma Fardell
  \author PARAPET project
*/
#include "stir/ProjData.h"
#include "stir/ExamInfo.h"
#include "stir/Succeeded.h"
#include "stir/RelatedViewgrams.h"
#include "stir/Viewgram.h"
#include "stir/DataSymmetriesForViewSegmentNumbers.h"

// for read_from_file
#include "stir/IO/FileSignature.h"
#include "stir/IO/interfile.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataFromStream.h" // needed for converting ProjDataFromStream* to ProjData*
#include "stir/ProjDataInMemory.h"   // needed for subsets
#include "stir/ProjDataInfoSubsetByView.h"
#include "stir/Viewgram.h"

#ifdef HAVE_HDF5
#  include "stir/ProjDataGEHDF5.h"
#  include "stir/IO/GEHDF5Wrapper.h"
#endif
#include "stir/IO/stir_ecat7.h"
#include "stir/ViewgramIndices.h"
#include "stir/is_null_ptr.h"
#include "stir/numerics/norm.h"
#include <cstring>
#include <fstream>
#include <algorithm>
#include "stir/error.h"

using std::istream;
using std::fstream;
using std::ios;
using std::string;
using std::vector;

START_NAMESPACE_STIR

/*!
   This function will attempt to determine the type of projection data in the file,
   construct an object of the appropriate type, and return a pointer to
   the object.

   The return value is a shared_ptr, to make sure that the caller will
   delete the object.

   If more than 1 projection data set is in the file, only the first is read.

   When the file is not readable for some reason, the program is aborted
   by calling error().

   Currently supported:
   <ul>
   <li> Interfile (using  read_interfile_PDFS())
   <li> ECAT 7 3D sinograms and attenuation files
   >li> GE RDF9 (in HDF5)
   </ul>

   Developer's note: ideally the return value would be an stir::unique_ptr.
*/

shared_ptr<ProjData>
ProjData::read_from_file(const string& filename, const std::ios::openmode openmode)
{
  std::string actual_filename = filename;
  // parse filename to see if it's like filename,options
  {
    const std::size_t comma_pos = filename.find(',');
    if (comma_pos != std::string::npos)
      {
        actual_filename.resize(comma_pos);
      }
  }

  const FileSignature file_signature(actual_filename);
  const char* signature = file_signature.get_signature();

#ifdef HAVE_LLN_MATRIX
  // ECAT 7
  if (strncmp(signature, "MATRIX", 6) == 0)
    {
#  ifndef NDEBUG
      info("ProjData::read_from_file trying to read " + filename + " as ECAT7", 3);
#  endif
      USING_NAMESPACE_ECAT;
      USING_NAMESPACE_ECAT7;

      if (is_ECAT7_emission_file(actual_filename) || is_ECAT7_attenuation_file(actual_filename))
        {
          info("Reading frame 1, gate 1, data 0, bed 0 from file " + actual_filename, 3);
          shared_ptr<ProjData> proj_data_sptr(ECAT7_to_PDFS(filename, /*frame_num, gate_num, data_num, bed_num*/ 1, 1, 0, 0));
          return proj_data_sptr;
        }
      else
        {
          if (is_ECAT7_file(actual_filename))
            error("ProjData::read_from_file ECAT7 file " + actual_filename + " is of unsupported file type");
        }
    }
#endif // HAVE_LLN_MATRIX

  // Interfile
  if (is_interfile_signature(signature))
    {
#ifndef NDEBUG
      info("ProjData::read_from_file trying to read " + filename + " as Interfile", 3);
#endif
      shared_ptr<ProjData> ptr(read_interfile_PDFS(filename, openmode));
      if (!is_null_ptr(ptr))
        return ptr;
    }

#ifdef HAVE_HDF5
  if (GE::RDF_HDF5::GEHDF5Wrapper::check_GE_signature(actual_filename))
    {
#  ifndef NDEBUG
      info("ProjData::read_from_file trying to read " + filename + " as GE HDF5", 3);
#  endif
      shared_ptr<ProjData> ptr(new GE::RDF_HDF5::ProjDataGEHDF5(filename));
      if (!is_null_ptr(ptr))
        return ptr;
    }
#endif // GE HDF5

  error("ProjData::read_from_file could not read projection data " + filename
        + ".\n"
          "Unsupported file format? Aborting.");
  // need to return something to satisfy the compiler, but we never get here
  shared_ptr<ProjData> null_ptr;
  return null_ptr;
}

// void
// ProjData::set_exam_info(ExamInfo const& new_exam_info)
//{
//   this->exam_info_sptr.reset(new ExamInfo(new_exam_info));
// }

unique_ptr<ProjDataInMemory>
ProjData::get_subset(const std::vector<int>& views) const
{
  auto subset_proj_data_info_sptr = std::make_shared<ProjDataInfoSubsetByView>(proj_data_info_sptr, views);
  unique_ptr<ProjDataInMemory> subset_proj_data_uptr(new ProjDataInMemory(exam_info_sptr, subset_proj_data_info_sptr));

  for (int timing_pos_num = this->get_min_tof_pos_num(); timing_pos_num <= this->get_max_tof_pos_num(); ++timing_pos_num)
    for (int segment_num = get_min_segment_num(); segment_num <= get_max_segment_num(); ++segment_num)
      {
        for (int subset_view_num = 0; subset_view_num < static_cast<int>(views.size()); ++subset_view_num)
          {
            const auto viewgram = this->get_viewgram(views[subset_view_num], segment_num, false, timing_pos_num);
            // construct new one with data from viewgram, but appropriate meta-data
            const Viewgram<float> subset_viewgram(
                viewgram, subset_proj_data_info_sptr, subset_view_num, segment_num, timing_pos_num);
            if (subset_proj_data_uptr->set_viewgram(subset_viewgram) != Succeeded::yes)
              error("ProjData::get_subset failed to set a viewgram");
          }
      }

  return subset_proj_data_uptr;
}

Viewgram<float>
ProjData::get_empty_viewgram(const ViewgramIndices& ind) const
{
  return proj_data_info_sptr->get_empty_viewgram(ind);
}

Sinogram<float>
ProjData::get_empty_sinogram(const SinogramIndices& ind) const
{
  return proj_data_info_sptr->get_empty_sinogram(ind);
}

Viewgram<float>
ProjData::get_empty_viewgram(const int view_num,
                             const int segment_num,
                             const bool make_num_tangential_poss_odd,
                             const int timing_pos) const
{
  return proj_data_info_sptr->get_empty_viewgram(view_num, segment_num, make_num_tangential_poss_odd, timing_pos);
}

Sinogram<float>
ProjData::get_empty_sinogram(const int ax_pos_num,
                             const int segment_num,
                             const bool make_num_tangential_poss_odd,
                             const int timing_pos) const
{
  return proj_data_info_sptr->get_empty_sinogram(ax_pos_num, segment_num, make_num_tangential_poss_odd, timing_pos);
}

SegmentBySinogram<float>
ProjData::get_empty_segment_by_sinogram(const int segment_num,
                                        const bool make_num_tangential_poss_odd,
                                        const int timing_pos) const
{
  return proj_data_info_sptr->get_empty_segment_by_sinogram(segment_num, make_num_tangential_poss_odd, timing_pos);
}

SegmentByView<float>
ProjData::get_empty_segment_by_view(const int segment_num, const bool make_num_tangential_poss_odd, const int timing_pos) const
{
  return proj_data_info_sptr->get_empty_segment_by_view(segment_num, make_num_tangential_poss_odd, timing_pos);
}

SegmentBySinogram<float>
ProjData::get_empty_segment_by_sinogram(const SegmentIndices& ind) const
{
  return proj_data_info_sptr->get_empty_segment_by_sinogram(ind);
}

SegmentByView<float>
ProjData::get_empty_segment_by_view(const SegmentIndices& ind) const
{
  return proj_data_info_sptr->get_empty_segment_by_view(ind);
}

RelatedViewgrams<float>
ProjData::get_empty_related_viewgrams(const ViewgramIndices& view_segmnet_num,
                                      const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_used,
                                      const bool make_num_tangential_poss_odd,
                                      const int timing_pos) const

{
  return proj_data_info_sptr->get_empty_related_viewgrams(
      view_segmnet_num, symmetries_used, make_num_tangential_poss_odd, timing_pos);
}

RelatedViewgrams<float>
ProjData::get_related_viewgrams(const ViewgramIndices& viewgram_indices,
                                const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_used,
                                const bool make_num_tangential_poss_odd,
                                const int timing_pos) const
{
  vector<ViewSegmentNumbers> pairs;
  symmetries_used->get_related_view_segment_numbers(pairs, viewgram_indices);

  vector<Viewgram<float>> viewgrams;
  viewgrams.reserve(pairs.size());

  for (unsigned int i = 0; i < pairs.size(); i++)
    {
      // TODO optimise to get shared proj_data_info_ptr
      // TODOTOF
      pairs[i].timing_pos_num() = timing_pos;
      viewgrams.push_back(this->get_viewgram(pairs[i]));
    }

  return RelatedViewgrams<float>(viewgrams, symmetries_used);
}

// std::vector<float>
// ProjData::get_related_bin_values(const std::vector<Bin>& r_bins) const
//{

//    std::vector<float> values;
//    values.reserve(r_bins.size());

//    for (std::vector <Bin>::const_iterator r_bins_iterator = r_bins.begin();
//         r_bins_iterator != r_bins.end(); ++r_bins_iterator)
//    {
//        values.push_back(this->get_bin_value(*r_bins_iterator));
//    }

//    return values;
//}

Succeeded
ProjData::set_related_viewgrams(const RelatedViewgrams<float>& viewgrams)
{

  RelatedViewgrams<float>::const_iterator r_viewgrams_iter = viewgrams.begin();
  while (r_viewgrams_iter != viewgrams.end())
    {
      if (set_viewgram(*r_viewgrams_iter) == Succeeded::no)
        return Succeeded::no;
      ++r_viewgrams_iter;
    }
  return Succeeded::yes;
}

#if 0
  for (int i=0; i<viewgrams.get_num_viewgrams(); ++i)
  {
    if (set_viewgram(viewgrams.get_viewgram_reference(i)) == Succeeded::no)
      return Succeeded::no;
  }
  return Succeeded::yes;
}
#endif

SegmentBySinogram<float>
ProjData::get_segment_by_sinogram(const int segment_num, const int timing_pos) const
{
  SegmentBySinogram<float> segment = proj_data_info_sptr->get_empty_segment_by_sinogram(segment_num, false, timing_pos);
  // TODO optimise to get shared proj_data_info_ptr
  for (int view_num = get_min_view_num(); view_num <= get_max_view_num(); ++view_num)
    segment.set_viewgram(get_viewgram(view_num, segment_num, false, timing_pos));
  return segment;
}

SegmentByView<float>
ProjData::get_segment_by_view(const int segment_num, const int timing_pos) const
{
  SegmentByView<float> segment = proj_data_info_sptr->get_empty_segment_by_view(segment_num, false, timing_pos);
  // TODO optimise to get shared proj_data_info_ptr
  for (int view_num = get_min_view_num(); view_num <= get_max_view_num(); ++view_num)
    segment.set_viewgram(get_viewgram(view_num, segment_num, false, timing_pos));
  return segment;
}

Succeeded
ProjData::set_segment(const SegmentBySinogram<float>& segment)
{
  for (int view_num = get_min_view_num(); view_num <= get_max_view_num(); ++view_num)
    {
      if (set_viewgram(segment.get_viewgram(view_num)) == Succeeded::no)
        return Succeeded::no;
    }
  return Succeeded::yes;
}

Succeeded
ProjData::set_segment(const SegmentByView<float>& segment)
{
  for (int view_num = get_min_view_num(); view_num <= get_max_view_num(); ++view_num)
    {
      if (set_viewgram(segment.get_viewgram(view_num)) == Succeeded::no)
        return Succeeded::no;
    }
  return Succeeded::yes;
}

void
ProjData::fill(const float value)
{
  for (int timing_pos_num = this->get_min_tof_pos_num(); timing_pos_num <= this->get_max_tof_pos_num(); ++timing_pos_num)
    {
      for (int segment_num = this->get_min_segment_num(); segment_num <= this->get_max_segment_num(); ++segment_num)
        {
          SegmentByView<float> segment(this->get_empty_segment_by_view(segment_num, false, timing_pos_num));
          segment.fill(value);
          if (this->set_segment(segment) == Succeeded::no)
            error("Error setting segment of projection data");
        }
    }
}

void
ProjData::fill(const ProjData& proj_data)
{
  shared_ptr<ProjDataInfo> source_proj_data_info_sptr = proj_data.get_proj_data_info_sptr()->create_shared_clone();
  source_proj_data_info_sptr->reduce_segment_range(std::max(this->get_min_segment_num(), proj_data.get_min_segment_num()),
                                                   std::min(this->get_max_segment_num(), proj_data.get_max_segment_num()));
  if ((*this->get_proj_data_info_sptr()) != (*source_proj_data_info_sptr))
    error("Filling projection data from incompatible  source");

  for (int segment_num = this->get_min_segment_num(); segment_num <= this->get_max_segment_num(); ++segment_num)
    {
      for (int timing_pos_num = this->get_min_tof_pos_num(); timing_pos_num <= this->get_max_tof_pos_num(); ++timing_pos_num)
        {
          if (this->set_segment(proj_data.get_segment_by_view(segment_num, timing_pos_num)) == Succeeded::no)
            error("Error setting segment of projection data");
        }
    }
}

ProjData::ProjData()
    : ExamData()
{}

ProjData::ProjData(const shared_ptr<const ExamInfo>& exam_info_sptr, const shared_ptr<const ProjDataInfo>& proj_data_info_sptr)
    : ExamData(exam_info_sptr),
      proj_data_info_sptr(proj_data_info_sptr)
{}

Succeeded
ProjData::write_to_file(const string& output_filename) const
{

  ProjDataInterfile out_projdata(get_exam_info_sptr(), this->proj_data_info_sptr, output_filename, ios::out);

  out_projdata.fill(*this);
  // will have thrown if it failed, so return it was ok
  return Succeeded::yes;
}

void
ProjData::axpby(const float a, const ProjData& x, const float b, const ProjData& y)
{
  xapyb(x, a, y, b);
}

void
ProjData::xapyb(const ProjData& x, const float a, const ProjData& y, const float b)
{
  if (*get_proj_data_info_sptr() != *x.get_proj_data_info_sptr() || *get_proj_data_info_sptr() != *y.get_proj_data_info_sptr())
    error("ProjData::xapyb: ProjDataInfo don't match");

  const int s_min = get_min_segment_num();
  const int s_max = get_max_segment_num();

  for (int timing_pos_num = this->get_min_tof_pos_num(); timing_pos_num <= this->get_max_tof_pos_num(); ++timing_pos_num)
    for (int s = s_min; s <= s_max; ++s)
      {
        auto seg = get_empty_segment_by_sinogram(s, false, timing_pos_num);
        const auto sx = x.get_segment_by_sinogram(s, timing_pos_num);
        const auto sy = y.get_segment_by_sinogram(s, timing_pos_num);
        seg.xapyb(sx, a, sy, b);
        if (set_segment(seg) == Succeeded::no)
          error("ProjData::xapyb: set_segment failed. Write-only file?");
      }
}

void
ProjData::xapyb(const ProjData& x, const ProjData& a, const ProjData& y, const ProjData& b)
{
  if (*get_proj_data_info_sptr() != *x.get_proj_data_info_sptr() || *get_proj_data_info_sptr() != *y.get_proj_data_info_sptr()
      || *get_proj_data_info_sptr() != *a.get_proj_data_info_sptr() || *get_proj_data_info_sptr() != *b.get_proj_data_info_sptr())
    error("ProjData::xapyb: ProjDataInfo don't match");

  const int s_min = get_min_segment_num();
  const int s_max = get_max_segment_num();

  for (int timing_pos_num = this->get_min_tof_pos_num(); timing_pos_num <= this->get_max_tof_pos_num(); ++timing_pos_num)
    for (int s = s_min; s <= s_max; ++s)
      {
        auto seg = get_empty_segment_by_sinogram(s, false, timing_pos_num);
        const auto sx = x.get_segment_by_sinogram(s, timing_pos_num);
        const auto sy = y.get_segment_by_sinogram(s, timing_pos_num);
        const auto sa = a.get_segment_by_sinogram(s, timing_pos_num);
        const auto sb = b.get_segment_by_sinogram(s, timing_pos_num);

        seg.xapyb(sx, sa, sy, sb);
        if (set_segment(seg) == Succeeded::no)
          error("ProjData::xapyb: set_segment failed. Write-only file?");
      }
}

void
ProjData::sapyb(const float a, const ProjData& y, const float b)
{
  this->xapyb(*this, a, y, b);
}

void
ProjData::sapyb(const ProjData& a, const ProjData& y, const ProjData& b)
{
  this->xapyb(*this, a, y, b);
}

float
ProjData::sum() const
{
  double t = 0.;
  for (int timing_pos_num = this->get_min_tof_pos_num(); timing_pos_num <= this->get_max_tof_pos_num(); ++timing_pos_num)
    for (int s = this->get_min_segment_num(); s <= this->get_max_segment_num(); ++s)
      {
        const SegmentIndices ind(s, timing_pos_num);
        t += this->get_segment_by_sinogram(ind).sum();
      }
  return static_cast<float>(t);
}

float
ProjData::find_max() const
{
  float t = 0;
  bool init = true;
  for (int timing_pos_num = this->get_min_tof_pos_num(); timing_pos_num <= this->get_max_tof_pos_num(); ++timing_pos_num)
    for (int s = this->get_min_segment_num(); s <= this->get_max_segment_num(); ++s)
      {
        const SegmentIndices ind(s, timing_pos_num);
        const auto t_seg = this->get_segment_by_sinogram(ind).find_max();
        if (init)
          {
            init = false;
            t = t_seg;
          }
        else
          {
            t = std::max(t, t_seg);
          }
      }
  return t;
}

float
ProjData::find_min() const
{
  float t = 0;
  bool init = true;
  for (int timing_pos_num = this->get_min_tof_pos_num(); timing_pos_num <= this->get_max_tof_pos_num(); ++timing_pos_num)
    for (int s = this->get_min_segment_num(); s <= this->get_max_segment_num(); ++s)
      {
        const SegmentIndices ind(s, timing_pos_num);
        const auto t_seg = this->get_segment_by_sinogram(ind).find_min();
        if (init)
          {
            init = false;
            t = t_seg;
          }
        else
          {
            t = std::min(t, t_seg);
          }
      }
  return t;
}

double
ProjData::norm_squared() const
{
  double t = 0.;
  for (int timing_pos_num = this->get_min_tof_pos_num(); timing_pos_num <= this->get_max_tof_pos_num(); ++timing_pos_num)
    for (int s = this->get_min_segment_num(); s <= this->get_max_segment_num(); ++s)
      {
        const SegmentIndices ind(s, timing_pos_num);
        const auto seg = this->get_segment_by_sinogram(ind);
        t += stir::norm_squared(seg.begin_all(), seg.end_all());
      }
  return t;
}

double
ProjData::norm() const
{
  return std::sqrt(this->norm_squared());
}

// static helper functions to iterate over segment and apply a function

// func(s1, s2) is supposed to modify s1
template <typename Func>
static ProjData&
apply_func(ProjData& self, const ProjData& arg, Func func)
{
  for (int timing_pos_num = self.get_min_tof_pos_num(); timing_pos_num <= self.get_max_tof_pos_num(); ++timing_pos_num)
    for (int s = self.get_min_segment_num(); s <= self.get_max_segment_num(); ++s)
      {
        const SegmentIndices ind(s, timing_pos_num);
        auto seg = self.get_segment_by_sinogram(ind);
        func(seg, arg.get_segment_by_sinogram(ind));
        self.set_segment(seg);
      }
  return self;
}

// func(s1) is supposed to modify s1
template <typename Func>
static ProjData&
apply_func(ProjData& self, Func func)
{
  for (int timing_pos_num = self.get_min_tof_pos_num(); timing_pos_num <= self.get_max_tof_pos_num(); ++timing_pos_num)
    for (int s = self.get_min_segment_num(); s <= self.get_max_segment_num(); ++s)
      {
        const SegmentIndices ind(s, timing_pos_num);
        auto seg = self.get_segment_by_sinogram(ind);
        func(seg);
        self.set_segment(seg);
      }
  return self;
}

ProjData&
ProjData::operator+=(const ProjData& arg)
{
  return apply_func(*this, arg, [](SegmentBySinogram<float>& s, const SegmentBySinogram<float>& s_arg) { s += s_arg; });
}

ProjData&
ProjData::operator-=(const ProjData& arg)
{
  return apply_func(*this, arg, [](SegmentBySinogram<float>& s, const SegmentBySinogram<float>& s_arg) { s -= s_arg; });
}

ProjData&
ProjData::operator*=(const ProjData& arg)
{
  return apply_func(*this, arg, [](SegmentBySinogram<float>& s, const SegmentBySinogram<float>& s_arg) { s *= s_arg; });
}

ProjData&
ProjData::operator/=(const ProjData& arg)
{
  return apply_func(*this, arg, [](SegmentBySinogram<float>& s, const SegmentBySinogram<float>& s_arg) { s /= s_arg; });
}

ProjData&
ProjData::operator+=(float arg)
{
  return apply_func(*this, [arg](SegmentBySinogram<float>& s) { s += arg; });
}

ProjData&
ProjData::operator-=(float arg)
{
  return apply_func(*this, [arg](SegmentBySinogram<float>& s) { s -= arg; });
}

ProjData&
ProjData::operator*=(float arg)
{
  return apply_func(*this, [arg](SegmentBySinogram<float>& s) { s *= arg; });
}

ProjData&
ProjData::operator/=(float arg)
{
  return apply_func(*this, [arg](SegmentBySinogram<float>& s) { s /= arg; });
}

std::vector<int>
ProjData::standard_segment_sequence(const ProjDataInfo& pdi)
{
  std::vector<int> segment_sequence(pdi.get_num_segments());
  if (pdi.get_num_segments() == 0)
    return segment_sequence;

  const int max_segment_num = pdi.get_max_segment_num();
  const int min_segment_num = pdi.get_min_segment_num();
  segment_sequence[0] = 0;
  unsigned idx = 1;
  int segment_num = 1;
  while (idx < segment_sequence.size())
    {
      if (segment_num <= max_segment_num)
        segment_sequence[idx++] = segment_num;
      if (-segment_num >= min_segment_num)
        segment_sequence[idx++] = -segment_num;
      ++segment_num;
    }
  return segment_sequence;
}

END_NAMESPACE_STIR
