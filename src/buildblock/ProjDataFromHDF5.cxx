/*!

  \file
  \ingroup projdata

  \brief Implementations for class stir::ProjDataGEAdvance

  \author Palak Wadhwa
  \author Nikos Efthimiou
  \author Kris Thielemans

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2009-06-22, Hammersmith Imanet Ltd
    Copyright (C) 2011, Kris Thielemans
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/



#include "stir/ProjDataFromHDF5.h"
#include "stir/IndexRange.h"
#include "stir/IndexRange3D.h"
#include "stir/IO/HDF5Wrapper.h"

START_NAMESPACE_STIR

ProjDataFromHDF5::ProjDataFromHDF5(shared_ptr<ProjDataInfo> input_proj_data_info_sptr,
                                   const std::string& input_filename) :
    ProjData()
{
    m_input_hdf5_sptr.reset(new HDF5Wrapper(input_filename));
    exam_info_sptr = m_input_hdf5_sptr->get_exam_info_sptr();
    proj_data_info_ptr = input_proj_data_info_sptr;

    initialise_segment_sequence();
    initialise_ax_pos_offset();
}

ProjDataFromHDF5::ProjDataFromHDF5(shared_ptr<ProjDataInfo> input_proj_data_info_sptr,
                                   shared_ptr<HDF5Wrapper> input_hdf5) :
    ProjData(input_hdf5->get_exam_info_sptr(), input_proj_data_info_sptr)
{
    m_input_hdf5_sptr = input_hdf5;

    initialise_segment_sequence();
    initialise_ax_pos_offset();
}

ProjDataFromHDF5::ProjDataFromHDF5(shared_ptr<ExamInfo> input_exam_info_sptr,
                                   shared_ptr<ProjDataInfo> input_proj_data_info_sptr,
                                   shared_ptr<HDF5Wrapper> input_hdf5) :
    ProjData(input_exam_info_sptr, input_proj_data_info_sptr)
{  
    m_input_hdf5_sptr = input_hdf5;

    initialise_segment_sequence();
    initialise_ax_pos_offset();
}

void ProjDataFromHDF5::initialise_segment_sequence()
{
    segment_sequence.resize(2*get_max_segment_num()+1);
    segment_sequence[0] = 0;

    for (int segment_num = 1; segment_num<=get_max_segment_num(); ++segment_num)
    {
        segment_sequence[2*segment_num-1] = -segment_num;
        segment_sequence[2*segment_num] = segment_num;
    }
}

void ProjDataFromHDF5::initialise_ax_pos_offset()
{
  seg_ax_offset.resize(get_num_segments());

  seg_ax_offset[0] = 0;

  unsigned int previous_value = 0;

  for (unsigned int i_seg = 1; i_seg < get_num_segments(); ++i_seg)
  {
      int segment_num = segment_sequence[i_seg-1];

      seg_ax_offset[i_seg] = static_cast<unsigned int>(get_num_axial_poss(segment_num)) +
                                                       previous_value;
      previous_value = seg_ax_offset[i_seg];
  }
}

Viewgram<float>
ProjDataFromHDF5::
get_viewgram(const int view_num, const int segment_num,
             const bool make_num_tangential_poss_odd) const
{

    Viewgram<float> ret_viewgram = get_empty_viewgram(view_num, segment_num, make_num_tangential_poss_odd);
    ret_viewgram.fill(0.0);
    //! \todo NE: Get the number of tof positions from the proj_data_info_ptr
    const unsigned int num_tof_poss = 27;

    m_input_hdf5_sptr->initialise_proj_data_data("", view_num + 1);

    std::array<unsigned long long int, 3> stride = {1, 1, 1};
    const std::array<unsigned long long int, 3> count  =
    {static_cast<unsigned long long int>(get_num_axial_poss(segment_num)), num_tof_poss,
     static_cast<unsigned long long int>(get_num_tangential_poss())};

    std::array<unsigned long long int, 3> offset = {seg_ax_offset[find_segment_index_in_sequence(segment_num)], 0, 0};
    std::array<unsigned long long int, 3> block = {1, 1, 1};
    unsigned int total_size = get_num_tangential_poss() * num_tof_poss * get_num_axial_poss(segment_num);

    stir::Array<1, unsigned char> tmp(0, total_size);

    m_input_hdf5_sptr->get_from_dataset(offset, count, stride, block, tmp);

    // Flatten the TOF poss
    for(int axial_pos = ret_viewgram.get_min_axial_pos_num(), i_axial = 0; axial_pos <= ret_viewgram.get_max_axial_pos_num(); ++axial_pos, ++i_axial)
    {
        unsigned int ii = i_axial * get_num_tangential_poss() * num_tof_poss;
        for(unsigned int i_tof = 0; i_tof < num_tof_poss; ++i_tof)
        {
            unsigned int jj = get_num_tangential_poss() * i_tof;
            for (int tang_pos = ret_viewgram.get_min_tangential_pos_num(), i_tang = 0; tang_pos <= ret_viewgram.get_max_tangential_pos_num(); ++tang_pos, ++i_tang)
            {
                ret_viewgram[axial_pos][tang_pos] += static_cast<float>(tmp[i_tang + jj + ii]);
            }
        }
    }

    return ret_viewgram;
}


Succeeded ProjDataFromHDF5::set_viewgram(const Viewgram<float>& v)
{
    // but this is difficult: how to adjust the scale factors when writing only 1 viewgram ?
    error("ProjDataFromGEHDF5::set_viewgram not implemented yet\n");
    return Succeeded::no;
}

Sinogram<float> ProjDataFromHDF5::get_sinogram(const int ax_pos_num, const int segment_num,const bool make_num_tangential_poss_odd) const
{ 
    // TODO
    warning("ProjDataGEAdvance::get_sinogram not implemented yet\n");
    return get_empty_sinogram(ax_pos_num, segment_num);
}

Succeeded ProjDataFromHDF5::set_sinogram(const Sinogram<float>& s)
{
    error("ProjDataFromHDF5::set_sinogram not implemented yet\n");
    return Succeeded::no;
}



//! \todo move inlines in seprate file.
/*
 *
 *  INLINES
 *
 *
 */

unsigned int
ProjDataFromHDF5::find_segment_index_in_sequence(const int segment_num) const
{
#ifndef STIR_NO_NAMESPACES
  std::vector<int>::const_iterator iter =
    std::find(segment_sequence.begin(), segment_sequence.end(), segment_num);
#else
  vector<int>::const_iterator iter =
    find(segment_sequence.begin(), segment_sequence.end(), segment_num);
#endif
  // TODO do some proper error handling here
  assert(iter !=  segment_sequence.end());
  return static_cast<int>(iter - segment_sequence.begin());
}

std::vector<int>
ProjDataFromHDF5::get_segment_sequence_in_hdf5() const
{ return segment_sequence; }


END_NAMESPACE_STIR

