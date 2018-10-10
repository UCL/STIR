/*!

  \file
  \ingroup projdata

  \brief Implementations for class stir::ProjDataGEHDF5

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
#include "stir/IndexRange4D.h"
#include "stir/IO/HDF5Wrapper.h"
#include "stir/display.h"
#include "stir/CPUTimer.h"
#include "stir/HighResWallClockTimer.h"
#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::fstream;
using std::ios;
#endif

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
    initialise_viewgram_buffer();
}

ProjDataFromHDF5::ProjDataFromHDF5(shared_ptr<ProjDataInfo> input_proj_data_info_sptr,
                                   shared_ptr<HDF5Wrapper> input_hdf5) :
    ProjData(input_hdf5->get_exam_info_sptr(), input_proj_data_info_sptr)
{
    m_input_hdf5_sptr = input_hdf5;

    initialise_segment_sequence();
    initialise_ax_pos_offset();
    initialise_viewgram_buffer();
}

ProjDataFromHDF5::ProjDataFromHDF5(shared_ptr<ExamInfo> input_exam_info_sptr,
                                   shared_ptr<ProjDataInfo> input_proj_data_info_sptr,
                                   shared_ptr<HDF5Wrapper> input_hdf5) :
    ProjData(input_exam_info_sptr, input_proj_data_info_sptr)
{
    m_input_hdf5_sptr = input_hdf5;

    initialise_segment_sequence();
    initialise_ax_pos_offset();
    initialise_viewgram_buffer();
}

void ProjDataFromHDF5::initialise_viewgram_buffer()
{
//! \todo NE: Get the number of tof positions from the proj_data_info_ptr
const unsigned int num_tof_poss = 27;
const unsigned int max_num_axial_poss = 1981;
const unsigned int get_num_viewgrams = 224;
unsigned int total_size = get_num_tangential_poss() * num_tof_poss * max_num_axial_poss;

//stir::Array<1, unsigned char> tmp(0, total_size-1);
this->tof_data.resize(IndexRange4D(get_num_viewgrams, get_num_tangential_poss(), num_tof_poss, max_num_axial_poss));

Array<1,unsigned char> buffer(total_size);

 for (int view_num = get_min_view_num(); view_num <= get_max_view_num(); view_num++)
{
    m_input_hdf5_sptr->initialise_proj_data_data("", view_num + 1);

    std::array<unsigned long long int, 3> stride = {1, 1, 1};
    const std::array<unsigned long long int, 3> count  = {max_num_axial_poss, num_tof_poss,static_cast<unsigned long long int>(get_num_tangential_poss())};
//    {static_cast<unsigned long long int>(get_num_axial_poss(segment_num)), num_tof_poss,
//     static_cast<unsigned long long int>(get_num_tangential_poss())};

    std::array<unsigned long long int, 3> offset = {0,0,0};
//    {seg_ax_offset[find_segment_index_in_sequence(segment_num)], 0, 0};
    std::array<unsigned long long int, 3> block = {1, 1, 1};

m_input_hdf5_sptr->get_from_dataset(offset, count, stride, block, buffer);
std::copy(buffer.begin(), buffer.end(), tof_data[view_num].begin_all());
}

}

void ProjDataFromHDF5::initialise_segment_sequence()
{
    segment_sequence.resize(2*get_max_segment_num()+1);
    segment_sequence[0] = 0;
// PW Flipped the segments, segment sequence is now as: 0,1,-1 and so on.
    for (int segment_num = 1; segment_num<=get_max_segment_num(); ++segment_num)
    {
        segment_sequence[2*segment_num-1] = segment_num;
        segment_sequence[2*segment_num] = -segment_num;
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
    std::cout<<"Now processing segment "<<segment_num<<"for view "<<view_num<<std::endl;
    Viewgram<float> ret_viewgram = get_empty_viewgram(view_num, segment_num, make_num_tangential_poss_odd);
    ret_viewgram.fill(0.0);
    //! \todo NE: Get the number of tof positions from the proj_data_info_ptr
    const unsigned int num_tof_poss = 27;
    const unsigned int max_num_axial_poss = 1981;

    // PW Attempt to flip the tangential and view numbers.
   for (int tang_pos = ret_viewgram.get_min_tangential_pos_num(), i_tang = 0; tang_pos <= ret_viewgram.get_max_tangential_pos_num(), i_tang<=static_cast<unsigned long long int>(get_num_tangential_poss())-1; ++tang_pos, ++i_tang)
      for(int i_axial=0, axial_pos = seg_ax_offset[find_segment_index_in_sequence(segment_num)]; i_axial<=static_cast<unsigned long long int>(get_num_axial_poss(segment_num))-1 , axial_pos <= seg_ax_offset[find_segment_index_in_sequence(segment_num)]+static_cast<unsigned long long int>(get_num_axial_poss(segment_num))-1; i_axial++, axial_pos++)
        for (int tof_poss = 0; tof_poss <= num_tof_poss-1; tof_poss++)
      {
                ret_viewgram[i_axial][-tang_pos] += static_cast<float> (tof_data[223-view_num][i_tang][tof_poss][axial_pos]);
            }

#if 0
    ofstream write_tof_data;
    write_tof_data.open("uncompressed_sino_tof_data.txt",ios::out);
    for ( int i =tof_data.get_min_index(); i<=tof_data.get_max_index();i++)
      {
        for ( int j =tof_data[i].get_min_index(); j <=tof_data[i].get_max_index(); j++)
           {
            for(int k = tof_data[i][j].get_min_index(); k <=tof_data[i][j].get_max_index(); k++)
         write_tof_data << tof_data[i][j][k] << "   " ;
           }
                  write_tof_data << std::endl;
           }

    ofstream write_ret_viewgram_data;
    write_ret_viewgram_data.open("uncompressed_sino_ret_viewgram_data.txt",ios::out);
    for ( int i =ret_viewgram.get_min_index(); i<=ret_viewgram.get_max_index();i++)
      {
        for ( int j =ret_viewgram[i].get_min_index(); j <=ret_viewgram[i].get_max_index(); j++)
           {
         write_ret_viewgram_data << ret_viewgram[i][j] << "   " ;
           }
                  write_ret_viewgram_data << std::endl;
           }
#endif
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
