/*!

  \file
  \ingroup projdata
  \ingroup GE
  \brief Implementations for class stir::GE::RDF_HDF5::ProjDataGEHDF5

  \author Palak Wadhwa
  \author Nikos Efthimiou
  \author Kris Thielemans

*/
/*
    Copyright (C) 2017-2019, University of Leeds
    Copyright (C) 2018-2019 University of Leeds
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



#include "stir/ProjDataGEHDF5.h"
#include "stir/IndexRange.h"
#include "stir/IndexRange3D.h"
#include "stir/IndexRange4D.h"
#include "stir/IO/GEHDF5Wrapper.h"
#include "stir/display.h"
#include "stir/CPUTimer.h"
#include "stir/HighResWallClockTimer.h"
#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::fstream;
using std::ios;
#endif

START_NAMESPACE_STIR

namespace GE {
namespace RDF_HDF5 {

ProjDataGEHDF5::ProjDataGEHDF5(const std::string& input_filename) :
  ProjData()
{
    this->m_input_hdf5_sptr.reset(new GEHDF5Wrapper(input_filename));
    this->initialise_from_wrapper();
}

ProjDataGEHDF5::ProjDataGEHDF5(shared_ptr<GEHDF5Wrapper> input_hdf5_sptr) :
  ProjData(),
  m_input_hdf5_sptr(input_hdf5_sptr)
{
    this->initialise_from_wrapper();
}

void ProjDataGEHDF5::initialise_from_wrapper()
{
    this->exam_info_sptr = this->m_input_hdf5_sptr->get_exam_info_sptr();
    shared_ptr<Scanner> scanner_sptr = m_input_hdf5_sptr->get_scanner_sptr();
    this->proj_data_info_sptr =
      ProjDataInfo::construct_proj_data_info(scanner_sptr,
                                            /*span*/ 2,
                                            /* max_delta*/ scanner_sptr->get_num_rings()-1,
                                            /* num_views */ scanner_sptr->get_num_detectors_per_ring()/2,
                                            /* num_tangential_poss */ scanner_sptr->get_max_num_non_arccorrected_bins(),
                                            /* arc_corrected */ false
                                             );

    this->initialise_segment_sequence();
    this->initialise_ax_pos_offset();
    this->initialise_viewgram_buffer();
}

void ProjDataGEHDF5::initialise_viewgram_buffer()
{

  if (!this->tof_data.empty())
    error("there is already data loaded. Aborting");

  tof_data.reserve(get_num_views());
  Array<3,unsigned char> buffer;

  for (int view_num = get_min_view_num(); view_num <= get_max_view_num(); view_num++)
  {
      // view numbering for initialise_proj_data starts from 1
      m_input_hdf5_sptr->initialise_proj_data(view_num - get_min_view_num() + 1);

      m_input_hdf5_sptr->get_from_dataset(buffer);
      this->tof_data.push_back(buffer);
  }

}

void ProjDataGEHDF5::initialise_segment_sequence()
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

void ProjDataGEHDF5::initialise_ax_pos_offset()
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
ProjDataGEHDF5::
get_viewgram(const int view_num, const int segment_num,
             const bool make_num_tangential_poss_odd) const
{
    if (make_num_tangential_poss_odd)
        error("make_num_tangential_poss_odd not supported by ProjDataGEHDF5");
    Viewgram<float> ret_viewgram = get_empty_viewgram(view_num, segment_num);
    ret_viewgram.fill(0.0);
    //! \todo NE: Get the number of tof positions from the proj_data_info_ptr
    const unsigned int num_tof_poss = 27;
   // const unsigned int max_num_axial_poss = 1981;
    //! \todo Hard-wired numbers to be changed.
    // PW Attempt to flip the tangential and view numbers. The TOF bins are added below to return non TOF viewgram.
   for (int tang_pos = ret_viewgram.get_min_tangential_pos_num(), i_tang = 0;
        tang_pos <= ret_viewgram.get_max_tangential_pos_num();
        ++tang_pos, ++i_tang)
     for(int i_axial=get_min_axial_pos_num(segment_num), axial_pos = seg_ax_offset[find_segment_index_in_sequence(segment_num)];
         i_axial<=get_max_axial_pos_num(segment_num);
         i_axial++, axial_pos++)
        for (int tof_poss = 0; tof_poss <= num_tof_poss-1; tof_poss++)
          {
            // TODO 223 -> get_num_views()-1
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


Succeeded ProjDataGEHDF5::set_viewgram(const Viewgram<float>& v)
{
    // but this is difficult: how to adjust the scale factors when writing only 1 viewgram ?
    error("ProjDataFromGEHDF5::set_viewgram not implemented yet");
    return Succeeded::no;
}

Sinogram<float> ProjDataGEHDF5::get_sinogram(const int ax_pos_num, const int segment_num,const bool make_num_tangential_poss_odd) const
{
    // TODO
    error("ProjDataGEHDF5::get_sinogram not implemented yet");
    return get_empty_sinogram(ax_pos_num, segment_num);
}

Succeeded ProjDataGEHDF5::set_sinogram(const Sinogram<float>& s)
{
    error("ProjDataGEHDF5::set_sinogram not implemented yet");
    return Succeeded::no;
}

unsigned int
ProjDataGEHDF5::find_segment_index_in_sequence(const int segment_num) const
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
ProjDataGEHDF5::get_segment_sequence_in_hdf5() const
{ return segment_sequence; }


} // namespace
}
END_NAMESPACE_STIR
