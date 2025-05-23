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

    SPDX-License-Identifier: Apache-2.0

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
    this->proj_data_info_sptr = this->m_input_hdf5_sptr->get_proj_data_info_sptr()->create_shared_clone();
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

      m_input_hdf5_sptr->read_sinogram(buffer);
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

  for (int i_seg = 1; i_seg < get_num_segments(); ++i_seg)
  {
      const int segment_num = segment_sequence[i_seg-1];

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
    // not necessary
    // ret_viewgram.fill(0.0);

    // find num TOF bins
    BasicCoordinate<3, int> min_index, max_index;
    tof_data[0].get_regular_range(min_index,max_index);
    const int num_tof_poss = max_index[2] - min_index[2] + 1;
    if (num_tof_poss <= 0)
      error("ProjDataGEHDF5: internal error on TOF data dimension");
    if (proj_data_info_sptr->get_scanner_ptr()->get_type() == Scanner::PETMR_Signa)
      if (num_tof_poss != 27)
      error("ProjDataGEHDF5: internal error on TOF data dimension for GE Signa");
      
    // PW flip the tangential and view numbers. The TOF bins are added below to return non TOF viewgram.
    if (get_min_view_num() != 0)
      error("ProjDataGEHDF5: internal error on views");
    if (get_max_tangential_pos_num()+get_min_tangential_pos_num() != 0)
      error("ProjDataGEHDF5: internal error on tangential positions");
   for (int tang_pos = ret_viewgram.get_min_tangential_pos_num(), i_tang = 0;
        tang_pos <= ret_viewgram.get_max_tangential_pos_num();
        ++tang_pos, ++i_tang)
     for(int i_axial=get_min_axial_pos_num(segment_num), axial_pos = seg_ax_offset[find_segment_index_in_sequence(segment_num)];
         i_axial<=get_max_axial_pos_num(segment_num);
         i_axial++, axial_pos++)
        for (int tof_poss = 0; tof_poss <= num_tof_poss-1; tof_poss++)
          {
            ret_viewgram[i_axial][-tang_pos] += static_cast<float> (tof_data[get_max_view_num()-view_num][i_tang][tof_poss][axial_pos]);
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
