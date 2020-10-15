//
//
/*!

  \file
  \ingroup projdata
  \ingroup GE
  \brief Declaration of class stir::GE::RDF_HDF5::ProjDataFromGEHDF5

  \author Palak Wadhwa
  \author Nikos Efthimiou
  \author Kris Thielemans


*/
/*  Copyright (C) 2018-2019 University College London
    Copyright (C) 2018-2019 University of Leeds
    Copyright (C) 2018-2019 University of Hull
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
#ifndef __ProjDataGEDF5_H__
#define __ProjDataGEDF5_H__

#include "stir/ProjData.h"
#include "stir/IO/GEHDF5Wrapper.h"
#include "stir/Array.h"

START_NAMESPACE_STIR

namespace GE {
namespace RDF_HDF5 {


/*!
  \ingroup projdata
  \ingroup GE
  \brief A class which reads projection data from a GE HDF5
  sinogram file.
*/
class ProjDataGEHDF5 : public ProjData
{
public:

    static ProjDataGEHDF5* ask_parameters(const bool on_disk = true);

    explicit ProjDataGEHDF5(shared_ptr<ProjDataInfo> input_proj_data_info_sptr,
                              const std::string& input_filename);

    explicit ProjDataGEHDF5(shared_ptr<ProjDataInfo> input_proj_data_info_sptr,
                              shared_ptr<GEHDF5Wrapper> input_hdf5_sptr);

    explicit ProjDataGEHDF5
      (shared_ptr<ExamInfo> input_exam_info_sptr,
                              shared_ptr<ProjDataInfo> input_proj_data_info_sptr,
                              shared_ptr<GEHDF5Wrapper> input_hdf5);

private:
    unsigned int find_segment_offset(const int segment_num) const;
    //! Set Viewgram<float>
    Succeeded set_viewgram(const Viewgram<float>& v);
    //! Set Sinogram<float>
    Succeeded set_sinogram(const Sinogram<float>& s);
    //! Get Viewgram<float>
    Viewgram<float> get_viewgram(const int view_num, const int segment_num,const bool make_num_tangential_poss_odd=false) const;
    //! Get Sinogram<float>
    Sinogram<float> get_sinogram(const int ax_pos_num, const int sergment_num,const bool make_num_tangential_poss_odd=false) const;
    //! Get the segment sequence
    std::vector<int> get_segment_sequence_in_hdf5() const;
        std::vector< unsigned int > seg_ax_offset;
        unsigned int find_segment_index_in_sequence(const int segment_num) const;
    //! Cache the segment sequence of the GE data.
    //! \author Kris Thielemans
    void initialise_segment_sequence();

    void initialise_ax_pos_offset();

    void initialise_viewgram_buffer();
    //! Handler of the HDF5 input data and header
    shared_ptr<GEHDF5Wrapper> m_input_hdf5_sptr;

    std::vector< int > segment_sequence;
    Array<4, unsigned char> tof_data;
};

} // namespace
}
END_NAMESPACE_STIR


#endif
