//
//
/*!

  \file
  \ingroup projdata

  \brief Declaration of class stir::ProjDataFromGEHDF5

  \author Palak Wadhwa
  \author Nikos Efthimiou
  \author Kris Thielemans


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    Copyright (C) 2018 University of Leeds
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
#ifndef __ProjDataFromGEHDF5_H__
#define __ProjDataFromGEHDF5_H__

#include "stir/ProjData.h"
#include "stir/NumericType.h"
#include "stir/ByteOrder.h"
#include "stir/IO/HDF5Wrapper.h"
#include "stir/Array.h"

#include <iostream>
#include <vector>

START_NAMESPACE_STIR


/*!
  \ingroup projdata
  \brief A class which reads projection data from a GE HDF5
  sinogram file.

  \warning support is still very basic.

*/
class ProjDataFromHDF5 : public ProjData, public HDF5Wrapper
{
public:
    
    static ProjDataFromHDF5* ask_parameters(const bool on_disk = true);
    
    explicit   ProjDataFromHDF5 (const shared_ptr<ExamInfo> &exam_info_sptr, const shared_ptr<ProjDataInfo> &proj_data_info_ptr, HDF5Wrapper* s);

//    explicit ProjDataFromHDF5 (const std::string& sinogram_filename);

//    explicit ProjDataFromHDF5 (const std::string& sinogram_filename);
  
    //! Get Viewgram<float> 
    Viewgram<float> get_viewgram(const int view_num, const int segment_num,const bool make_num_tangential_poss_odd=false) const;
    //! Set Viewgram<float>
    Succeeded set_viewgram(const Viewgram<float>& v);
    
    //! Get Sinogram<float> 
    Sinogram<float> get_sinogram(const int ax_pos_num, const int sergment_num,const bool make_num_tangential_poss_odd=false) const; 
    //! Set Sinogram<float>
    Succeeded set_sinogram(const Sinogram<float>& s);
 
    
private:
  //the file with the data
  //This has to be a reference (or pointer) to a stream, 
  //because assignment on streams is not defined;
  // TODO make shared_ptr
  std::iostream* sino_stream;
  //offset of the whole 3d sinogram in the stream
  std::streamoff  offset;
  
  const std::string _sinogram_filename;

  HDF5Wrapper* input;

  NumericType on_disk_data_type;
  
  ByteOrder on_disk_byte_order;

  int segment_offset;
  // view_scaling_factor is only used when reading data from file. Data are stored in
  // memory as float, with the scale factor multiplied out
 
  Array<1,float> buffer;

  Array<3, float> tof_data;
  
  std::vector<int> num_rings_orig;

  std::vector<int> segment_sequence_orig;


};

END_NAMESPACE_STIR


#endif
