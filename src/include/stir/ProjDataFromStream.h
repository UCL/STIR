//
//
/*!

  \file
  \ingroup projdata
  \brief Declaration of class stir::ProjDataFromStream

  \author Nikos Efthimiou
  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Claire Labbe
  \author PARAPET project


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2013, Hammersmith Imanet Ltd
    Copyright (C) 2016, University of Hull

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
#ifndef __ProjDataFromStream_H__
#define __ProjDataFromStream_H__

#include "stir/ProjData.h" 
#include "stir/NumericType.h"
#include "stir/ByteOrder.h"
#include "stir/shared_ptr.h"
#include "stir/Bin.h"
#include <iostream>
#include <vector>

START_NAMESPACE_STIR


/*!
  \ingroup projdata
  \brief A class which reads/writes projection data from/to a (binary) stream.

  At tehe end of every write (i.e., \ set_*) operation, the stream is flushed such that 
  subsequent read operations from the same file will be able this data even if the 
  stream isn't closed yet. This is important in an interactive context, as the object
  owning the stream might not be deleted yet before we try to read the file again.

  \warning Data have to be contiguous.
  \warning The parameter make_num_tangential_poss_odd (used in various 
  get_ functions) is temporary and will be removed soon.
  \warning Changing the sequence of the timing bins is not supported.
*/
class ProjDataFromStream : public ProjData
{
public:

  enum StorageOrder {
    Segment_AxialPos_View_TangPos,
      Segment_View_AxialPos_TangPos,
      Timing_Segment_View_AxialPos_TangPos,
    Unsupported };
#if 0    
  static  ProjDataFromStream* ask_parameters(const bool on_disk = true);
#endif

#if 0   
  //! Empty constructor
  ProjDataFromStream()
    {}
#endif    
    
  //! constructor taking all necessary parameters
  /*! 
    \param segment_sequence_in_stream has to be set according to the order
    in which the segments occur in the stream. segment_sequence_in_stream[i]
    is the segment number of the i-th segment in the stream.
  */
  ProjDataFromStream (shared_ptr<ExamInfo> const& exam_info_sptr,
		      shared_ptr<ProjDataInfo> const& proj_data_info_ptr,
		      shared_ptr<std::iostream> const& s, 
		      const std::streamoff offs, 
		      const std::vector<int>& segment_sequence_in_stream,
		      StorageOrder o = Segment_View_AxialPos_TangPos,
		      NumericType data_type = NumericType::FLOAT,
		      ByteOrder byte_order = ByteOrder::native,  
              float scale_factor = 1.f );

  //! as above, but with a default value for segment_sequence_in_stream
  /*! The default value for segment_sequence_in_stream is a vector with
    values min_segment_num, min_segment_num+1, ..., max_segment_num
  */
  ProjDataFromStream (shared_ptr<ExamInfo> const& exam_info_sptr,
              shared_ptr<ProjDataInfo> const& proj_data_info_ptr,
              shared_ptr<std::iostream> const& s,
              const std::streamoff offs = 0,
              StorageOrder o = Segment_View_AxialPos_TangPos,
              NumericType data_type = NumericType::FLOAT,
              ByteOrder byte_order = ByteOrder::native,
              float scale_factor = 1.f);
  //! Obtain the storage order
  inline StorageOrder get_storage_order() const;
    
  //! Get the offset -Changed into streamoff from int
  //inline int get_offset_in_stream() const;
  inline std::streamoff get_offset_in_stream() const;
    
  //! Get the data_type in the stream 
  inline NumericType get_data_type_in_stream() const;
    
  //! Get the byte order
  inline ByteOrder get_byte_order_in_stream() const;   
    
  //! Get the segment sequence
  inline std::vector<int> get_segment_sequence_in_stream() const;
  //! Get the timing bins sequence
  inline std::vector<int> get_timing_poss_sequence_in_stream() const;
    
  //! Get & set viewgram 
  Viewgram<float> get_viewgram(const int view_num, const int segment_num,
                               const bool make_num_tangential_poss_odd=false,
                               const int timing_pos=0) const;
  Succeeded set_viewgram(const Viewgram<float>& v);
    
  //! Get & set sinogram 
  Sinogram<float> get_sinogram(const int ax_pos_num, const int segment_num,
                               const bool make_num_tangential_poss_odd=false,
                               const int timing_pos=0) const;

  Succeeded set_sinogram(const Sinogram<float>& s);
    
  //! Get all sinograms for the given segment
  SegmentBySinogram<float> get_segment_by_sinogram(const int segment_num,
                                                   const int timing_num = 0) const;
  //! Get all viewgrams for the given segment
  SegmentByView<float> get_segment_by_view(const int segment_num,
                                           const int timing_pos = 0) const;
    
    
  //! Set all sinograms for the given segment
  Succeeded set_segment(const SegmentBySinogram<float>&);
  //! Set all viewgrams for the given segment
  Succeeded set_segment(const SegmentByView<float>&);

  //! Get scale factor
  float get_scale_factor() const;  

  //! Get the value of bin.
  float get_bin_value(const Bin& this_bin) const;
    
protected:
  //! the stream with the data
  shared_ptr<std::iostream> sino_stream;

private:

  void activate_TOF();
  //! offset of the whole 3d sinogram in the stream
  std::streamoff  offset;
  //! offset of a complete non-tof sinogram
  std::streamoff offset_3d_data;

  
  //!the order in which the segments occur in the stream
  std::vector<int> segment_sequence;
  //!the order in which the timing bins occur in the stream
  std::vector<int> timing_poss_sequence;
  
  inline int find_segment_index_in_sequence(const int segment_num) const;
  
  StorageOrder storage_order;
  
  NumericType on_disk_data_type;
  
  ByteOrder on_disk_byte_order;
  
  // scale_factor is only used when reading data from file. Data are stored in
  // memory as float, with the scale factor multiplied out
  float scale_factor;

  //! Calculate the offset of the give timing position
  //! \warning N.E: This function might be one the major components of STIR's speeds
  std::streamoff get_offset_timing(const int timing_num) const;
  
  //! Calculate the offset for the given segmnet
  //! \warning This function returns the offset of a segment *WITHING* a timing position
  //! If you like to get the offset of a segment from different timing positions it has to
  //! be combined with get_offset_timing().
  std::streamoff get_offset_segment(const int segment_num) const;
  
  //! Calculate offsets for viewgram data  
  std::vector<std::streamoff> get_offsets(const int view_num, const int segment_num,
                                          const int timing_num = 0) const;
  //! Calculate offsets for sinogram data
  std::vector<std::streamoff> get_offsets_sino(const int ax_pos_num, const int segment_num,
                                               const int timing_num = 0) const;
    
  //! Calculate the offsets for specific bins.
  std::vector<std::streamoff> get_offsets_bin(const int segment_num,
                                              const int ax_pos_num,
                                              const int view_num,
                                              const int tang_pos_num,
                                              const int timing_pos_num = 0) const;
  
};

END_NAMESPACE_STIR

#include "stir/ProjDataFromStream.inl"

#endif
