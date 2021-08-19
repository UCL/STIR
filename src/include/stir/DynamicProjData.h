/*
    Copyright (C) 2005 - 2011-01-04, Hammersmith Imanet Ltd
    Copyright (C) 2013, Kris Thielemans
    Copyright (C) 2013, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir__DynamicProjData__H__
#define __stir__DynamicProjData__H__
/*!
  \file
  \ingroup data_buildblock
  \brief Declaration of class stir::DynamicProjData
  \author Kris Thielemans
*/
#include "stir/MultipleProjData.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/SegmentByView.h"
#include "stir/Succeeded.h"
#include <string>

START_NAMESPACE_STIR

class Succeeded;
//! Dynamic projection data
/*! \ingroup buildblock
  Somewhat preliminary
 
  \todo Move read_from_file, write_to_ecat7 to usual registry methods
  \todo Interfile support currently doesn't set start_time_in_secs_since_1970
*/
class DynamicProjData :
 public MultipleProjData
{
public:
  static
  unique_ptr<DynamicProjData>
    read_from_file(const std::string& filename);

  DynamicProjData() {}

  DynamicProjData(const MultipleProjData& m):
    MultipleProjData(m)
  {}

  DynamicProjData(const shared_ptr<const ExamInfo>& exam_info_sptr)
    : MultipleProjData(exam_info_sptr)
  {
  }

  DynamicProjData(const shared_ptr<const ExamInfo>& exam_info_sptr,
                    const int num_gates)
        : MultipleProjData(exam_info_sptr,
                    num_gates)
  {}

  //! Return time of start of scan
  /*! \return the time in seconds since 1 Jan 1970 00:00 UTC, i.e. independent
        of your local time zone.

      Note that the return type is a \c double. This allows for enough accuracy
      for a long time to come. It also means that the start time can have fractional 
      seconds.

      The time frame definitions should be relative to this time.
  */
  const double get_start_time_in_secs_since_1970() const;

  //! set start of scan
  /*! \see get_start_time_in_secs_since_1970()
   */
  void set_start_time_in_secs_since_1970(const double start_time);  
  unsigned int get_num_frames() const
  {
    return this->get_num_proj_data();
  }

  Succeeded   
    write_to_ecat7(const std::string& filename) const;

  void set_time_frame_definitions(const TimeFrameDefinitions& time_frame_definitions);

  const TimeFrameDefinitions& get_time_frame_definitions() const;

  //! multiply data with a constant factor
  /*! \warning for most types of data, this will modify the data on disk */
  void
    calibrate_frames(const float cal_factor)
    {
      for (  unsigned int frame_num = 1 ; frame_num<=this->get_time_frame_definitions().get_num_frames() ;  ++frame_num ) 
	for (int segment_num = (this->_proj_datas[frame_num-1])->get_min_segment_num();
	     segment_num <= (this->_proj_datas[frame_num-1])->get_max_segment_num();	   ++segment_num)
	  {   
	    SegmentByView<float> segment_by_view
	      ((*(this->_proj_datas[frame_num-1])).get_segment_by_view(segment_num));
	    segment_by_view *= cal_factor;
	    if ((*(this->_proj_datas[frame_num-1])).set_segment(segment_by_view)
		==Succeeded::no)
	      {
		error("DynamicProjData:calibrate_frames failed because set_segment_by_view failed");
	      }
	}
    }

  //! divide data with the corresponding frame duration
  /*! \warning for most types of data, this will modify the data on disk */
  void
    divide_with_duration()
    {   
  // do reading/writing in a loop over segments
      for(unsigned int frame_num=1;frame_num<=this->get_time_frame_definitions().get_num_frames();++frame_num)
	for (int segment_num = (this->_proj_datas[frame_num-1])->get_min_segment_num();
	     segment_num <= (this->_proj_datas[frame_num-1])->get_max_segment_num();	   ++segment_num)
	  {   
	    SegmentByView<float> segment_by_view = 
	      (*(this->_proj_datas[frame_num-1])).get_segment_by_view(segment_num);
	    segment_by_view /= static_cast<float>(this->get_time_frame_definitions().get_duration(frame_num));
            if ((*(this->_proj_datas[frame_num-1])).set_segment(segment_by_view) 
                ==Succeeded::no) 
              { 
                error("DynamicProjData:calibrate_frames failed because set_segment_by_view failed"); 
              } 

	}
    }
};

//! Copy all bins to a range specified by an iterator
/*! 
  \ingroup copy_fill
  \return \a iter advanced over the range (as std::copy)
  
  \warning there is no range-check on \a iter
*/

template<>
struct CopyFill<DynamicProjData>
{ template < typename iterT>
    static
    iterT copy_to(const DynamicProjData& stir_object, iterT iter)
{
  //std::cerr<<"Using DynamicProjData::copy_to\n";
  return stir_object.copy_to(iter);
}
};

//! set all elements of a MultipleProjData  from an iterator
/*!  
  \ingroup copy_fill
  Implementation that resorts to MultipleProjData::fill_from
  \warning there is no size/range-check on \a iter
*/
template < typename iterT>
void fill_from(DynamicProjData& stir_object, iterT iter, iterT /*iter_end*/)
{
  return stir_object.fill_from(iter);
}

END_NAMESPACE_STIR
#endif
