//
//
/*
    Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
    Copyright (C) 2018, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup data_buildblock
  \brief Implementation of class stir::MultipleProjData
  \author Kris Thielemans
  \author Richard Brown

*/

#include "stir/MultipleProjData.h"
#include "stir/is_null_ptr.h"
#include "stir/KeyParser.h"
#include "stir/MultipleDataSetHeader.h"
#include <fstream>
#include "stir/format.h"
#include "stir/info.h"
#include "stir/error.h"

START_NAMESPACE_STIR

const shared_ptr<const ProjDataInfo>
MultipleProjData::get_proj_data_info_sptr() const
{
  if (get_num_gates() == 0)
    return shared_ptr<const ProjDataInfo>();
  else
    return _proj_datas[0]->get_proj_data_info_sptr();
}

void
MultipleProjData::set_proj_data_sptr(const shared_ptr<ProjData>& proj_data_sptr, const unsigned int gate_num)
{
  this->_proj_datas[gate_num - 1] = proj_data_sptr;
}

MultipleProjData::MultipleProjData(const shared_ptr<const ExamInfo>& exam_info_sptr, const int num_gates)
    : ExamData(exam_info_sptr)
{
  this->_proj_datas.resize(num_gates);
}

unique_ptr<MultipleProjData>
MultipleProjData::read_from_file(const std::string& parameter_file)
{
  MultipleDataSetHeader header;

  if (header.parse(parameter_file.c_str()) == false)
    error(format("MultipleProjData::read_from_file: Error parsing {}", parameter_file));

  const int num_data_sets = header.get_num_data_sets();

  if (num_data_sets == 0)
    error("MultipleProjData::read_from_file: no data sets provided");

  // Create the multiple proj data
  unique_ptr<MultipleProjData> multiple_proj_data(new MultipleProjData);

  // Read the projdata
  for (int i = 0; i < num_data_sets; ++i)
    {
      info(format("MultipleProjData::read_from_file: Reading {}", header.get_filename(i)));
      // Create each of the individual proj datas
      multiple_proj_data->_proj_datas.push_back(ProjData::read_from_file(header.get_filename(i)));
    }

  // Get the exam info (from the first ProjData)
  ExamInfo exam_info = multiple_proj_data->_proj_datas.at(0)->get_exam_info();

  // Update the time definitions based on each individual frame
  exam_info.time_frame_definitions.set_num_time_frames(num_data_sets);
  for (int i = 0; i < num_data_sets; ++i)
    {
      const TimeFrameDefinitions& tdef = multiple_proj_data->_proj_datas[i]->get_exam_info().time_frame_definitions;
      exam_info.time_frame_definitions.set_time_frame(i + 1, tdef.get_start_time(1), tdef.get_end_time(1));
    }
  multiple_proj_data->set_exam_info(exam_info);
  return multiple_proj_data;
}

#if 0
// currently inline
const ProjData & 
MultipleProjData::
get_proj_data(const unsigned int gate_num) const 
{
  return *this->_proj_datas[gate_num-1] ; 
}
#endif

END_NAMESPACE_STIR
