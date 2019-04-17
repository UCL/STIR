//
//
/*
    Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
    Copyright (C) 2018, University College London
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
#include <boost/format.hpp>
#include "stir/info.h"

START_NAMESPACE_STIR

const ProjDataInfo *
MultipleProjData::
get_proj_data_info_ptr() const
{
  if (get_num_gates() == 0)
    return 0;
  else
    return _proj_datas[0]->get_proj_data_info_ptr();
}

void 
MultipleProjData::
set_proj_data_sptr(const shared_ptr<ProjData >& proj_data_sptr, 
		 const unsigned int gate_num)
{
  this->_proj_datas[gate_num-1]=proj_data_sptr; 
}  

MultipleProjData::
MultipleProjData(const shared_ptr<ExamInfo>& exam_info_sptr,
                const int num_gates) :
    ExamData(exam_info_sptr)
{
    this->_proj_datas.resize(num_gates);
}

unique_ptr<MultipleProjData>
MultipleProjData::
read_from_file(const std::string &parameter_file)
{
   MultipleDataSetHeader header;

    if (header.parse(parameter_file.c_str()) == false)
        error(boost::format("MultipleProjData::read_from_file: Error parsing %1%") % parameter_file);

    int num_data_sets = header.get_num_data_sets();

    // Create the multiple proj data
    unique_ptr<MultipleProjData> multiple_proj_data( new MultipleProjData );

    // Read the projdata
    for (int i=0; i<num_data_sets; ++i) {
        info(boost::format("MultipleProjData::read_from_file: Reading %1%") % header.get_filename(i));
        // Create each of the individual proj datas
        multiple_proj_data->_proj_datas.push_back(ProjData::read_from_file(header.get_filename(i)));
    }

    // Get the exam info (from the first ProjData)
    multiple_proj_data->set_exam_info(multiple_proj_data->_proj_datas[0]->get_exam_info());

    // Update the time definitions based on each individual frame
    multiple_proj_data->get_exam_info_sptr()->time_frame_definitions.set_num_time_frames(num_data_sets);
    for (int i=0; i<num_data_sets; ++i) {
        const TimeFrameDefinitions &tdef = multiple_proj_data->_proj_datas[i]->get_exam_info_ptr()->time_frame_definitions;
        multiple_proj_data->get_exam_info_sptr()->time_frame_definitions.set_time_frame(i+1, tdef.get_start_time(1), tdef.get_end_time(1));
    }

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
