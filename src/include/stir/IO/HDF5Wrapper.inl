//
//
/*!

  \file
  \ingroup IO
  \brief

  \author Nikos Efthimiou


*/
/*
    Copyright (C) 2018 University of Hull
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

START_NAMESPACE_STIR

shared_ptr<Scanner>
HDF5Wrapper::get_scanner_sptr() const
{
    return this->scanner_sptr;
}

shared_ptr<ExamInfo>
HDF5Wrapper::get_exam_info_sptr() const
{
    return this->exam_info_sptr;
}

H5::DataSet* HDF5Wrapper::get_dataset_ptr() const
{
    return m_dataset_sptr.get();
}

hsize_t HDF5Wrapper::get_dataset_size() const
{
    return m_list_size;
}

TimeFrameDefinitions* HDF5Wrapper::get_timeframe_definitions() const
{
    //! \todo For examInfo get timeframe definitions
    return &exam_info_sptr->time_frame_definitions;
}

const H5::H5File& HDF5Wrapper::get_file() const
{ return file; }

END_NAMESPACE_STIR
