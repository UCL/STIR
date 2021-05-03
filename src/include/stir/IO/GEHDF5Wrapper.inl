//
//
/*!

  \file
  \ingroup IO
  \ingroup GE
  \brief Declaration of class stir::GE::RDF_HDF5::GEHDF5Wrapper

  \author Nikos Efthimiou
  \author Palak Wadhwa
  \author Ander Biguri

*/
/*
    Copyright (C) 2017-2019, University of Leeds
    Copyright (C) 2018 University of Hull
    Copyright (C) 2018-2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

START_NAMESPACE_STIR
namespace GE {
namespace RDF_HDF5 {

shared_ptr<Scanner>
GEHDF5Wrapper::get_scanner_sptr() const
{
    return this->proj_data_info_sptr->get_scanner_sptr();
}

shared_ptr<const ProjDataInfo>
GEHDF5Wrapper::get_proj_data_info_sptr() const
{
  return this->proj_data_info_sptr;
}

shared_ptr<ExamInfo>
GEHDF5Wrapper::get_exam_info_sptr() const
{
    return this->exam_info_sptr;
}

H5::DataSet* GEHDF5Wrapper::get_dataset_ptr() const
{
    return m_dataset_sptr.get();
}

hsize_t GEHDF5Wrapper::get_dataset_size() const
{
    return m_list_size;
}

unsigned int GEHDF5Wrapper::get_geo_dims() const
{
    return geo_dims;
}

const H5::H5File& GEHDF5Wrapper::get_file() const
{ return file; }

} // namespace
}
END_NAMESPACE_STIR
