/*
    Copyright (C) 2016 University College London
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
#ifndef __stir_IO_HDF5Wrapper_H__
#define __stir_IO_HDF5Wrapper_H__
/*!
  \file
  \ingroup
  \brief

*/

#include "stir/shared_ptr.h"
#include "stir/ExamInfo.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"
#include "stir/Array.h"

#include "H5Cpp.h"

#include <string>

START_NAMESPACE_STIR

class HDF5Wrapper
{
public:

    static bool check_GE_signature(const std::string filename);

    explicit HDF5Wrapper();

    explicit HDF5Wrapper(const std::string& filename);

    Succeeded open(const std::string& filename);

    Succeeded initialise_listmode_data(const std::string& path = "");

    Succeeded initialise_singles_data(const std::string& path = "");

    Succeeded initialise_proj_data_data(const std::string& path = "",
                                        const unsigned int view_num = 0);
    //PW Here I added the geo_factors_data_initialisation. This should initialise the factors for
    // specific path and slice_num.
    Succeeded initialise_geo_factors_data(const std::string& path = "",
                                          const unsigned int slice_num=0);

    Succeeded get_from_dataspace(std::streampos &current_offset,
                                 shared_ptr<char>& output);

    Succeeded get_dataspace(const unsigned int current_id,
                            Array<1, unsigned int> &output);

    Succeeded get_dataspace(const unsigned int current_id,
                            shared_ptr<Array<2, unsigned int> > &output);

    Succeeded get_from_dataset(const std::array<unsigned long long, 3> &offset,
                                 const std::array<unsigned long long, 3> &count,
                                 const std::array<unsigned long long, 3> &stride,
                                 const std::array<unsigned long long, 3> &block,
                                 Array<1, unsigned char> &output);

    //PW Here I added the get_from_2d_dataset which must read the hyperslab and memory space for 2D array
    // with specific offset, count, stride and block. This dataset that is read from this memory space,
    // is then read into a 1D output array.
    Succeeded get_from_2d_dataset(const std::array<unsigned long long, 2> &offset,
                                 const std::array<unsigned long long, 2> &count,
                                 const std::array<unsigned long long, 2> &stride,
                                 const std::array<unsigned long long, 2> &block,
                                 Array<1, unsigned int> &output);

    inline H5::DataSet* get_dataset_ptr() const;

    inline hsize_t get_dataset_size() const;

    inline TimeFrameDefinitions* get_timeframe_definitions() const;

    //! Get shared pointer to exam info
    /*! \warning Use with care. If you modify the object in a shared ptr, everything using the same
      shared pointer will be affected. */
    inline shared_ptr<ExamInfo>
    get_exam_info_sptr() const;

    inline shared_ptr<Scanner>
    get_scanner_sptr() const;

    //! \obsolete
    inline const H5::H5File& get_file() const;

    ~HDF5Wrapper() {}

protected:

    Succeeded initialise_scanner_from_HDF5();

    Succeeded initialise_exam_info();

    shared_ptr<Scanner> scanner_sptr;

    shared_ptr<ExamInfo> exam_info_sptr;

private:

    H5::H5File file;

    shared_ptr<H5::DataSet> m_dataset_sptr;

    H5::DataSpace m_dataspace;

    H5::DataSpace* m_memspace_ptr;

    uint64_t m_list_size = 0;

    //    shared_ptr<H5::DataSet> dataset_norm_sptr;

    //    shared_ptr<H5::DataSet> dataset_projdata_sptr;

    //    shared_ptr<H5::DataSet> dataset_singles_sptr;

    //    int dataset_singles_Ndims = 0;

    //    int dataset_projdata_Ndims = 0;

    //    int dataset_norm_Ndims = 0;

    std::string m_address;

    bool is_signa = false;

    hsize_t m_size_of_record_signature = 0;

    hsize_t m_max_size_of_record = 0;

    int m_NX_SUB = 0;    // hyperslab dimensions
    int m_NY_SUB = 0;
    int m_NZ_SUB = 0;
    int m_NX = 0;        // output buffer dimensions
    int m_NY = 0;
    int m_NZ = 0;

};

END_NAMESPACE_STIR

#include "stir/IO/HDF5Wrapper.inl"

#endif
