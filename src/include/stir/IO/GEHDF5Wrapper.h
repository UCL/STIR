/*  Copyright (C) 2017-2019, University of Leeds
    Copyright (C) 2018 University of Hull
    Copyright (C) 2018-2020, University College London
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
#ifndef __stir_IO_GEHDF5Wrapper_H__
#define __stir_IO_GEHDF5Wrapper_H__
/*!
  \file
  \ingroup IO
  \ingroup GE
  \brief Declaration of class stir::GE::RDF_HDF5::GEHDF5Wrapper

  \author Nikos Efthimiou
  \author Palak Wadhwa
  \author Ander Biguri
  \author Kris Thielemans
*/

#include "stir/shared_ptr.h"
#include "stir/ExamInfo.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"
#include "stir/Array.h"

#include "H5Cpp.h"

#include <string>

START_NAMESPACE_STIR

namespace GE {

namespace RDF_HDF5 {

/*!
  \ingroup IO
  \ingroup GE
  \brief A class that extracts and reads the necessary data from GE HDF5 input files.
*/

class GEHDF5Wrapper
{
public:

    //! check signature to see if this is a GE RDF9 or higher file
    /*! calls current_GE_signature(H5::H5file&) */
    static bool check_GE_signature(const std::string& filename);

    //! check signature to see if this is a GE RDF9 or higher HDF5 file
    /*! does minimal checks to see if expected datasets are present in the HDF5 object */
    static bool check_GE_signature(H5::H5File& file);

    GEHDF5Wrapper();

    explicit GEHDF5Wrapper(const std::string& filename);


    // bool is_list_file(const std::string& filename);

    //! Checks if input file is a listmode file
    bool is_list_file() const;

    bool is_sino_file() const;

    bool is_geo_file() const;

    bool is_norm_file() const;

    Succeeded open(const std::string& filename);

    Succeeded initialise_listmode_data();

    Succeeded initialise_singles_data();

    //! Initialises data for reading projections. Sets up reading addresses and inspect sizes.
    /*! \param view_num uses 1-based indexing

       this function has to be called for each projection/view_num
    */
    Succeeded initialise_proj_data(const unsigned int view_num);
    //! Initialises data for reading geometric normalisation factors. Sets up reading addresses and inspect sizes.
    /*! \param slice_num uses 1-based indexing
     */
    Succeeded initialise_geo_factors_data(const unsigned int slice_num);

    Succeeded initialise_efficiency_factors();

    //! reads a listmode event
    /* \param output: has to be pre-allocated and of the correct size (\c size_of_record_signature)
       \param current_offset will be incremented
    */
    Succeeded get_list_data(char* output,
                            std::streampos& current_offset);

    //! read singles at time slice \c current_id
    /*! \param current)id is 1-based index */
    Succeeded get_singles(Array<1, unsigned int> &output,
                          const unsigned int current_id);

    Succeeded get_from_dataset(Array<3, unsigned char> &output, 
                               const std::array<unsigned long long, 3> &offset={0,0,0},
                               const std::array<unsigned long long, 3> &stride={1,1,1});

    //PW Here I added the get_from_2d_dataset which must read the hyperslab and memory space for 2D array
    // with specific offset, count, stride and block. This dataset is read from this memory space and then
    // into a 1D output array.
    Succeeded get_from_2d_dataset(Array<1, unsigned int> &output,
                                             const std::array<unsigned long long int, 2>& offset={0,0},
                                             const std::array<unsigned long long int, 2>& stride={1,1});

    Succeeded get_from_2d_dataset(Array<1,float> &output,
                                             const std::array<unsigned long long int, 2>& offset={0,0},
                                             const std::array<unsigned long long int, 2>& stride={1,1});

    inline H5::DataSet* get_dataset_ptr() const;

    inline hsize_t get_dataset_size() const;

    unsigned int get_num_singles_samples();
 //   inline TimeFrameDefinitions* get_timeframe_definitions() const;

    //! Get shared pointer to exam info
    /*! \warning Use with care. If you modify the object in a shared ptr, everything using the same
      shared pointer will be affected. */
    inline shared_ptr<ExamInfo>
    get_exam_info_sptr() const;

    inline shared_ptr<Scanner>
    get_scanner_sptr() const;

    //! \obsolete
    inline const H5::H5File& get_file() const;

    ~GEHDF5Wrapper() {}

protected:

    Succeeded initialise_scanner_from_HDF5();

    Succeeded initialise_exam_info();

    shared_ptr<Scanner> scanner_sptr;

    shared_ptr<ExamInfo> exam_info_sptr;

private:

    Succeeded check_file(); 

    H5::H5File file;

    shared_ptr<H5::DataSet> m_dataset_sptr;

    H5::DataSpace m_dataspace;

    H5::DataSpace* m_memspace_ptr;

    uint64_t m_list_size = 0;

    unsigned int m_num_singles_samples;

    bool is_list = false;
    bool is_sino = false;
    bool is_geo  = false;
    bool is_norm = false;

    std::string m_address;

    unsigned int  rdf_ver = 0;

    hsize_t m_size_of_record_signature = 0;

    hsize_t m_max_size_of_record = 0;

    int m_NX_SUB = 0;    // hyperslab dimensions
    int m_NY_SUB = 0;
    int m_NZ_SUB = 0;
    // AB: todo these are never used. 
    int m_NX = 0;        // output buffer dimensions
    int m_NY = 0;
    int m_NZ = 0;

};

} // namespace
}
END_NAMESPACE_STIR

#include "stir/IO/GEHDF5Wrapper.inl"

#endif
