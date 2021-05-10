/*  Copyright (C) 2017-2019, University of Leeds
    Copyright (C) 2018 University of Hull
    Copyright (C) 2018-2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
#include "stir/ProjDataInfo.h"
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

    //! reads coincidence time window from file (in secs)
    float get_coincidence_time_window() const;

    //! reads listmode event(s)
    /* \param[output] output: has to be pre-allocated and of the correct size
       \param[in] offset: start in listmode data (in number of bytes)
       \param[in]: size to read (in number of bytes)
    */
    Succeeded read_list_data(char* output,
                             const std::streampos offset,
                             const hsize_t size) const;

    //! read singles at time slice \c current_id
    /*! \param[in] current_id is a 1-based index */
    Succeeded read_singles(Array<1, unsigned int> &output,
                          const unsigned int current_id);

    Succeeded read_sinogram(Array<3, unsigned char> &output, 
                               const std::array<hsize_t, 3> &offset={0,0,0},
                               const std::array<hsize_t, 3> &stride={1,1,1});

    //PW Here I added the get_from_2d_dataset which must read the hyperslab and memory space for 2D array
    // with specific offset, count, stride and block. This dataset is read from this memory space and then
    // into a 1D output array.
    Succeeded read_geometric_factors(Array<1, unsigned int> &output,
                                  const std::array<hsize_t, 2>& offset={0,0},
                                  const std::array<hsize_t, 2>& count ={0,0},
                                  const std::array<hsize_t, 2>& stride={1,1});

    Succeeded read_efficiency_factors(Array<1,float> &output,
                                  const std::array<hsize_t, 2>& offset={0,0},
                                  const std::array<hsize_t, 2>& stride={1,1});

    inline H5::DataSet* get_dataset_ptr() const;

    inline hsize_t get_dataset_size() const;

    inline unsigned int get_geo_dims() const;

    unsigned int get_num_singles_samples();
 //   inline TimeFrameDefinitions* get_timeframe_definitions() const;

    //! Get shared pointer to exam info
    /*! \warning Use with care. If you modify the object in a shared ptr, everything using the same
      shared pointer will be affected. */
    inline shared_ptr<ExamInfo>
    get_exam_info_sptr() const;

    inline shared_ptr<Scanner>
    get_scanner_sptr() const;

    inline shared_ptr<const ProjDataInfo>
    get_proj_data_info_sptr() const;

    //! \obsolete
    inline const H5::H5File& get_file() const;

    ~GEHDF5Wrapper() {}


    std::uint32_t read_dataset_uint32(const std::string& dataset_name);
    std::int32_t read_dataset_int32(const std::string& dataset_name);
    
protected:
    //! enum for encoding head/feet first in the RDF file
    enum AcqPatientEntries
    { ACQ_HEAD_FIRST=0, ACQ_FEET_FIRST=1};
    //! enum for encoding patient orientation in the RDF file
    enum AcqPatientPositions
    { ACQ_SUPINE=0, ACQ_PRONE=1, ACQ_LEFT_DECUB=2, ACQ_RIGHT_DECUB=3};

    void initialise_proj_data_info_from_HDF5();
    void initialise_exam_info();

    shared_ptr<ProjDataInfo> proj_data_info_sptr;
    shared_ptr<ExamInfo> exam_info_sptr;

private:

    Succeeded check_file(); 

    unsigned int check_geo_type();
    shared_ptr<Scanner> get_scanner_from_HDF5();

    H5::H5File file;

    shared_ptr<H5::DataSet> m_dataset_sptr;

    H5::DataSpace m_dataspace;

    std::uint64_t m_list_size = 0;
    int m_dataset_list_Ndims;
    unsigned int m_num_singles_samples;

    bool is_list = false;
    bool is_sino = false;
    bool is_geo  = false;
    bool is_norm = false;

    // There are two types of geometry that behave very differently. We need to know which type it is.
    // In essence, the options are 2D (2) or 3D (3). In 2D, all transaxial values are the same. 
    unsigned int geo_dims = 0;
    std::string m_address;

    unsigned int  rdf_ver = 0;

    hsize_t m_size_of_record_signature = 0;

    hsize_t m_max_size_of_record = 0;

    hsize_t m_NX_SUB = 0;    // hyperslab dimensions
    hsize_t m_NY_SUB = 0;
    hsize_t m_NZ_SUB = 0;

    static const int m_max_dataset_dims =5;
#if 0
    // AB: todo these are never used. 
    hsize_t m_NX = 0;        // output buffer dimensions
    hsize_t m_NY = 0;
    hsize_t m_NZ = 0;
#endif
};

} // namespace
}
END_NAMESPACE_STIR

#include "stir/IO/GEHDF5Wrapper.inl"

#endif
