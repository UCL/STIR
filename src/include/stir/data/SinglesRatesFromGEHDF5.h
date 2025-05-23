//
/*
    Copyright (C) 2017-2019, University of Leeds
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup singles_buildblock
  \ingroup GE
  \brief Declaration of class stir::GE::RDF_HDF5::SinglesRatesFromGEHDF5

  \author Palak Wadhwa
  \author Kris Thielemans

*/

#ifndef __stir_data_SinglesRatesFromGEHDF5_H__
#define __stir_data_SinglesRatesFromGEHDF5_H__

#include "stir/data/SinglesRatesForTimeSlices.h"
#include "stir/RegisteredParsingObject.h"


START_NAMESPACE_STIR
namespace GE {
namespace RDF_HDF5 {

class GEHDF5Wrapper;

/*!
  \ingroup singles_buildblock
  \ingroup GE
  \brief A class for reading singles over the number of time samples from an GE HDF5 .BLF listmode file format.

  .BLF files are generated as a result of PET scan by GE SIGNA PET/MR scanners.

  \todo expose GE::RDF_HDF5::GEHDF5Wrapper.get_exam_info_sptr()

*/
class SinglesRatesFromGEHDF5 : 
  public RegisteredParsingObject<SinglesRatesFromGEHDF5, SinglesRates, SinglesRatesForTimeSlices>

{ 
public:

 //! Name which will be used when parsing a SinglesRatesFromGEHDF5 object 
 static const char * const registered_name; 

 //! Default constructor 
 SinglesRatesFromGEHDF5();

 //! construct from filename
 explicit SinglesRatesFromGEHDF5(const std::string& rdf_filename)
 { read_from_file(rdf_filename); }

 void read_from_file(const std::string& rdf_filename);
 

private:

 shared_ptr<GEHDF5Wrapper> m_input_sptr;
 
 std::string _rdf_filename;

 virtual void set_defaults();
 virtual void initialise_keymap();
 virtual bool post_processing();
 
};

} // namespace
}
END_NAMESPACE_STIR


#endif
