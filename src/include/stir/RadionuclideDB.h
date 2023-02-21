/*!

  \file
  \ingroup ancillary
  \brief Declaration of class stir::RadionuclideDB

  \author Daniel Deidda
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2021, National Physical Laboratory
    Copyright (C) 2021, 2022, University College London
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_RADIONUCLIDEDB_H
#define __stir_RADIONUCLIDEDB_H


#include "stir/Radionuclide.h"
#include "stir/ImagingModality.h"

#ifdef nlohmann_json_FOUND
#include <nlohmann/json.hpp>
#endif

/*!
  \ingroup ancillary
  \brief A class in that reads the radionuclide information from a Json file

  Values for half life branching ratio are taken from: 
  http://www.lnhb.fr/donnees-nucleaires/donnees-nucleaires-tableau/

  The class also supports using "aliases" for radionuclide names. This is useful
  as different manufacturers use different conventions.

  STIR comes with a default database and lookup table. The database uses DICOM
  conventions for the names. The lookup table provides alternatives for GE and Siemens
  (in STIR 5.0).

  \par Format of the radionuclide database

  This file is in JSON format. An example is distributed with STIR, a brief extract follows
  \verbatim
{
    "nuclides": [

        {
            "name":  "^18^Fluorine",
            "decays": [                
                {  "modality": "PET",
                   "keV": 511,
                   "branching_ratio": 0.9686,
                   "half_life": 6584.04
                }
            ]
        },
        {
            "name": "^99m^Technetium",
            "decays": [
                {  "modality": "nucmed",
                   "keV": 140.511,
                   "branching_ratio": 0.885,
                   "half_life": 21624.12
                }
            ]
            
        },
      # more entries like the above
   ]
}

\endverbatim

  \par Format of the radionuclide name look-up database

  This file is in JSON format. An example is distributed with STIR, a brief extract follows
  \verbatim
[
     [ "^18^Fluorine", "18F", "F-18" ],
     [ "^11^Carbon", "11C", "C-11" ]
]
  \endverbatim
  The first name is the one that will be used to find an entry in the radionuclide database.

  Note that both files need to have the same number of radionuclides..
*/

START_NAMESPACE_STIR

class RadionuclideDB
{
public:
  
  //! Default constructor
  /*! Reads the database from radionuclide_info.json and lookup table from radionuclide_names.json,
      with their locations found via find_STIR_config_file().

      If STIR is compiled without nlohmann_json support, this constructor does nothing.
  */
  RadionuclideDB();
  
  //! set the radionuclide database from a JSON file
  /*!
    This function could be used to override the default database information.

    If STIR is compiled without nlohmann_json support, this function calls error().
  */
  void read_from_file(const std::string& filename);

  //! Finds the radionuclide in the database
  /*!
    If \a rname is \c "default" or empty, use ^18F^Fluorine" for PET and "^99m^Technetium" for NM.
    Otherwise, first looks-up \a rname to translate to standard naming convention, then finds it in the database.
    If not found in the database, this function uses warning() and returns
    a default constructed Radionuclide() object (i.e. unknown).

    If STIR is compiled without nlohmann_json support, a hard-coded database is
    used (currently only containing values for the above default radionuclides).
  */
  Radionuclide get_radionuclide(ImagingModality rmodality, const std::string& rname);
  
protected:


private:
  std::string database_filename;
  std::string radionuclide_lookup_table_filename;
  
#ifdef nlohmann_json_FOUND
  nlohmann::json radionuclide_json;
#endif

  //! Finds the radionuclide info in the database
  /*!
    \a rname should have been standardised already.

    If not found in the database, this function uses warning() and returns
    a default constructed Radionuclide() object (i.e. unknown).
  */
  Radionuclide get_radionuclide_from_json(ImagingModality rmodality, const std::string& rname) const;
  
  //! Convert the radionuclide name using the lookup table
  /*!
    If \a rname is empty, return \c "default". Otherwise, attempt to find it in the lookup table and translate it.
    If not found (or if STIR is compiled without nlohmann_json support), return \a rname.
  */
  std::string get_radionuclide_name_from_lookup_table(const std::string& rname) const;
};

END_NAMESPACE_STIR
#endif // __stir_RADIONUCLIDEDB_H
