/*!

  \file
  \ingroup ancillary
  \brief Declaration of class stir::RadionuclideDB
    
  \author Daniel Deidda
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2021, National Physical Laboratory
    Copyright (C) 2021, University College London
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_RADIONUCLIDEDB_H
#define __stir_RADIONUCLIDEDB_H

/*!
  \brief A class in that reads the radionuclide information from a Json file

  Values for half life branching ratio are taken from: 
  http://www.lnhb.fr/donnees-nucleaires/donnees-nucleaires-tableau/

  \par Format of the radionuclide data base

  This file is in JSON format. An example is distributed with STIR.
  \verbatim
{   "nuclide": 

 {  "^18^Fluorine":

  {  "modality": 
    {   "PET":
      {  "properties": [
               {
                "kev": 511,
                "BRatio": 0.9686,
                "half_life": 6584.04
                }      ]
      }
    }
 },  

   "^99m^Technetium":

  {  "modality": 
    {   "NM":
      {  "properties": [
               {
                "kev": 140.511,
                "BRatio": 0.885,
                "half_life": 21624.12,
                }       ]
      }  
    }
  },
      # more entries like the above
   }
}

\endverbatim
*/

#include "stir/RegisteredParsingObject.h"
#include "stir/Radionuclide.h"
#include "stir/ImagingModality.h"

#ifdef nlohmann_json_FOUND
#include <nlohmann/json.hpp>
#endif

START_NAMESPACE_STIR

class RadionuclideDB
{
public:
//  static constexpr const char * const registered_name = "nuclideDB"; 
  
  //! Default constructor
  RadionuclideDB();
  
  //! set the JSON filename with the radionuclides
  void read_from_file(const std::string& filename);
  //! get the radionuclide 
  Radionuclide get_radionuclide(ImagingModality rmodality, const std::string& rname);
  
protected:


private:
  std::string filename;
  Radionuclide radionuclide;
  
//  std::string nuclide_name;
  std::string radionuclide_lookup_table_str;
  
#ifdef nlohmann_json_FOUND
nlohmann::json radionuclide_json;
#endif

//! the following extracts the radionuclide information from the JSON file and returns a radionculide object 
  Radionuclide get_radionuclide_from_json(ImagingModality rmodality, const std::string& rname) const;
  
//! the following looks at the radionuclide name in input and converts it if necessary to the Dicom format
  std::string get_radionuclide_name_from_lookup_table(const std::string& rname) const;
};

END_NAMESPACE_STIR
#endif // __stir_RADIONUCLIDEDB_H
