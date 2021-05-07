/*!

  \file
  \ingroup projdata
  \brief Declaration of class stir::RadionuclideDBProcessor
    
  \author Daniel Deidda
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2021, NPL
    Copyright (C) 2021, UCL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_RADIONUCLIDEDBPROCESSOR_H
#define __stir_RADIONUCLIDEDBPROCESSOR_H

/*!
  \brief A class in that reads the radionuclide information from a Json file

  Values for half life branching ratio are taken from: 
  http://www.lnhb.fr/donnees-nucleaires/donnees-nucleaires-tableau/

  \par Format of the radionuclide data base

  This file is in JSON format. An example is distributed with STIR.
  \verbatim
{     "nuclide": 

  {  "modality": 
    {   :"PET"
      {  "properties": [
               {
                "kev": 511,
                "BRatio": 0.9686,
                "half_life": 6584.04,
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

START_NAMESPACE_STIR

class RadionuclideDBProcessor//: public Radionuclide
{
public:
//  static constexpr const char * const registered_name = "nuclideDB"; 
  
  //! Default constructor
  RadionuclideDBProcessor();
  
  //! constructor based on information that we already have from the interfile
  RadionuclideDBProcessor(ImagingModality modality, std::string nuclide_name);

  //! set the JSON filename with the radionuclides
  void set_DB_filename(const std::string& filename);
  //! get the radionuclide 
  Radionuclide get_radionuclide();
  
protected:


private:
  std::string filename;
  Radionuclide radionuclide;
  
  std::string nuclide_name;
  std::string isotope_lookup_table_str;
  float energy;
  float branching_ratio;
  float half_life;
  std::string modality_str;
  ImagingModality modality;

  void get_record_from_json();
  
  void get_isotope_name_from_lookup_table();
};

END_NAMESPACE_STIR
#endif // __stir_RADIONUCLIDEDBPROCESSOR_H
