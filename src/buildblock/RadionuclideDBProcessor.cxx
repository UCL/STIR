/*!

  \file
  \ingroup projdata
  \brief Implementation of class stir::RadionuclideDBProcessor
  
  \author Daniel Deidda
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2021, NPL
    Copyright (C) 2021, UCL
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

#include "stir/RadionuclideDBProcessor.h"
#include "stir/info.h"
#include "stir/round.h"
#include "stir/error.h"
//#include "stir/ImagingModality.h"
#include <nlohmann/json.hpp>
#include "stir/findSTIRConfig.h"

START_NAMESPACE_STIR

RadionuclideDBProcessor::
RadionuclideDBProcessor()
{
}

RadionuclideDBProcessor::
RadionuclideDBProcessor(ImagingModality rmodality, std::string rname)
:nuclide_name(rname),
 modality(rmodality)
{
    set_DB_filename(find_STIR_config_file("radionuclide_info.json"));
    modality_str=modality.get_name();
}

void
RadionuclideDBProcessor::
set_DB_filename(const std::string& arg)
{
  this->filename = arg;
}

void
RadionuclideDBProcessor::
get_record_from_json()
{
  if (this->filename.empty())
    error("RadionuclideDB: no filename set for the Radionuclide info");
  if (this->modality_str.empty())
    error("RadionuclideDB: no modality set for the Radionuclide info");
  if (this->nuclide_name.empty())
    error("RadionuclideDB: no nuclide set for the Radionuclide info");

  //Read Radionuclide file
  
  std::string s =this->filename;
  std::ifstream json_file_stream(s);
  
  if(!json_file_stream)
      error("Could not open Json file!");
//  std::string out;
  
//  while(json_file_stream >> out)
//      std::cout << out;
  
//  std::cout<<json_file_stream.rdbuf();
//  json_file_stream.close();
  nlohmann::json radionuclide_json;
  json_file_stream >> radionuclide_json;
//  radionuclide_json.parse(json_file_stream);
//  std::cout<< radionuclide_json;
  
  if (radionuclide_json.find("nuclide") == radionuclide_json.end())
  {
    error("RadionuclideDB: No or incorrect JSON radionuclide set (could not find \"nuclide\" in file \""
          + filename + "\")");
  }
  

  std::string name = this->nuclide_name;

  //Get desired keV as float value
  float  keV = this->energy;
  //Get desired kVp  as float value
  float  h_life = this->half_life;
  info("RadionuclideDB: finding record radionuclide:" + nuclide_name+
       "in file "+ filename);

  //Extract appropriate chunk of JSON file for given nuclide.
  nlohmann::json target = radionuclide_json["nuclide"][name]["modality"][modality_str]["properties"];
//[name]["modality"][modality_str]["properties"]
  int location = -1;
  int pos = 0;
  for (auto entry : target){
      location = pos;
    pos++;
  }

  if (location == -1){
    error("RadionuclideDB: Desired radionuclide not found!");
  }

  //Extract properties for specific nuclide and modality.
  nlohmann::json properties = target[location];
  {
    std::stringstream str;
    str << properties.dump(6);
    info("RadionuclideDB: JSON record found:" + str.str(),2);
  }
  this->energy = properties["kev"];
  this->branching_ratio = properties["BRatio"];
  this->half_life = properties["half_life"];
  
  //  this->breakPoint = properties["break"];
  
//Set Radionuclide member
  Radionuclide rnuclide(nuclide_name,
                        energy,
                        branching_ratio,
                        half_life,
                        modality);
  
  this->radionuclide=rnuclide;

}

Radionuclide 
RadionuclideDBProcessor::
get_radionuclide(){
    get_record_from_json();
    return this->radionuclide;
}

END_NAMESPACE_STIR


