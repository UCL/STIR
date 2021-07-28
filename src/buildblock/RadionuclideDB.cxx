/*!

  \file
  \ingroup projdata
  \brief Implementation of class stir::RadionuclideDB
  
  \author Daniel Deidda
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2021, National Physical Laboratory
    Copyright (C) 2021, University Colleg London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/RadionuclideDB.h"
#include "stir/info.h"
#include "stir/round.h"
#include "stir/error.h"
#include "stir/find_STIR_config.h"

START_NAMESPACE_STIR

RadionuclideDB::
RadionuclideDB()
{
    read_from_file(find_STIR_config_file("radionuclide_info.json"));
    this->radionuclide_lookup_table_str=find_STIR_config_file("radionuclide_names.json");
}

void
RadionuclideDB::
read_from_file(const std::string& arg)
{
    #ifdef nlohmann_json_FOUND
    this->filename = arg;
    
//    modality_str=modality.get_name();
    
    //Read Radionuclide file and set JSON member for DB
    
    std::string s =this->filename;
    std::ifstream json_file_stream(s);
    
    if(!json_file_stream)
        error("Could not open Json file!");
  
  //  nlohmann::json radionuclide_json;
    json_file_stream >> this->radionuclide_json;
  //  radionuclide_json.parse(json_file_stream);
  //  
    
    if (radionuclide_json.find("nuclide") == radionuclide_json.end())
    {
      error("RadionuclideDB: No or incorrect JSON radionuclide set (could not find \"nuclide\" in file \""
            + filename + "\")");
    }
#endif
}

std::string 
RadionuclideDB::
get_radionuclide_name_from_lookup_table(const std::string& rname) const
{
    #ifdef nlohmann_json_FOUND
    if (this->radionuclide_lookup_table_str.empty())
        error("Lookup table: no filename set");
    if (this->filename.empty())
      error("RadionuclideDB: no filename set for the Radionuclide info");
    
    
    std::string s =this->radionuclide_lookup_table_str;
    std::ifstream json_file_stream(s);
    
    if(!json_file_stream)
        error("Could not open Json file!");
    
    nlohmann::json table_json;
    json_file_stream >> table_json;
    
//    Check that lookup table and database have the same numbe of elements
    if (radionuclide_json["nuclide"].size() != table_json.size())
        error("The lookup table and the radionuclide database do not have the same number of elements. " 
              "If you added a radionuclide you also need to add the same in the lookup table");
    
    for (int l=0; l<table_json.size(); l++)
        for (int c=0; c<table_json.at(0).size(); c++)
        {
            if(table_json.at(l).at(c)==rname)
            return table_json.at(l).at(0);
        }
#else
    if (rname.empty())
        return "default";
    else
        return rname;
#endif
}

Radionuclide
RadionuclideDB::
get_radionuclide_from_json(ImagingModality rmodality, const std::string &rname) const
{
//    std::string nuclide_name=get_radionuclide_name_from_lookup_table(rname);
    
  if (this->filename.empty())
    error("RadionuclideDB: no filename set for the Radionuclide info");
  if (rmodality.get_name().empty())
    error("RadionuclideDB: no modality set for the Radionuclide info");
  if (rname.empty())
    error("RadionuclideDB: no nuclide set for the Radionuclide info");

  
  

  std::string name = rname;

  float  keV ;
  float  h_life ;
  float branching_ratio;
  
  info("RadionuclideDB: finding record radionuclide: " + rname+
       " in file "+ filename);
  #ifdef nlohmann_json_FOUND
  //Extract appropriate chunk of JSON file for given nuclide.
  nlohmann::json target = radionuclide_json["nuclide"][name]["modality"][rmodality.get_name()]["properties"];
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
  keV = properties["kev"];
  branching_ratio = properties["BRatio"];
  h_life = properties["half_life"];
  
  //  this->breakPoint = properties["break"];
  
//Set Radionuclide member
  Radionuclide rnuclide(rname,
                        keV,
                        branching_ratio,
                        h_life,
                        rmodality);
  
  return rnuclide;
#endif
}

Radionuclide 
RadionuclideDB::
get_radionuclide(ImagingModality rmodality, const std::string& rname){
    std::string nuclide_name = get_radionuclide_name_from_lookup_table(rname);
#ifdef nlohmann_json_FOUND
   return get_radionuclide_from_json(rmodality,nuclide_name);
#else
    if(rmodality.get_modality()==ImagingModality::PT){
        warning("Since I did not find nlohmann-json-dev, the radionuclide information are the same as F-18."
                " Decay correction and Branching ratio could be wrong!");
        Radionuclide rnuclide("^18^Fluorine",
                              511,
                              0.9686,
                              6584.04,
                              rmodality);
        this->radionuclide=rnuclide;
    }else if(rmodality.get_modality()==ImagingModality::NM){
        warning("Since I did not find nlohmann-json-dev, the radionuclide information are the same as Tc-99m."
                " Decay correction could be wrong!");
        Radionuclide rnuclide("^99m^Technetium",
                              140.511,
                              0.885,
                              21624.12,
                              rmodality);
        this->radionuclide=rnuclide;
    }
    return this->radionuclide;
    #endif
}

END_NAMESPACE_STIR

