//
//
/*
    Copyright (C) 2005-2009, Hammersmith Imanet Ltd
    Copyright (C) 2010-2013, King's College London
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
/*!
  \file
  \ingroup data_buildblock
  \brief Declaration of class stir::GatedProjData
 \author Kris Thielemans
 \author Charalampos Tsoumpas

*/
#include "stir/MultipleProjData.h"
#include "stir/TimeGateDefinitions.h"
#include <string>

START_NAMESPACE_STIR

class Succeeded;

class GatedProjData :
 public MultipleProjData
{
public:
  static
  GatedProjData*
    read_from_file(const std::string& filename);

  GatedProjData() {};

  unsigned int get_num_gates() const
  {
    return this->get_num_proj_data();
  }

  Succeeded   
    write_to_ecat7(const std::string& filename) const;
  //Succeeded
  //  write_to_files(const std::string& filename) const;

 private:
  TimeGateDefinitions _time_gate_definitions;
  static GatedProjData* read_from_gdef(const string& filename);
};

END_NAMESPACE_STIR
