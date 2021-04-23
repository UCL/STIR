//
//
/*
    Copyright (C) 2005-2009, Hammersmith Imanet Ltd
    Copyright (C) 2010-2013, King's College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
  unique_ptr<GatedProjData>
    read_from_file(const std::string& filename);

  GatedProjData() {};

  GatedProjData(const MultipleProjData& m):
    MultipleProjData(m)
  {}

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
  static GatedProjData* read_from_gdef(const std::string& filename);
};

END_NAMESPACE_STIR
