//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
  \brief Implementation of class stir::GatedProjData
  \author Kris Thielemans
  \author Charalampos Tsoumpas
  
  $Date$
  $Revision$
*/

#include "stir/GatedProjData.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/ProjDataFromStream.h"
#include <iostream>
#include "stir/Succeeded.h"
#include "stir/KeyParser.h"
#include "stir/is_null_ptr.h"
#include <fstream>

START_NAMESPACE_STIR
#if 0
static
GatedProjData * 
read_multi_gated_proj_data(const string& filename)
#endif

GatedProjData*
GatedProjData::
read_from_file(const string& filename) // The written image is read in respect to its center as origin!!!
{
  const int max_length=300;
  char signature[max_length];

  // read signature
  {
    std::fstream input(filename.c_str(), std::ios::in | std::ios::binary);
    if (!input)
      error("GatedProjData::read_from_file: error opening file %s\n", filename.c_str());
    input.read(signature, max_length);
    signature[max_length-1]='\0';
  }

  GatedProjData * gated_proj_data_ptr = 0;

#ifdef HAVE_LLN_MATRIX
  if (strncmp(signature, "MATRIX", 6) == 0)
  {
#ifndef NDEBUG
    warning("GatedProjData::read_from_file trying to read %s as ECAT7\n", filename.c_str());
#endif
    USING_NAMESPACE_ECAT
    USING_NAMESPACE_ECAT7

    if (is_ECAT7_emission_file(filename))
    {
      Main_header mhead;
      if (read_ECAT7_main_header(mhead, filename) == Succeeded::no)
	{
	  warning("GatedProjData::read_from_file cannot read %s as ECAT7\n", filename.c_str());
	  return 0;
	}
      gated_proj_data_ptr = new GatedProjData;
      gated_proj_data_ptr->_scanner_sptr =
	find_scanner_from_ECAT_system_type(mhead.system_type);

      const unsigned int num_gates =
	static_cast<unsigned int>(mhead.num_gates); // TODO +1?
      gated_proj_data_ptr->_proj_datas.resize(num_gates); 

      for (unsigned int gate_num=1; gate_num <= num_gates; ++ gate_num)
	{
	  gated_proj_data_ptr->_proj_datas[gate_num-1] =
	    ECAT7_to_PDFS(filename,
			  1,
			  gate_num, 
			  /*  data_num, bed_num, */ 0,0);
	}
      if (is_null_ptr(gated_proj_data_ptr->_proj_datas[0]))
	      error("GatedProjData: No gate available\n");
    }
    else
    {
      if (is_ECAT7_file(filename))
	warning("GatedProjData::read_from_file ECAT7 file %s should be an emission sinogram\n", filename.c_str());
    }
  }
  else 
#endif // end of HAVE_LLN_MATRIX
   if (strncmp(signature, "Multigate", 9) == 0)
     {
       //#ifndef NDEBUG
       warning("GatedProjData::read_from_file trying to read %s as Multigate", filename.c_str());
       //#endif
       //return read_multi_gated_proj_data(filename);

       std::vector<std::string> filenames;
       KeyParser parser;
       parser.add_start_key("Multigate");
       parser.add_stop_key("end");
       parser.add_key("filenames", &filenames);
       if (parser.parse(filename.c_str()) == false)
	 {
	   warning("GatedProjData:::read_from_file: Error parsing %s", filename.c_str());
	   return 0;
	 }
    
       gated_proj_data_ptr = new GatedProjData;
       const unsigned int num_gates =
	 static_cast<unsigned int>(filenames.size());
       gated_proj_data_ptr->_proj_datas.resize(num_gates); 

       for (unsigned int gate_num=1; gate_num <= num_gates; ++ gate_num)
	 {
	   std::cerr<<" Reading " << filenames[gate_num-1]<<'\n';
	   gated_proj_data_ptr->_proj_datas[gate_num-1] =
	     ProjData::read_from_file(filenames[gate_num-1]);
	 }
      gated_proj_data_ptr->_scanner_sptr =
	new Scanner(*gated_proj_data_ptr->_proj_datas[0]->get_proj_data_info_ptr()->get_scanner_ptr());
      return gated_proj_data_ptr;
     }    
  
  if (is_null_ptr(gated_proj_data_ptr))   
    error("GatedProjData::read_from_file unrecognised file format for file '%s'",
	  filename.c_str());
  return gated_proj_data_ptr;
}

Succeeded 
GatedProjData::
write_to_ecat7(const string& filename) const 
{
#ifndef HAVE_LLN_MATRIX
  return Succeeded::no;
#else

  Main_header mhead;
  ecat::ecat7::make_ECAT7_main_header(mhead, filename, 
				      *get_proj_data(1).get_proj_data_info_ptr() );
  mhead.num_gates = 1;
  mhead.num_gates = this->get_num_gates();
  mhead.acquisition_type =
    mhead.num_gates>1 ? DynamicEmission : StaticEmission;

  MatrixFile* mptr= matrix_create (filename.c_str(), MAT_CREATE, &mhead);
  if (mptr == 0)
    {
      warning("GatedProjData::write_to_ecat7 cannot write output file %s\n", filename.c_str());
      return Succeeded::no;
    }
  for (  unsigned int gate_num = 1 ; gate_num<=this->get_num_gates() ;  ++gate_num ) 
    {
      if (ecat::ecat7::ProjData_to_ECAT7(mptr,
					 get_proj_data(gate_num),
					 1,
					 gate_num)
	  == Succeeded::no)
      {
        matrix_close(mptr);
        return Succeeded::no;
      }
    }
  matrix_close(mptr);
  return Succeeded::yes;
#endif // end of HAVE_LLN_MATRIX
}

END_NAMESPACE_STIR
