/*
    Copyright (C) 2005-2011, Hammersmith Imanet Ltd
    Copyright (C) 2009-2013, King's College London
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
*/

#include "stir/GatedProjData.h"
#include "stir/IO/FileSignature.h"
#include "stir/IO/stir_ecat7.h"
#include "stir/ProjDataFromStream.h"
#include <iostream>
#include "stir/Succeeded.h"
#include "stir/KeyParser.h"
#include "stir/is_null_ptr.h"
#include <fstream>
#include <sstream>

using std::string;

START_NAMESPACE_STIR

GatedProjData*
GatedProjData::
read_from_file(const string& filename) // The written image is read in respect to its center as origin!!!
{
  std::fstream input(filename.c_str(), std::ios::in | std::ios::binary);
  if (!input)
    {
      warning("GatedProjData::read_from_file cannot read file '%s'. Will now attempt to append .gdef", filename.c_str());
      return read_from_gdef(filename);
    }

  const FileSignature file_signature(input);
  const char * signature = file_signature.get_signature();

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

      const unsigned int num_gates =
	static_cast<unsigned int>(mhead.num_gates); // TODO +1?
      gated_proj_data_ptr->_proj_datas.resize(num_gates); 

      for (unsigned int gate_num=1; gate_num <= num_gates; ++ gate_num)
	{
	  gated_proj_data_ptr->_proj_datas[gate_num-1].reset(
	    ECAT7_to_PDFS(filename,
			  1,
			  gate_num, 
			  /*  data_num, bed_num, */ 0,0));
	}
      if (is_null_ptr(gated_proj_data_ptr->_proj_datas[0]))
	      error("GatedProjData: No gate available\n");
      // Get the exam info (from the first ProjData)
      if (num_gates>0)
        gated_proj_data_ptr->set_exam_info(gated_proj_data_ptr->_proj_datas[0]->get_exam_info());
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
       // Get the exam info (from the first ProjData)
       if (num_gates>0)
         gated_proj_data_ptr->set_exam_info(gated_proj_data_ptr->_proj_datas[0]->get_exam_info());
      return gated_proj_data_ptr;
     }    
  
  if (is_null_ptr(gated_proj_data_ptr))   
    error("GatedProjData::read_from_file unrecognised file format for file '%s'",
	  filename.c_str());
  return gated_proj_data_ptr;
}

GatedProjData* 
GatedProjData::read_from_gdef(const string& filename)
{
  const string gdef_filename=filename+".gdef";
  std::cout << "GatedProjData: Reading gate definitions " << gdef_filename.c_str() << std::endl;
  GatedProjData * gated_proj_data_ptr = new GatedProjData;
  gated_proj_data_ptr->_time_gate_definitions.read_gdef_file(gdef_filename);
  gated_proj_data_ptr->_proj_datas.resize(gated_proj_data_ptr->_time_gate_definitions.get_num_gates());
  for ( unsigned int num = 1 ; num<=(gated_proj_data_ptr->_time_gate_definitions).get_num_gates() ;  ++num ) 
    {	
      std::stringstream gate_num_stream;
      gate_num_stream << gated_proj_data_ptr->_time_gate_definitions.get_gate_num(num);
      const string input_filename=filename+"_g"+gate_num_stream.str()+".hs";
      std::cout << "GatedProjData: Reading gate projection file: " << input_filename.c_str() << std::endl;
      gated_proj_data_ptr->_proj_datas[num-1] = ProjData::read_from_file(input_filename);
    }	
  if (is_null_ptr(gated_proj_data_ptr))   
    error("GatedProjData::read_from_file unrecognised file format for projection files with prefix '%s'",
          filename.c_str());
  // Get the exam info (from the first ProjData)
  if (gated_proj_data_ptr->get_num_gates()>0)
     gated_proj_data_ptr->set_exam_info(gated_proj_data_ptr->_proj_datas[0]->get_exam_info());
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
				      *get_proj_data(1).get_exam_info_ptr(),
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

#if 0 // This is not necessary for the moment as we only read the files.
Succeeded 
GatedProjData::
write_to_files(const string& filename) const 
{
	for (  unsigned int num = 1 ; num<=(_time_gate_definitions).get_num_gates() ;  ++num ) 
	{
		std::stringstream gate_num_stream;
		gate_num_stream << _time_gate_definitions.get_gate_num(num);
		const string output_filename=filename+"_g"+gate_num_stream.str()+".hv";
		std::cout << "GatedProjData: Writing new gate file: " << output_filename.c_str() << std::endl;
		if(XXX->write_to_file(output_filename, this->get_density(num))
		   == Succeeded::no)
			return Succeeded::no;
	}	
	if((this->_time_gate_definitions).get_num_gates()==0)
		std::cout << "GatedDiscretisedDensity: No gates to write, please check!!" <<  std::endl;
	return Succeeded::yes;	 
	
  return Succeeded::yes;
}
#endif

END_NAMESPACE_STIR
