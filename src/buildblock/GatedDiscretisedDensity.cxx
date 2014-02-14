//
/*
 Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
 Copyright (C) 2009- 2013, King's College London
 This file is part of STIR.
 
 This file is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.3 of the License, or
 (at your option) any later version.
 
 This file is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
 
 See STIR/LICENSE.txt for details
 */  
/*!
  \file
  \ingroup buildblock
  \brief Implementation of class stir::GatedDiscretisedDensity
  \author Charalampos Tsoumpas
  \author Kris Thielemans
  
*/

#include "stir/GatedDiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include <sstream>
#include <string>
#include "stir/IO/interfile.h"
#include "stir/IO/OutputFileFormat.h"

#ifndef STIR_NO_NAMESPACES
using std::fstream;
using std::cerr;
using std::istringstream;
#endif

START_NAMESPACE_STIR

GatedDiscretisedDensity::
GatedDiscretisedDensity(const GatedDiscretisedDensity& argument)
{
  (*this) = argument;
}

GatedDiscretisedDensity&
GatedDiscretisedDensity::
operator=(const GatedDiscretisedDensity& argument)
{
  this->_time_gate_definitions = argument._time_gate_definitions;
  this->_densities.resize(argument._densities.size());
  for (unsigned int i=0; i<argument._densities.size(); ++i)
    this->_densities[i].reset(argument._densities[i]->clone());
  return *this;
}

GatedDiscretisedDensity::
GatedDiscretisedDensity(const string& filename)
{
  const string gdef_filename=filename+".gdef";
  std::cout << "GatedDiscretisedDensity: Reading gate definitions " << gdef_filename.c_str() << std::endl;
  this->_time_gate_definitions.read_gdef_file(gdef_filename);
  this->_densities.resize(this->_time_gate_definitions.get_num_gates());
  for ( unsigned int num = 1 ; num<=(this->_time_gate_definitions).get_num_gates() ;  ++num ) 
    {	
      std::stringstream gate_num_stream;
      gate_num_stream << this->_time_gate_definitions.get_gate_num(num);
      const string input_filename=filename+"_g"+gate_num_stream.str()+".hv";
      //		const shared_ptr<VoxelsOnCartesianGrid<float>  > read_sptr = new VoxelsOnCartesianGrid<float> ; 
      std::cout << "GatedDiscretisedDensity: Reading gate file: " << input_filename.c_str() << std::endl;
      this->_densities[num-1].reset(DiscretisedDensity<3,float>::read_from_file(input_filename));
    }
}

GatedDiscretisedDensity::
GatedDiscretisedDensity(const shared_ptr<DiscretisedDensity<3,float> >& density_sptr, 
                        const unsigned int num_gates)
{
  this->_densities.resize(num_gates);
  for ( unsigned int num = 1 ; num<=num_gates;  ++num )
    this->_densities[num-1].reset(density_sptr->get_empty_discretised_density());
}


void 
GatedDiscretisedDensity::
set_density_sptr(const shared_ptr<DiscretisedDensity<3,float> >& density_sptr, 
		 const unsigned int gate_num)
{  this->_densities[gate_num-1]=density_sptr; }  

const std::vector<shared_ptr<DiscretisedDensity<3,float> > > &
GatedDiscretisedDensity::
get_densities() const 
{  return this->_densities ; }

const DiscretisedDensity<3,float> & 
GatedDiscretisedDensity::
get_density(const unsigned int gate_num) const 
{  return *this->_densities[gate_num-1] ; }

DiscretisedDensity<3,float> & 
GatedDiscretisedDensity::
get_density(const unsigned int gate_num)
{  return *this->_densities[gate_num-1] ; }

const TimeGateDefinitions & 
GatedDiscretisedDensity::
get_time_gate_definitions() const
{ return this->_time_gate_definitions; }

void GatedDiscretisedDensity::
fill_with_zero()
{
  for (unsigned int gate_num=0; gate_num<this->_time_gate_definitions.get_num_gates(); ++gate_num)
    this->_densities[gate_num].reset((this->_densities[gate_num])->get_empty_discretised_density());
}

GatedDiscretisedDensity*
GatedDiscretisedDensity::
read_from_files(const string& filename) // The written image is read in respect to its center as origin!!!
{
  const string gdef_filename=filename+".gdef";
  std::cout << "GatedDiscretisedDensity: Reading gate definitions " << gdef_filename.c_str() << std::endl;
  GatedDiscretisedDensity * gated_image_ptr = 0;
  gated_image_ptr = new GatedDiscretisedDensity;
  gated_image_ptr->_time_gate_definitions.read_gdef_file(gdef_filename);
  gated_image_ptr->_densities.resize(gated_image_ptr->_time_gate_definitions.get_num_gates());
  for ( unsigned int num = 1 ; num<=(gated_image_ptr->_time_gate_definitions).get_num_gates() ;  ++num ) 
    {	
      std::stringstream gate_num_stream;
      gate_num_stream << gated_image_ptr->_time_gate_definitions.get_gate_num(num);
      const string input_filename=filename+"_g"+gate_num_stream.str()+".hv";
      const shared_ptr<VoxelsOnCartesianGrid<float>  > read_sptr(new VoxelsOnCartesianGrid<float> ); 
      std::cout << "GatedDiscretisedDensity: Reading gate file: " << input_filename.c_str() << std::endl;
      gated_image_ptr->_densities[num-1].reset(DiscretisedDensity<3,float>::read_from_file(input_filename));
    }
  return gated_image_ptr;
}

GatedDiscretisedDensity*
GatedDiscretisedDensity::
read_from_files(const string& filename,const string& suffix) // The written image is read in respect to its center as origin!!!
{
  const string gdef_filename=filename+".gdef";
  std::cout << "GatedDiscretisedDensity: Reading gate definitions " << gdef_filename.c_str() << std::endl;
  GatedDiscretisedDensity * gated_image_ptr = 0;
  gated_image_ptr = new GatedDiscretisedDensity;
  gated_image_ptr->_time_gate_definitions.read_gdef_file(gdef_filename);
  gated_image_ptr->_densities.resize(gated_image_ptr->_time_gate_definitions.get_num_gates());
  for ( unsigned int num = 1 ; num<=(gated_image_ptr->_time_gate_definitions).get_num_gates() ;  ++num ) 
    {	
      std::stringstream gate_num_stream;
      gate_num_stream << gated_image_ptr->_time_gate_definitions.get_gate_num(num);
      const string input_filename=filename+"_g"+gate_num_stream.str()+suffix+".hv";
      //		const shared_ptr<VoxelsOnCartesianGrid<float>  > read_sptr = new VoxelsOnCartesianGrid<float> ; 
      std::cout << "GatedDiscretisedDensity: Reading gate file: " << input_filename.c_str() << std::endl;
      gated_image_ptr->_densities[num-1].reset(DiscretisedDensity<3,float>::read_from_file(input_filename));
    }
  return gated_image_ptr;
}

Succeeded 
GatedDiscretisedDensity::
write_to_files(const string& filename) const 
{
  for (  unsigned int num = 1 ; num<=(_time_gate_definitions).get_num_gates() ;  ++num ) 
    {
      std::stringstream gate_num_stream;
      gate_num_stream << _time_gate_definitions.get_gate_num(num);
      const string output_filename=filename+"_g"+gate_num_stream.str()+".hv";
      std::cout << "GatedDiscretisedDensity: Writing new gate file: " << output_filename.c_str() << std::endl;
      if(OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->write_to_file(output_filename, this->get_density(num))
         == Succeeded::no)
        return Succeeded::no;
    }	
  if((this->_time_gate_definitions).get_num_gates()==0)
    std::cout << "GatedDiscretisedDensity: No gates to write, please check!!" <<  std::endl;
  return Succeeded::yes;	 
}

Succeeded 
GatedDiscretisedDensity::
write_to_files(const string& filename,const string& suffix) const
{
  for (  unsigned int num = 1 ; num<=(_time_gate_definitions).get_num_gates() ;  ++num ) 
    {
      std::stringstream gate_num_stream;
      gate_num_stream << _time_gate_definitions.get_gate_num(num);
      const string output_filename=filename+"_g"+gate_num_stream.str()+suffix+".hv";
      std::cout << "GatedDiscretisedDensity: Writing new gate file: " << output_filename.c_str() << std::endl;
      if(OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->write_to_file(output_filename, this->get_density(num))
         == Succeeded::no)
        return Succeeded::no;
    }	
  if((this->_time_gate_definitions).get_num_gates()==0)
    std::cout << "GatedDiscretisedDensity: No gates to write, please check!!" <<  std::endl;
  return Succeeded::yes;	 
}

END_NAMESPACE_STIR
