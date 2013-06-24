//
/*
 Copyright (C) 2000- 2008, Hammersmith Imanet Ltd
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
 \brief Implementation of class stir::TimeGateDefinitions 
 \author Charalampos Tsoumpas
 \author Kris Thielemans
*/

#include "stir/TimeGateDefinitions.h"
#include <iostream>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::make_pair;
using std::cerr;
using std::endl;
using std::ifstream;
#endif

START_NAMESPACE_STIR


double
TimeGateDefinitions::
get_gate_duration(unsigned int num) const
{
  return this->_gate_sequence[num-1].second;
}

unsigned int
TimeGateDefinitions::
get_gate_num(unsigned int num) const
{
  return this->_gate_sequence[num-1].first;
}

unsigned int
TimeGateDefinitions::
get_num_gates() const
{
  return this->_gate_sequence.size();
}

unsigned int
TimeGateDefinitions::
get_num_time_gates() const
{
  return this->_gate_sequence.size();
}

TimeGateDefinitions::
TimeGateDefinitions()
{}

TimeGateDefinitions::
TimeGateDefinitions(const string& gdef_filename)
{
	TimeGateDefinitions::read_gdef_file(gdef_filename);
}

    
void
TimeGateDefinitions::
read_gdef_file(const string& gdef_filename)
{
  ifstream in(gdef_filename.c_str());
  if (!in)
    {
      const string gdef_newfilename=gdef_filename+".gdef";
      warning("TimeGateDefinitions: Warning failed reading \"%s\"\n Trying with .gdef extension...", gdef_filename.c_str());
      ifstream innew(gdef_filename.c_str());
      if (!innew)
        error("TimeGateDefinitions: Error reading \"%s\"\n", gdef_newfilename.c_str());
    }
  while (true)
    {
      int gate_num;
      double duration;
      in >> gate_num >> duration;
      if (!in)
        break;
      if (gate_num<0 || (gate_num>0 && duration<0))
        error("TimeGateDefinitions: Reading gate_def file \"%s\":\n"
	      "encountered negative numbers (%d, %g)\n",
	      gdef_filename.c_str(), gate_num, duration);
      this->_gate_sequence.push_back(make_pair(gate_num, duration));
    }
  if (this->get_num_gates()==0)
    error("TimeGateDefinitions: Reading gate definitions file \"%s\":\n"
	  "I didn't discover any gates. Wrong file format?\n"
	  "A text file with something like\n\n"
	  "3 50.5\n1 10\n10 7\n\n"
	  "for 3rd gate of 50.5 secs, 1st gate of 10 secs, 10th gate of 7 secs.",
	  gdef_filename.c_str());
}

TimeGateDefinitions::
TimeGateDefinitions(const vector<pair<unsigned int, double> >& gate_sequence)
  : _gate_sequence(gate_sequence)
{
  if (gate_sequence.size() == 0)
    error("TimeGateDefinitions: constructed with gate_sequence of no gates");
  return;
	
  this->_gate_sequence.resize(gate_sequence.size());
  for (unsigned int current_gate = 1; 
       current_gate <= this->_gate_sequence.size(); 
       ++current_gate)
    {
      this->_gate_sequence[current_gate-1].first = 
        gate_sequence[current_gate-1].first;
      this->_gate_sequence[current_gate-1].second = 
        gate_sequence[current_gate-1].second;
    }
}

TimeGateDefinitions::
TimeGateDefinitions(const vector<unsigned int>& gate_num_vector, 
		            const vector<double>& duration_vector)
{
  if (gate_num_vector.size() != duration_vector.size())
    error("TimeGateDefinitions: constructed with gate_sequence "
          "and durations of different length");
  this->_gate_sequence.resize(gate_num_vector.size());
  for (unsigned int current_gate = 1; 
       current_gate <= gate_num_vector.size(); 
       ++current_gate)
    {
      this->_gate_sequence[current_gate-1].first = 
        gate_num_vector[current_gate-1];
      this->_gate_sequence[current_gate-1].second = 
        duration_vector[current_gate-1];
    }
}

END_NAMESPACE_STIR
