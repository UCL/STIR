//
/*
 Copyright (C) 2000- 2008, Hammersmith Imanet Ltd
 Copyright (C) 2009 - 2013, King's College London
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
  \brief Declaration of class stir::TimeGateDefinitions
    
  \author Charalampos Tsoumpas
  \author Kris Thielemans
 
  \todo This files needs proper test
      
*/
#ifndef __stir_TimeGateDefinitions_H__
#define __stir_TimeGateDefinitions_H__

#include "stir/common.h"
#include <string>
#include <vector>
#include <utility>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::pair;
using std::vector;
#endif

START_NAMESPACE_STIR
/*!
  \ingroup buildblock
  \brief Class used for storing time gate durations

  Times are supposed to be relative to the scan start time.

  Currently this class can read gate info from an ECAT6, ECAT7 and a 'gate definition'
  file. See the documentation for the constructor.

  Will probably be superseded by Study classes.
*/
class TimeGateDefinitions
{
 public:
  //! Default constructor: no time gates at all
  TimeGateDefinitions();
  TimeGateDefinitions(const vector<unsigned int>& gate_num_vector, 
                      const vector<double>& duration_vector);
  TimeGateDefinitions(const vector<pair<unsigned int, double> >& gate_sequence);
  explicit TimeGateDefinitions(const string& gdef_filename);

  //! Read the gate definitions from a file
  /*! 
    The filename can point to a simple ASCII text file.
    The format is a number of lines, each existing of 2 numbers
    \verbatim
    gate_num   duration_in_secs
    \endverbatim
    This duration is a double number.

    This class in fact allows an extension of the above. Setting 
    \a gate_num_of_this_duration to 0 allows skipping
    a time period of the corresponding \a duration_in_secs.
  */
  void read_gdef_file(const string& gdef_filename);

  //! \name get info for a gate
  //@{
  double get_gate_duration(unsigned int num) const;
  unsigned int	get_gate_num(unsigned int num) const;

  //@}

  //! Get number of gates
  unsigned int get_num_gates() const;
  //! Get number of gates
  unsigned int get_num_time_gates() const;

 private:
  //! Stores start and end time for each gate
  vector<pair<unsigned int, double> > _gate_sequence;
};

END_NAMESPACE_STIR
#endif
