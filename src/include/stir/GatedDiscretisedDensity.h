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
 \ingroup data_buildblock
 \brief Implementation of class stir::GatedDiscretisedDensity
 \author Charalampos Tsoumpas
 \author Kris Thielemans
 */
#ifndef __stir_GatedDiscretisedDensity_H__
#define __stir_GatedDiscretisedDensity_H__

#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include "stir/TimeGateDefinitions.h"
#include <vector>
#include <string>

START_NAMESPACE_STIR

class Succeeded;

/*! \ingroup data_buildblock
  \brief Class of multiple image gates.
*/
class GatedDiscretisedDensity
{
 public:
  static
    GatedDiscretisedDensity*
    read_from_file(const std::string& filename);
	
  static
    GatedDiscretisedDensity*
    read_from_files(const std::string& filename);
	
  static
    GatedDiscretisedDensity*
    read_from_files(const std::string& filename,const std::string& suffix);
	
  GatedDiscretisedDensity() {}

  GatedDiscretisedDensity(const GatedDiscretisedDensity&argument);
  GatedDiscretisedDensity(const std::string& filename);

  GatedDiscretisedDensity(const TimeGateDefinitions& time_gate_definitions)
    {
      _densities.resize(time_gate_definitions.get_num_gates());
      _time_gate_definitions=time_gate_definitions;
    }

  GatedDiscretisedDensity(const shared_ptr<DiscretisedDensity<3,float> >& density_sptr, 
                          const unsigned int num_gates);
	
	
  //!  Construct an empty GatedDiscretisedDensity based on a shared_ptr<DiscretisedDensity<3,float> >
  GatedDiscretisedDensity(const TimeGateDefinitions& time_gate_definitions,
                          const shared_ptr<DiscretisedDensity<3,float> >& density_sptr)
    {  
      _densities.resize(time_gate_definitions.get_num_gates());
      _time_gate_definitions=time_gate_definitions;
    
      for (unsigned int gate_num=0; gate_num<time_gate_definitions.get_num_gates(); ++gate_num)
        this->_densities[gate_num].reset(density_sptr->get_empty_discretised_density()); 
    }  

#if 0
  //!  Construct an empty GatedDiscretisedDensity based on another GatedDiscretisedDensity
  GatedDiscretisedDensity(const GatedDiscretisedDensity gated_density)
    {  
      _densities.resize(gated_density.get_num_gates());
      _time_gate_definitions=gated_density.get_time_gate_definitions();		
      for (unsigned int gate_num=0; gate_num<time_gate_definitions.get_num_gates(); ++gate_num)
        this->_densities[gate_num] = (gated_density.get_densities[0])->get_empty_discretised_density(); 
    }
#endif
	
  GatedDiscretisedDensity&
    operator=(const GatedDiscretisedDensity& argument);

  /*! \name get/set the densities
    \warning The gate_num starts from 1
  */
  //@{
  /*!
    \warning This function is likely to disappear later, and is dangerous to use.
  */
  void 
    set_density_sptr(const shared_ptr<DiscretisedDensity<3,float> >& density_sptr, 
		     const unsigned int gate_num);

  void resize_densities(const TimeGateDefinitions& time_gate_definitions)
  {
    _densities.resize(time_gate_definitions.get_num_gates());
    _time_gate_definitions=time_gate_definitions;
  }
	
  /*
    GatedDiscretisedDensity(  TimeGateDefinitions time_gate_defintions,
    std::vector<shared_ptr<DiscretiseDensity<3,float> > _densities);
  */

  const std::vector<shared_ptr<DiscretisedDensity<3,float> > > &
    get_densities() const ;

  const DiscretisedDensity<3,float> & 
    get_density(const unsigned int gate_num) const ;

  const DiscretisedDensity<3,float> & 
    operator[](const unsigned int gate_num) const 
    { return this->get_density(gate_num); }

  DiscretisedDensity<3,float> & 
    get_density(const unsigned int gate_num);

  DiscretisedDensity<3,float> & 
    operator[](const unsigned int gate_num)  
    { return this->get_density(gate_num); }
  //@}

  void set_time_gate_definitions(TimeGateDefinitions time_gate_definitions) 
  {this->_time_gate_definitions=time_gate_definitions;}

  const TimeGateDefinitions & 
    get_time_gate_definitions() const ;

  unsigned get_num_gates() const
  {
    return this->get_time_gate_definitions().get_num_time_gates();
  }
  void fill_with_zero();	

  /*! \brief write data to file
    Currently only in format.
    \warning write_time_gate_definitions() is not yet implemented, so time information is missing.
  */
  Succeeded   
    write_to_files(const std::string& filename) const;
  Succeeded   
    write_to_files(const std::string& filename,const std::string& suffix) const;


 private:
  // warning: if adding any new members, you have to change the copy constructor as well.
  TimeGateDefinitions _time_gate_definitions;
  std::vector<shared_ptr<DiscretisedDensity<3,float> > > _densities;
};

END_NAMESPACE_STIR

#endif //__stir_GatedDiscretisedDensity_H__
