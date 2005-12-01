//
// $Id$
//
/*!
  \file
  \ingroup utilities
  \brief Declaration of class stir::DynamicDiscretisedDensity
  \author Kris Thielemans
  \author Charalampos Tsoumpas
  
  $Date$
  $Revision$
*/
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
#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/Scanner.h"
#include <vector>
#include <string>

START_NAMESPACE_STIR

class Succeeded;

class DynamicDiscretisedDensity
{
public:
  /*
    \warning the image is read in respect to its center as origin!!!
  */
  static
  DynamicDiscretisedDensity*
    read_from_file(const std::string& filename);

  DynamicDiscretisedDensity() {};

  DynamicDiscretisedDensity(const TimeFrameDefinitions& time_frame_definitions,
			    const shared_ptr<Scanner>& scanner_sptr)
  {
    _densities.resize(time_frame_definitions.get_num_frames());
    _time_frame_definitions=time_frame_definitions;
    _scanner_sptr=scanner_sptr;
  }
  /*!
    \warning This function is likely to disappear later, and is dangerous to use.
  */
  void 
    set_density_sptr(const shared_ptr<DiscretisedDensity<3,float> >& density_sptr, 
		     const unsigned int frame_num);
  /*
  DynamicDiscretisedDensity(  TimeFrameDefinitions time_frame_defintions,shared_ptr<Scanner>,
		 std::vector<shared_ptr<DiscretiseDensity<3,float> > _densities);
  */
  /*
    \warning The frame_num starts from 1
  */
  const DiscretisedDensity<3,float> & 
    get_density(const unsigned int frame_num) const ;
  /*
    \warning The frame_num starts from 1
  */
  const DiscretisedDensity<3,float> & 
    operator[](const unsigned int frame_num) const 
    { return this->get_density(frame_num); }

  const TimeFrameDefinitions & 
    get_time_frame_definitions() const ;

 /*
   \Warning write_time_frame_definitions() is not yet implemented, so time information is missing.
 */
  Succeeded   
 write_to_ecat7(const std::string& filename) const;

private:
  TimeFrameDefinitions _time_frame_definitions;
  std::vector<shared_ptr<DiscretisedDensity<3,float> > >_densities;
  shared_ptr<Scanner> _scanner_sptr;
};

END_NAMESPACE_STIR
