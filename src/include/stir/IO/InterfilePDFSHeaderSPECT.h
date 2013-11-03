/*
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
    Copyright (C) 2013, University College London
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
  \ingroup InterfileIO
  \brief  This file declares the class stir::InterfilePDFSHeaderSPECT

  \author Berta Marti Fuster
  \author Kris Thielemans
*/


#ifndef __stir_INTERFILEHEADERSPECT_H__
#define __stir_INTERFILEHEADERSPECT_H__

#include "stir/ByteOrder.h"
#include "stir/NumericInfo.h"
#include "stir/KeyParser.h"
#include "stir/PatientPosition.h"
#include "stir/ProjDataFromStream.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/IO/InterfileHeader.h"


START_NAMESPACE_STIR

class ProjDataInfo;

/*!
  \brief a class for Interfile keywords (and parsing) specific to 
  projection data (i.e. ProjDataFromStream)
  \ingroup InterfileIO
  */
class InterfilePDFSHeaderSPECT : public InterfileHeader
{
public:
  InterfilePDFSHeaderSPECT();

protected:

  //! Returns false if OK, true if not.
  virtual bool post_processing();

public:
 
  vector<int> num_axial;
  vector<double> radius_of_rotation;
  
  // derived values
  int num_segments;
  int num_views;
  int num_bins;
  int start_angle;
  string direction_of_rotation;
  int extent_of_rotation;
  string orbit;
  
  ProjDataFromStream::StorageOrder storage_order;
  shared_ptr<ProjDataInfo> data_info_sptr;

private:
  double bin_size_in_cm; 

};


END_NAMESPACE_STIR

#endif // __stir_INTERFILEHEADERSPECT_H__
