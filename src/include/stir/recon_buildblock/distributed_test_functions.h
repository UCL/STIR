//
//
/*
    Copyright (C) 2007- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_recon_buildblock_DistributedTestFunctions_h__
#define __stir_recon_buildblock_DistributedTestFunctions_h__

/*!
  \file 
  \ingroup distributable
 
  \brief Declaration of test functions for the distributed namespace

  This is a collection of functions to test the function implemented 
  in DistributedFunction.cxx . Each of the possible tests consists of a 
  master and a slave function to be called by different processes.

  Note that every master function has a corresponding slave function.
  
  \todo Currently no independent test functions are implemented. The tests are used
  by embedding them into the reconstruction functions and calling them once. 
  
  This is only done in debug mode and can be enabled/disabled with the following 
  parsing parameter: 
  \verbatim
  enable distributed tests := 1
  \endverbatim
  The default value is 0.
	
  This obviously not a good way to do testing, so there is a need to write independent
  test functions. The problem with that is, that all needed objects have to be set up.
  The function headers give an idea of what would have to be constructed. 
	
  \author Tobias Beisel

*/


#include "mpi.h"
#include "stir/shared_ptr.h"
#include "stir/DataSymmetriesForViewSegmentNumbers.h"
#include "stir/DiscretisedDensity.h"
#include "stir/RelatedViewgrams.h"
#include "stir/Viewgram.h"
#include "stir/ProjDataInfo.h"
#include <iostream>
#include <fstream>

namespace distributed
{
  //-----------------------test functions------------------------------------------
	
	
  void test_viewgram_slave(const  stir::shared_ptr<stir::ProjDataInfo>& proj_data_info_ptr);
	
  void test_viewgram_master(stir::Viewgram<float> viewgram, const  stir::shared_ptr<stir::ProjDataInfo>& proj_data_info_ptr);
	
  void test_image_estimate_master(const stir::DiscretisedDensity<3,float>* input_image_ptr, int slave);
	
  void test_image_estimate_slave();
	
  void test_related_viewgrams_master(const stir::shared_ptr<stir::ProjDataInfo>& proj_data_info_ptr, 
				     const stir::shared_ptr<stir::DataSymmetriesForViewSegmentNumbers> symmetries_sptr,
				     stir::RelatedViewgrams<float>* y, int slave);
	
  void test_related_viewgrams_slave(const stir::shared_ptr<stir::ProjDataInfo>& proj_data_info_ptr, 
				    const stir::shared_ptr<stir::DataSymmetriesForViewSegmentNumbers> symmetries_sptr
				    );
	
  void test_parameter_info_master(const std::string str, int slave, char const * const text);
	
  void test_parameter_info_slave(const std::string str);
	
  void test_bool_value_master(bool value, int slave);
	
  void test_bool_value_slave();
	
  void test_int_value_master(int value, int slave);
		
  void test_int_value_slave();
	
  void test_int_values_master(int slave);
	
  void test_int_values_slave();
}

#endif

