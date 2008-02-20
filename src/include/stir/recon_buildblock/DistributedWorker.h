//
// $Id$
//
/*
    Copyright (C) 2007- $Date$, Hammersmith Imanet Ltd
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

#ifndef __stir_recon_buildblock_DistributedWorker_h__
#define __stir_recon_buildblock_DistributedWorker_h__


/*!
  \file 
  \ingroup recon_buildblock
 
  \brief declares the stir::DistributedWorker class


  \author Tobias Beisel

  $Date$
*/
//class PoissonLogLikelihoodWithLinearModelForMeanAndProjData;

#include "mpi.h"
#include "stir/Viewgram.h"
#include "stir/shared_ptr.h"
#include "stir/TimedObject.h"
#include "stir/ParsingObject.h"
#include "stir/DiscretisedDensity.h"
#include "stir/RelatedViewgrams.h"
#include "stir/DataSymmetriesForViewSegmentNumbers.h"
#include "stir/ProjDataInMemory.h"
#include "stir/IO/InterfileHeader.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/recon_buildblock/IterativeReconstruction.h"
#include <string>
#include <vector>

START_NAMESPACE_STIR

/*!
  \ingroup distributable
  \brief This implements the Worker for the distributed reconstruction.

  It first initializes all objects needed for the reconstruction within the \c start() method,
  before running into two nested infinite loops to receive work packages.
  
  The loops are running within the \c distributed_compute_sub_gradient() fuinction 
  which does the actual work. The outer loop represents the loop over the iterations and receives 
  the values of the current image estimate at the beginning of each iteration.
  
  The inner loop receives the related viegrams and computes the RPC_process_related_viewgrams_gradient() 
  function with the received values. When an end_iteration_notification is received calls the 
  reduction of the output_image.
  
  In each inner loop the worker first receives the vs_num and the information whether this is a 
  new viewgram or a previously received viewgram. The latter case only emerges if distribute caching is 
  enabled.  If so, the worker does not have to receive the related viewgrams, but just gets it from 
  its saved viewgrams.
  
  \c distributed_compute_sensitivity() is intended to compute the sensitivity in a 
  distributed manner within the set_up() method. This has to be done.
  */
template <class TargetT>
class DistributedWorker : public TimedObject, public ParsingObject
{	
 private:	
  MPI_Status status;
				
  float* log_likelihood_ptr;
  bool zero_seg0_end_planes;
		
  int image_buffer_size; //to save the image_size
		
  shared_ptr<ProjData> proj_data_ptr; 	
  shared_ptr<ProjData> binwise_correction;
		
  int my_rank; //rank of the worker
		
  bool cache_enabled;

		
 public:
	
  //Default constructor
  DistributedWorker();
   	
  //Default destructor
  ~DistributedWorker() ;
      
  /*!
    \ingroup distributable
    \brief 
		  
    This function in used to start the slave. It sets up all needed object communicating with the 
    master. It differs between the objective_functions, where the list-mode case is not yet implemented.
    The objective_functions are differed with the value sent. PoissonLogLikelihoodWithLinearModelForMeanAndListModeData
    is 100, while PoissonLogLikelihoodWithLinearModelForProjData is 200.
		  
    The following objects are set up:
    - bool zero_seg0_end_planes
    - target_image_sptr
    - ProjDataInfo pointer
    - ProjectorByBinPair pointer
    - PoissonLogLikelihoodWithLinearModelForMeanAndProjData object
    - ProjDataInMemory to save the received related viewgrams
		  
    Setting uo the ProjectorByBinPair pointer includes setting up the DataSymmetriesForViewSegmentNumbers
    pointer and the ProjMatrix Pointer.
    Additionaly some values for testing are set up. 
		  
  */
  void start(int my_rank);
			
 protected:
	
  /*!
    \ingroup distributable
    \brief sets defaults for this object 
  */
  void set_defaults();
   		
   		
  /*!
    \ingroup distributable
    \brief not yet implemented parallel version of the sensitivity computation 
  */
  void distributed_compute_sensitivity(shared_ptr<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT > > &objective_function_ptr);
   		
   		
  /*!
    \ingroup distributable
    \brief this does the actual reconstruction of the received related viewgrams
  */
  void distributed_compute_sub_gradient(shared_ptr<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT > > &objective_function_ptr, 
					const shared_ptr<ProjDataInfo> proj_data_info_sptr, 
					const shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr);
};


END_NAMESPACE_STIR

#endif
// __DistributedWorker_h__
k
