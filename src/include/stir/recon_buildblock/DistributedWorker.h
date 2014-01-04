/*
    Copyright (C) 2007 - 2011-02-23, Hammersmith Imanet Ltd
    Copyright (C) 2014, University College London
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
  \author Kris Thielemans
*/

#include "stir/shared_ptr.h"
#include "stir/TimedObject.h"
//#include "stir/ParsingObject.h"
#include "stir/ProjData.h"
#include "stir/recon_buildblock/ProjectorByBinPair.h"
#include "stir/recon_buildblock/distributable.h"
#include <string>
#include <vector>

START_NAMESPACE_STIR

class ExamInfo;

/*!
  \ingroup distributable
  \brief This implements the Worker for the stir::distributable_computation() function.

  The start() method is an infinite loop waiting for a task from the master. Very few tasks
  are implemented at the moment: set_up, compute, stop.
  
  The \c distributable_computation() function does the actual work. It is the slave-part
  of stir::distributable_computation() which runs on the master.  It is a loop receiving the related 
  viewgrams and calling an RPC_process_related_viewgrams_type function 
  with the received values. When an end_iteration_notification is received, it calls the 
  reduction of the output_image.
  
  In each inner loop the worker first receives the vs_num and the information whether this is a 
  new viewgram or a previously received viewgram. The latter case only emerges if distributed caching is 
  enabled.  If so, the worker does not have to receive the related viewgrams, but just gets it from 
  its saved viewgrams.

  \todo The log_likelihood_ptr argument to the RPC function is currently always NULL.
  \todo Currently the only computation that is supported corresponds to the gradient computation.
  It would be trivial to add others.
*/
template <class TargetT>
class DistributedWorker : public TimedObject //, public ParsingObject
{       
 private:       
                                
  double* log_likelihood_ptr;
  bool zero_seg0_end_planes;
  shared_ptr<ProjectorByBinPair> proj_pair_sptr;
  shared_ptr<ExamInfo> exam_info_sptr;
  shared_ptr<ProjDataInfo> proj_data_info_sptr;
  shared_ptr<TargetT> target_sptr;
                
  int image_buffer_size; //to save the image_size

  // cache variables
  bool cache_enabled;
  shared_ptr<ProjData> proj_data_ptr;   
  shared_ptr<ProjData> binwise_correction;
  shared_ptr<ProjData> mult_proj_data_sptr;

  int my_rank; //rank of the worker

                
 public:
        
  //Default constructor
  DistributedWorker();
        
  //Default destructor
  ~DistributedWorker() ;
      
  /*!
    \brief Infinite loop waiting for tasks from the master
  */
  void start();
                        
 protected:
        
  /*!
    \brief sets defaults for this object 
  */
  void set_defaults();

  /*!
    \brief Get basic information from the master.
                  
    It sets up all needed objects by communicating with the 
    master. 
                  
    The following objects are set up:
    - bool zero_seg0_end_planes
    - target_image_sptr
    - ProjDataInfo pointer
    - ProjectorByBinPair pointer
    - ProjDataInMemory to save the received related viewgrams
                  
    Additionally some values for testing are set up. 
                  
  */
  void setup_distributable_computation();
  /*!
    \brief this does the actual computation corresponding to distributable_computation()
  */
  void distributable_computation(RPC_process_related_viewgrams_type * RPC_process_related_viewgrams);
};


END_NAMESPACE_STIR

#endif
// __DistributedWorker_h__

