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

#ifndef __stir_recon_buildblock_DistributedFunctions_h__
#define __stir_recon_buildblock_DistributedFunctions_h__

/*!
  \file 
  \ingroup distributable
 
  \brief Declaration of functions in the distributed namespace

  \author Tobias Beisel

  $Date$
*/

/*!
  \namespace distributed
  \brief Namespace for distributed computation with MPI

  This is a collection of functions to send and receive objects and data 
  needed for distributed computation with MPI. They all come in a separate 
  namespace "distributed". There was no need to have an object providing this
  functionality as that would have been more costly.
	
  Note that every send function has a corresponding receive function.

  \see STIR_MPI
  \see STIR_MPI_TIMINGS

  Compilation instructions are in the Users Guide.
*/
#ifdef DOXYGEN_SKIP
// need to define the following to get the documentation
#define STIR_MPI
#define STIR_MPI_TIMINGS
#endif
/*!
  \def STIR_MPI
  \brief Precompiler-define that needs to be set to enable the parallel version of OSMAPOSL.
*/

/*!
  \def STIR_MPI_TIMINGS
  \brief Precompiler-define that needs to be set to enable timings in the parallel version of OSMAPOSL.

  The functions in the distributable namespace can include run-time
  measurements included which print times from start to end of a single
  Send or Receive command. This will uncover some maybe unnecessary long waiting times.
  It was actually used while developing the parallel version to check whether some code
  rearrangements would lead to faster computation. 
  To enable these timings, compile the MPI-Version with the preprocessor variable
  STIR_MPI_TIMINGS defined.
	
  If compiled with STIR_MPI_TIMINGS defined, you still have the possibility to enable/disable the timings by
  using the parsing parameter 
  \verbatim
  enable message timings := 1
  \endverbatim
  To give the printed times a threshold use 
  \verbatim
  message timings threshold := 0.1
  \endverbatim
*/

#include "mpi.h"
#include "stir/shared_ptr.h"
#include "stir/ParsingObject.h"
#include "stir/DataSymmetriesForViewSegmentNumbers.h"
#include "stir/ProjDataInMemory.h"
#include "stir/IO/InterfileHeader.h"
#include "stir/ProjDataInterfile.h"
#include "stir/recon_buildblock/DistributedCachingInformation.h"
#include "stir/recon_buildblock/ProjectorByBinPair.h"
#include "stir/DiscretisedDensity.h"
#include "stir/RelatedViewgrams.h"
#include "stir/Viewgram.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfo.h"
#include <cstring>
#include <iostream>
#include <fstream>
#include "stir/HighResWallClockTimer.h"

namespace distributed
{
  //!the number of processes used for distributed computation 
  extern int num_processors;
	
  //!some stuff in distributable_computation needs to be done only in the first iteration
  extern bool first_iteration;
	
  //!needed in cache enabled case to check whether the viewgrams have to be distributed 
  //or if the computation is already progressed enough to use cached values
  extern int iteration_counter;
	
  //!enable/disable tests
  extern bool test; 					
	
  //for timings
  extern bool rpc_time; 		//!enable timings for PRC_process_related_viewgrams_gradient() computation	
  extern bool test_send_receive_times;	//!enable timings for every single send/receive operation 
	
  extern double total_rpc_time;		//!adding up the time used for PRC_process_related_viewgrams_gradient() computation at all slaves
  extern double total_rpc_time_2;	//!adding up the time used for PRC_process_related_viewgrams_gradient() computation at a single slave
  extern double total_rpc_time_slaves;	//!value to reduce the total_rpc_time values
  extern double min_threshold;		//!threshold for displaying send/receive times, initially set to 0.1 seconds
	
	
  //----------------------Send operations----------------------------------
	
  /*! \brief sends or broadcasts an integer value
   * \param value the int value to be sent
   * \param destination the process id where to send the interger value. If set to -1 a Broadcast will be done
   * 
   * This function is currently only used to determine the objective function using an idertifier.
   * It should be replaced by the more general function \csend_int_values()
   */
  void send_int_value(int value, int destination);
	
	
  /*! \brief sends or broadcasts a string
   * \param str the string to be sent
   * \param tag identifier to associate messages
   * \param destination the process id where to send the string. If set to -1 a Broadcast will be done
   */
  void send_string(const string& str, int tag, int destination);
	
	
  /*! \brief send or broadcast a bool value
   * \param value the bool value to be sent
   * \param tag identifier to associate messages. If the tag is -1 a Broadcast will be done
   * \param destination the process id where to send the bool value. If set to -1 a Broadcast will be done
   * 
   * This function actually sends an integer value (0 or 1) as there is no bool datatype in MPI
   */
  void send_bool_value(bool value, int tag, int destination);
	
	
  /*! \brief sends or broadcasts some integer values
   * \param values pointer to integer values to be sent
   * \param count the count of integer values to be sent
   * \param tag identifier to associate messages
   * \param destination the process id where to send the int values. If set to -1 a Broadcast will be done
   */
  void send_int_values(int * values, int count, int tag, int destination);
	
	
  /*! \brief send or broadcast double values
   * \param values pointer to the double values to be sent
   * \param count the count of double values to be sent
   * \param tag identifier to associate messages
   * \param destination the process id where to send the double values. If set to -1 a Broadcast will be done
   */
  void send_double_values(double * values, int count, int tag, int destination);
	
	
  /*! \brief sends or broadcasts the parameters of a DiscretisedDensity object
   * \param input_image_ptr the image_ptr to be sent
   * \param tag identifier to associate messages
   * \param destination the process id where to send the image parameters. If set to -1 a Broadcast will be done
   * 
   * This function sends all parameters needed to construct a corresponding image object at the 
   * slave. The image values are sent separately. This is done, as sending the parameters is not 
   * needed everytime the values are sent. The slave needs to receive the current_image_estimate
   * every iteration, but he already knows the parameters.
   * 
   * The actual parameters sent are the image dimensions, the origin and the grid_spacing 
   */
  void send_image_parameters(const stir::DiscretisedDensity<3,float>* input_image_ptr, int tag, int destination);
	
	
  /*! \brief sends or broadcasts the values of a DiscretisedDensity object
   * \param input_image_ptr the image_ptr to be sent
   * \param tag identifier to associate messages
   * \param destination the process id where to send the image values. If set to -1 a Broadcast will be done
   * 
   * This function sends the values of an image. The values are serialized to a one-dimensional array
   * as MPI only sends that kind of data structures.
   */
  void send_image_estimate(const stir::DiscretisedDensity<3,float>* input_image_ptr, int destination);
	
	
  /*! \brief sends or broadcasts the parameter_info() of the ProjDataInfo pointer
   * \param data the ProjDataInfo pointer to be sent
   * \param destination the process id where to send the image values. If set to -1 a Broadcast will be done
   * 
   * For sending the ProjDataInfo object a rather "dirty trick" is used:
   * Instead of sending all needed data for constructing a ProjDataInfd object, 
   * the parameter_info() is temporarily stored as ProjDataInterfile and then buffered
   * as text from the file. Afterwards, that is sent to the slave. 
   * 
   * Doing this, the slave is able to construct the ProjDataInfo by using the received infomation
   * for a InterfilePDFSHeader object. The sent char-array is used as stream-
   * input to the parse() function of InterfilePDFSHeader. 
   */
  void send_proj_data_info(stir::ProjDataInfo* data, int destination);
	
	
  /*! \brief sends a RelatedViegrams object
   * \param viewgrams the viewgrams to be sent
   * \param destination the process id where to send the related viewgrams
   * 
   * This function iterates through the relatedviewgrams object and calls \csend_viewgram() 
   * for each comprised viewgram. 
   * The only value sent is the count of viewgrams contained within the related_viewgrams object
   * to make sure that the worker knows how many viewgrams he has to receive.
   */
  void send_related_viewgrams(stir::RelatedViewgrams<float>* viewgrams, int destination);
	
	
  /*! \brief sends a Viewgram object
   * \param viewgram the viewgrams to be sent
   * \param destination the process id where to send the viewgram
   * 
   * This function sends all parameters needed for the construction of the viewgram at the worker,
   * as well as the actual values of the viewgram. That would mean that 2 Messages are sent:
   * 1. The dimensions of the viewgram and the vs_num
   * 2. The values detwermined by iterating through the viewgram and serializing it to a one-dimensional array
   */
  void send_viewgram(const stir::Viewgram<float>& viewgram, int destination);
	
	
  //----------------------Receive operations----------------------------------
	
	
  /*! \brief receives a single integer value
   * \param source the process id from which to receive the interger value. If set to -1 the receive will be done from broadcast
   * \returns the received int value
   */
  int receive_int_value(int source);
	
	
  /*! \brief receives a string
   * \param tag unique identifier to associate messages
   * \param source the process id from which to receive the string
   * \returns the received string
   */
  string receive_string(int tag, int source);


  /*! \brief receives all needed information to subsequently construct a ProjectorByBinPair object
   * \param projector_pair_ptr address pointer of the new ProjectorByBinPair pointer 
   * \param source the process id from which to receive the ProjectorByBinPair
   * 
   * First the registered_name string of the ProjectorByBinPair is received. 
   * Then the parameter_info() of the masters ProjectorByBinPair is received. 
   * Using the registered_name and the parameter_info() as stream, both can be used as 
   * input to the read_registered_object function, which then constructs the new ProjectorByBinPair pointer.
   * 
   * The passed address pointer parameter will then be redirected to the address of ProjectorByBinPair created 
   * using the received parameters.  
   */
  void receive_and_initialize_projectors(stir::shared_ptr<stir::ProjectorByBinPair> &projector_pair_ptr, int source);
	
	
  /*! \brief receives a bool value
   * \param tag unique identifier to associate messages
   * \param source the process id from which to receive the bool value
   * \returns the received bool value
   * 
   * This function actually receives an integer value (0 or 1) 
   * as there is no bool datatype in MPI, but it will return a bool value
   */
  bool receive_bool_value(int tag, int source);
	
	
  /*! \brief receives some integer values
   * \param values pointer to the receive buffer
   * \param count the count of integer values to be received
   * \param tag identifier to associate messages
   * \returns MPI_Status object to query the source of the message
   * 
   * The tag needs to be set to ARBITRARY_TAG (=8) if MPI_ANY_TAG shall be used
   */
  MPI_Status receive_int_values(int * values, int count, int tag);
	
	
  /*! \brief receives some double values
   * \param values pointer to the receive buffer
   * \param count the count of double values to be received
   * \param tag identifier to associate messages
   * \returns MPI_Status object to query the source of the message
   * 
   * The tag needs to be set to ARBITRARY_TAG (=8) if MPI_ANY_TAG shall be used
   */
  MPI_Status receive_double_values(double* values, int count, int tag);
	
	
  /*! \brief receives the parameters of a DiscretisedDensity object
   * \param image_ptr address pointer of the new DiscretisedDensity 
   * \param buffer saves the image buffer size to be reused when receiving the image values
   * \param tag identifier to associate messages. If set to -1 a Broadcast will be done
   * \param source the process id from which to receive the image parameters.
   * 
   * This function receives all parameters needed to construct an image object at the 
   * slave. The image values are sent separately.
   * 
   * The actual parameters received are the image dimensions, the origin and the grid_spacing
   * 
   * The function currently only supports VoxelsOnCartesianGrid 
   */
  void receive_and_set_image_parameters(stir::shared_ptr<stir::DiscretisedDensity<3, float> > &image_ptr, int &buffer, int tag, int source);
	
	
  /*! \brief receives the values of a DiscretisedDensity object
   * \param image_ptr the image_ptr to be sent
   * \param buffer_size gives the needed size of the receive buffer
   * \param source the process id from which to receive the image values.
   * \returns MPI_Status object to query the source of the message
   * 
   * The image_ptr is filled by iterating through the target pointer and copying 
   * the single values from the receive buffer.
   * 
   * The buffer_size is used again to reduce the image values
   */
  MPI_Status receive_image_values_and_fill_image_ptr(stir::shared_ptr<stir::DiscretisedDensity<3,float> > &image_ptr, int buffer_size, int source);
  	
  	
  /*! \brief receives the parameter_info() of the ProjDataInfo pointer and constructs a new ProjDataInfo object from it
   * \param proj_data_info_sptr the new ProjDataInfo pointer to be set up
   * \param source the process id from which to receive the ProjDataInfo
   * 
   * The parameter info is received as a Interfile Header sting. That way the slave is able 
   * to construct a ProjDataInfo within a InterfilePDFSHeader using the received 
   * char-array as stream-input to the parse() function of InterfilePDFSHeader. 
   */
  void receive_and_construct_proj_data_info_ptr(stir::shared_ptr<stir::ProjDataInfo>& proj_data_info_sptr, int source);
  	
  	
  /*! \brief receives and constructs a RelatedViegrams object
   * \param proj_data_info_ptr the ProjDataInfo pointer describing the data
   * \param symmetries_sptr the symmetries pointer constructed when setting up the projectors
   * \param source the process id from which to receive the ProjDataInfo
   * 
   * First of all it is important to notice, that this function is not independent. To construct
   * a new RelatedViegrams object, the symmetries_ptr must be available. That would mean, 
   * that the slave has to call \c receive_and_initialize_projectors() to set up the symmetries_ptr
   * before the related_vewgrams can be received. 
   * Additionaly the  ProjDataInfo-pointer must be available for receiving a single viewgrams.
   * That implies calling \c receive_and_construct_proj_data_info_ptr() before. 
   * To make this function independent, both of these objects have to be sent here. On the other hand 
   * that would lead to the overhead of sending it everytime a related_viewgram is sent, 
   * which is really expensive.  
   * 
   * This function receives the count of viewgrams to be received and calls
   * \c receive_and_construct_viewgram that often. Every received viewgram is pushed back
   * to a viewgram vector, which afterwards is used with the symmetries to construct 
   * a RelatedViewgrams object.  
   */
  void receive_and_construct_related_viewgrams(stir::RelatedViewgrams<float>*& viewgrams, 
					       const stir::shared_ptr<stir::ProjDataInfo>& proj_data_info_ptr, 
					       const stir::shared_ptr<stir::DataSymmetriesForViewSegmentNumbers> symmetries_sptr,
					       int source);
	
  /*! \brief receives a Viewgram object
   * \param viewgram the viewgrams to be constructed
   * \param proj_data_info_ptr the ProjDataInfo pointer describing the data
   * \param source the process id from which to receive the ProjDataInfo
   * 
   * This function received all parameters needed for the construction of the viewgram at the worker,
   * as well as the actual values of the viewgram. That would mean that 2 Messages are received:
   * 1. The dimensions of the viewgram and the vs_num
   * 2. The values of the viewgram
   * 
   * The buffer_size needed to receive the values is calculated from the dimensions received.
   * The viewgram is filled by iterating througn it and copying the values of the received values.
   */
  void receive_and_construct_viewgram(stir::Viewgram<float>*& viewgram, 
				      const stir::shared_ptr<stir::ProjDataInfo>& proj_data_info_ptr, 
				      int source); 
	
  //-----------------------reduce operations-------------------------------------
	
  /*! \brief the function called by the master to reduce the output image
   * \param output_image_ptr the image pointer where the reduced image is saved 
   * \param destination the process id where the output_image is reduced 
   */
  void reduce_received_output_image(stir::DiscretisedDensity<3,float>* output_image_ptr, int destination);
	
  /*! \brief the function called by the slaves to reduce the output image
   * \param output_image_ptr the image pointer where the reduced image is saved
   * \param image_buffer_size the buffer size needed for the image 
   * \param my_rank rank of the slave, only used for screen output
   * \param destination the process id where the output_image is reduced 
   * 
   * The buffer size was calculated in \c receive_image_values_and_fill_image_ptr().
   * Alternatively it can be calculated by the image parameters.
   */
  void reduce_output_image(stir::shared_ptr<stir::DiscretisedDensity<3,float> > &output_image_ptr, int image_buffer_size, int my_rank, int destination);

  /* Tag-names currently used */
#define AVAILABLE_NOTIFICATION_TAG 2
#define END_ITERATION_TAG 3
#define END_RECONSTRUCTION_TAG 4
#define END_NOTIFICATION_TAG 5

#define BINWISE_CORRECTION_TAG 6
#define INT_TAG 7
#define ARBITRARY_TAG 8

#define REUSE_VIEWGRAM_TAG 10
#define NEW_VIEWGRAM_TAG 11

#define PARAMETER_INFO_TAG 21
#define IMAGE_ESTIMATE_TAG 23
#define IMAGE_PARAMETER_TAG 24
#define REGISTERED_NAME_TAG 25

#define VIEWGRAM_DIMENSIONS_TAG 27
#define VIEWGRAM_TAG 28
#define VIEWGRAM_COUNT_TAG 29

#define PROJECTION_DATA_INFO_TAG 30

}

#endif

