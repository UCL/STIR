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

#ifndef __stir_recon_buildblock_DistributedCachingInformation_h__
#define __stir_recon_buildblock_DistributedCachingInformation_h__

/*!
  \ingroup distributable
  
  \brief Declaration of stir::DistributedCachingInformation()

  \author Tobias Beisel

  $Date$
*/

#include "stir/ViewSegmentNumbers.h"
#include <vector>

START_NAMESPACE_STIR

/*!
  \ingroup distributable
  \brief This class implements the logic needed to support caching in a distributed manner.

  To enable distributed caching, the master needs to store, how the view segment numbers or the 
  related viewgrams respectively were distributed between the workers. Doing that, the master 
  is able to send those viegrams to a slave requesting for work, which that specific slave already 
  handled before and potentially still has cached locally. 
  Additionally it is possible to only send the view segment number instead of the whole projection
  data, if the the worker stores the recieved related viewgrams.
  
  This class stores all needed information to determine a not yet processed view segment number
  that will be sent to the requesting slave in that way, that the belonging data most likely 
  is stored in the slaves cache. 
  
  The function \c get_unprocessed_vs_num() can be called by the master, passing the requesting worker
  and the current subset to determine the next processed view segment number
  
  The logic is a bit sophisticated, as it has to make sure, that every vs_num is 
  only processed once and that load balancing is forced. That is, why a lot of data storage has
  to be done as a trade-off to enable fast computation.
  
  The subset number is generally used as a multiplier to determine array- or vector positions
  
  
  Whether to use the cache enabled function or not can be set by the parsing parameter
\verbatim
  enable distributed caching := 0
\endverbatim
  within the parameter specification for the objective function, where 1 activates it and 0 deactivates
  caching. The default is set to 0.
    
  */
class DistributedCachingInformation
{
 public:

  //general values needed
  int num_workers;
  int num_vs_numbers;
  int num_views;
  int num_segments;
  int num_subsets;
	
  //standard constructor
  DistributedCachingInformation();
	
  //constructor to initialize the number of segments, views and subsets
  DistributedCachingInformation(int num_segments, int num_views, int num_subs);
  virtual ~DistributedCachingInformation();

  /*! \brief to be called at the beginning of each iteration to initialize the data structures
   * calls set_all_vs_num_unprocessed() and initialize_counts()
   */ 
  void initialize_new_iteration();
	
  /*! \brief main function of the class, to be called by distributable_computation to get next vs_number
   * \warning this must only be called if there for sure is an unprocessed vs_num left, otherwise it might cause an error
   * \param proc the processor for which the View-Segment-Numbers are calculated
   * \param subset_num the current subset
   */ 
  ViewSegmentNumbers get_unprocessed_vs_num(int proc, int subset_num);

  /*! \brief gets the count of unprocessed View-Segment-Numbers assigned to a processor
   * \param proc the processor for which the View-Segment-Numbers are retured
   * \param subset_num the current subset
   */
  int get_remaining_vs_nums(int proc, int subset_num);
	
  /*! \brief adds a process
   * \param proc the processor to be added to the processor list
   * \param vs_num the ViewSegmentNumbers to which the processor is added
   * the list is needed to find other processiors which have this vs_num in cache to adjust 
   * their remaining-wor-counters in adjust_counters_for_all_proc
   */
  void add_proc_to_vs_num(ViewSegmentNumbers vs_num, int proc);
	
  /*! \brief
   * \param proc the processor to whose list the View-Segment-Numbers is saved
   * \param vs_num the vs_number to be saved to the processors list of cached numbers
   * \param subset_num the current subset
   * this list is needed to send vs_nums to the slaves, which they already processed before
   */
  void add_vs_num_to_proc(int proc, ViewSegmentNumbers vs_num, int subset_num);
	

 private:
  //std::vector<ViewSegmentNumbers> vs_vector(num_vs_numbers);
  //std::vector<std::vector<ViewSegmentNumbers> > proc_vs_nums(num_workers*num_subsets, vs_vector);
	
  //stores the vs_nums for every processor/subset combination
  std::vector<std::vector<ViewSegmentNumbers>* > proc_vs_nums;
	
  /*saves the size of each vs_num-vector
   *This is needed to determine the number of vs_nums handeled by a specific slave 
   * to enable fast access */
  int * proc_vs_nums_sizes;						 
	
  //stores the processor numbers for every vs_num
  //needed to adjust counters of other processors if the vs_num is processed 
  std::vector<std::vector<int>* > vs_num_procs; 	
  //saves the size of each processor-vector
  int * vs_num_procs_sizes;						
	
  //counts the remaining (already cached) work to do for every processor
  int * remaining_viewgrams; 						
	
  //counts the already done work for every processor
  int * processed_count; 							
	
  //marks the vs_num already processed
  int * processed; 								
	
  /*! \brief called by the constructor to initialize the data structures
   */
  void initialize();
	
  /*! \brief initializes the processed_count and the remaining_viewgrams in each iteration
   */
  void initialize_counts();

	
  /*! \brief gets the count of View-Segment-Numbers processed processed by a processor
   * \param proc the processor for which the View-Segment-Numbers are calculated
   * \param subset_num the currently processed subset number
   */
  int get_processed_count(int proc, int subset_num);	

	 
  /*! \brief gets a vector of processor-ids for a specific View-Segment-Numbers
   * \param proc the processor for which the View-Segment-Numbers are returned 
   * 
   * The resulting vector contains all processor which processed the specific vs_num at least once 	 
   */
  std::vector<int>* get_procs_from_vs_num(ViewSegmentNumbers vs_num);


  /*! \brief gets the vs_num which is (most likely) the oldest in cache of the specific processor
   * \param proc the processor for which the View-Segment-Numbers are calculated
   * \param subset_num the currently processed subset number 
   * 
   * This is called by \cget_vs_num_of_proc_with_most_work_left() if there is any unprocessed vs_num
   */
  ViewSegmentNumbers get_oldest_unprocessed_vs_num(int proc, int subset_num);

	
  /*! \brief gets a vs_num of the processor, which has the most work left
   * \param proc the processor for which the View-Segment-Numbers are calculated
   * \param subset_num the currently processed subset number
   * 	 
   * this function is called if a processor requests work and already accomplished 
   * all of his own work. Getting a viewgram from the slave with most work left
   * encourages load balancing without having a lot of extra communicatrions.
   * That way the probability of requesting already cached work is kept high.  
   */
  ViewSegmentNumbers get_vs_num_of_proc_with_most_work_left(int proc, int subset_num);
	
	
  /*! \brief to decrease the remaining vs_num counter for a specific processor
   * \param proc the processor for which the remaining_vs_nums are decreased
   * \param subset_num the currently processed subset number
   * 
   * this function is always called if one of the remaining vs_nums of a processor is 
   * chosen to be sent to a slave. It's also called for processors, from which other 
   * processors inherited work from.  
   */
  void decrease_remaining_vs_nums(int proc, int subset_num);
	
	
  /*! \brief to increase the processed vs_num counter for a specific processor
   * \param proc the processor for which the View-Segment-Numbers are calculated
   * \param subset_num the currently processed subset number
   * 
   * this increases the counter of work done for each processor to know where to access 
   * the array of saved vs_nums
   */
  void increase_processed_count(int proc, int subset_num);
	
  /*! \brief to set a ViewSegmentNumbers as already processed
   * \param vs_num the vs_num to be set processed
   * 
   * accesses the processed-array, in which every vs_num is assigned
   * a 0 (not processed) or 1 (processed)
   */
  void set_processed(ViewSegmentNumbers vs_num);
	
  /*! \brief to initialize the processed array
   * this function sets all ViewSegmentNumbers as unprocessed
   */
  void set_all_vs_num_unprocessed();
	
  /*! \brief check if a vs_num is already processed
   * \param vs_num the view segment number to be checked
   * 
   * This function checks whether a view segment number is already processed or not
   */
  bool is_processed(ViewSegmentNumbers vs_num);

  /*! \brief 
   * \param vs_num the processed view segment number
   * \param subset_num the currently processed subset number
   * 
   * This function decreases the remaining_viewgram_counts of all processors which have 
   * the vs_num on their remaining work list.
   */
  void adjust_counters_for_all_proc(ViewSegmentNumbers vs_num, int proc, int subset_num);
};

END_NAMESPACE_STIR

#endif /*__stir_recon_buildblock_DistributedCachingInformation_h__*/
