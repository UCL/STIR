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
  
  \brief Declaration of class stir::DistributedCachingInformation

  \author Tobias Beisel
  \author Kris Thielemans

  $Date$
*/

#include "stir/ViewSegmentNumbers.h"
#include <vector>
#include <algorithm>

START_NAMESPACE_STIR

/*!
  \ingroup distributable
  \brief This class implements the logic needed to support caching in a distributed manner.

  To enable distributed caching, the master needs to store how the view segment numbers or the 
  related viewgrams respectively were distributed between the workers. Doing that, the master 
  is able to send those viegrams to a slave requesting for work, which that specific slave already 
  handled before and still has cached locally. 
  Additionally it is possible to only send the view segment number instead of the whole projection
  data, if the the worker stores the recieved related viewgrams.
  
  This class stores all needed information to determine a not yet processed view segment number
  that will be sent to the requesting slave in such a way that the belonging data most likely 
  is stored in the slaves cache. 
  
  The function \c get_unprocessed_vs_num() can be called by the master, passing the requesting worker
  and the current subset to determine the next processed view segment number
  
  The logic is a bit sophisticated, as it has to make sure, that every vs_num is 
  only processed once and that load balancing is forced. T
  
  Process numbers are expected to be between 0 and \c num_workers
  
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
  //! constructor, calls initialise()
  explicit DistributedCachingInformation(const int num_processors);
  
  //! destructor to clean up data structures
  virtual ~DistributedCachingInformation();

  /*! \brief initialise all data structures
   * called by the constructor, but should be called before processing a new set of data.
   * calls initialise_new_subiteration() with an empty vector
   */
  void initialise();
  /*! \brief to be called at the beginning of the processing of a set of data
   * The caching data is kept, such that the cache will be re-used over multiple runs.
   */ 
  void initialise_new_subiteration(const std::vector<ViewSegmentNumbers>& vs_nums_to_process);
	
  /*! \brief get the next work-package for a given processor
   * \warning this must only be called if there for sure is an unprocessed vs_num left, 
   * otherwise it will call stir::error().
   * \param[out] vs_num will be set accordingly
   * \param[in] proc the processor for which the View-Segment-Numbers are calculated
   * \return \c true if the vs_num was not in the cache of the processor
   * This function updates internal cache values etc. The user can just repeatedly call
   * the function without worrying about the caching algorithm.
   */ 
  bool get_unprocessed_vs_num(ViewSegmentNumbers& vs_num, int proc);
	
private:

  //! Number of processors available
  int num_workers;

  //! stores which data that have to be processed in this subiteration
  std::vector<ViewSegmentNumbers> vs_nums_to_process;	

  //!stores the vs_nums in the cache of every processor
  std::vector<std::vector<ViewSegmentNumbers> > proc_vs_nums;
	
  //! marks the vs_num that still need to be processed
  /*! Has the same length as vs_nums_to_process */
  std::vector<bool> still_to_process;

  //! \brief find where \a vs_num is in vs_nums_to_process
  int find_vs_num_position_in_list_to_process(const ViewSegmentNumbers& vs_num) const;

  //! \brief find the first vs_num which has not be processed at all
  int find_position_of_first_unprocessed() const;
	
  //! count how many data-sets that are cached remain to be processed by processor \a proc
  int get_num_remaining_cached_data_to_process(int proc) const;

  /*! \brief store the work-package in the cache-list of the processor
   * \param proc the processor to whose list the View-Segment-Numbers is saved
   * \param vs_num the vs_number to be saved to the processors list of cached numbers
   * Calls set_processed() to make sure we do not process it again.
   */
  void add_vs_num_to_proc(int proc, const ViewSegmentNumbers& vs_num);

  /*! \brief gets the vs_num which is (most likely) the oldest in cache of the specific processor
   * \param[out] vs_num will be set accordingly
   * \param[in] proc the processor for which the View-Segment-Numbers are calculated
   * \return \c true if there was an unprocessed data-set in the cache of processor \a proc
   * This is called by get_unprocessed_vs_num
   */
  bool get_oldest_unprocessed_vs_num(ViewSegmentNumbers& vs_num, int proc) const;

	
  /*! \brief gets a vs_num of the processor, which has the most work left
   * \param proc processor that will not be checked (i.e. the one for which 
   *      we are trying to find some work)
   * 	 
   * this function is called if a processor requests work and already accomplished 
   * everything in its cache. Allocating work from the slave with most work left
   * encourages load balancing without having a lot of extra communicatrions.
   * That way the probability of requesting already cached work is kept high.  
   */
  ViewSegmentNumbers get_vs_num_of_proc_with_most_work_left(int proc) const;
		
  /*! \brief set a ViewSegmentNumbers as already processed
   * \param vs_num the vs_num to be set processed
   */
  void set_processed(const ViewSegmentNumbers& vs_num);
	
  /*! \brief reset all data to unprocessed
   */
  void set_all_vs_num_unprocessed();
	
  /*! \brief check if a vs_num is still to be processed or not
   * \param vs_num the view segment number to be checked
   * Also returns false if the \a vs_num is not in the list to process at all.
   */
  bool is_still_to_be_processed(const ViewSegmentNumbers& vs_num) const;

};

END_NAMESPACE_STIR

#endif /*__stir_recon_buildblock_DistributedCachingInformation_h__*/
