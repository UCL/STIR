//
//
/*
    Copyright (C) 2007- 2011, Hammersmith Imanet Ltd
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
  \ingroup recon_buildblock

  \brief Implementation of class stir::DistributedCachingInformation

  \author Tobias Beisel  
  \author Kris Thielemans
*/


#include "stir/recon_buildblock/DistributedCachingInformation.h"
#include "stir/recon_buildblock/distributed_functions.h"

START_NAMESPACE_STIR

DistributedCachingInformation::DistributedCachingInformation(const int num_workers_v)
  : num_workers(num_workers_v)
{
  initialise();
}


DistributedCachingInformation::~DistributedCachingInformation()
{       
}

void DistributedCachingInformation::initialise()
{        
  //initialize vector sizes
  this->proc_vs_nums.resize(this->num_workers);
  for (int i=0; i<this->num_workers; i++) 
    this->proc_vs_nums[i].resize(0);
         
  this->vs_nums_to_process.resize(0);
  this->initialise_new_subiteration(this->vs_nums_to_process);
}

void DistributedCachingInformation::initialise_new_subiteration(const std::vector<ViewSegmentNumbers>& vs_nums_to_process_v)
{
  this->vs_nums_to_process= vs_nums_to_process_v;
  this->set_all_vs_num_unprocessed();
}

void DistributedCachingInformation::set_all_vs_num_unprocessed()
{
  this->still_to_process.resize(this->vs_nums_to_process.size());
  std::fill(this->still_to_process.begin(), this->still_to_process.end(), true);
}

int DistributedCachingInformation::find_vs_num_position_in_list_to_process(const ViewSegmentNumbers& vs_num) const
{
  std::vector<ViewSegmentNumbers>::const_iterator iter = 
    std::find(this->vs_nums_to_process.begin(), this->vs_nums_to_process.end(), vs_num);
  if (iter== this->vs_nums_to_process.end())
    error("Internal error: asked for vs_num that is not in the list");
  return iter - this->vs_nums_to_process.begin();
}

int DistributedCachingInformation::find_position_of_first_unprocessed() const
{
  std::vector<bool>::const_iterator iter = 
    std::find(this->still_to_process.begin(), this->still_to_process.end(), true);
  if (iter== this->still_to_process.end())
    error("Internal error: asked for unprocessed, but all done");
  return iter - this->still_to_process.begin();
}

void DistributedCachingInformation::set_processed(const ViewSegmentNumbers& vs_num)
{
  this->still_to_process[this->find_vs_num_position_in_list_to_process(vs_num)]=false;
}

bool DistributedCachingInformation::is_still_to_be_processed(const ViewSegmentNumbers& vs_num) const
{
  std::vector<ViewSegmentNumbers>::const_iterator iter = 
    std::find(this->vs_nums_to_process.begin(), this->vs_nums_to_process.end(), vs_num);
  if (iter== this->vs_nums_to_process.end())
    return false;
  else
    return this->still_to_process[iter - this->vs_nums_to_process.begin()];
}

void DistributedCachingInformation::add_vs_num_to_proc(int proc, const ViewSegmentNumbers& vs_num)
{       
  this->proc_vs_nums[proc].push_back(vs_num);
  this->set_processed(vs_num);
}

int DistributedCachingInformation::get_num_remaining_cached_data_to_process(int proc) const
{
  int cnt = 0;
  for (std::vector<ViewSegmentNumbers>::const_iterator iter=this->proc_vs_nums[proc].begin();
       iter!=this->proc_vs_nums[proc].end(); ++ iter)
    {
      if (this->is_still_to_be_processed(*iter)) 
        cnt++;
    }
  return cnt;
}

bool DistributedCachingInformation::get_oldest_unprocessed_vs_num(ViewSegmentNumbers& vs, int proc) const
{
  //find unprocessed vs_num
  for (std::vector<ViewSegmentNumbers>::const_iterator iter=this->proc_vs_nums[proc].begin();
       iter!=this->proc_vs_nums[proc].end(); ++ iter)
    {   
      if (this->is_still_to_be_processed(*iter)) 
        {
          vs=*iter;
          return true;
        }
    }
         
  return false; 
}

bool DistributedCachingInformation::get_unprocessed_vs_num(ViewSegmentNumbers& vs, int proc)
{
  const bool in_cache = this->get_oldest_unprocessed_vs_num(vs, proc);
  if (!in_cache)
    {
      // no cached unprocessed data found
      // get work from worker with most work left
      vs = this->get_vs_num_of_proc_with_most_work_left(proc);
    }
        
  this->add_vs_num_to_proc(proc, vs);
  return !in_cache;     
}

ViewSegmentNumbers DistributedCachingInformation::get_vs_num_of_proc_with_most_work_left(int proc) const
{                
  int proc_with_max_work=0;
  int max_work=0;

  //find processor with most work left
  for (int i=0; i<this->num_workers;i++)
    {
      if (i==proc) continue;
      const int cnt = this->get_num_remaining_cached_data_to_process(i);
      if (cnt>max_work)
        {
          max_work=cnt;
          proc_with_max_work = i;
        } 
    }

  ViewSegmentNumbers vs;
  //get vs_number of the processor with most work left
  if (max_work>0) 
    {
      this->get_oldest_unprocessed_vs_num(vs, proc_with_max_work);
    }
  else
    {
      // just return first unprocessed 
      vs = this->vs_nums_to_process[this->find_position_of_first_unprocessed()];
    }
        
  return vs;
}


END_NAMESPACE_STIR
