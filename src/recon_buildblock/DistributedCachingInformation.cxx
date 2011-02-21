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
/*!

  \file
  \ingroup recon_buildblock

  \brief Implementation of stir::DistributedCachingInformation()

  \author Tobias Beisel  

  $Date$
*/


#include "stir/recon_buildblock/DistributedCachingInformation.h"
#include "stir/recon_buildblock/distributed_functions.h"

START_NAMESPACE_STIR

DistributedCachingInformation::DistributedCachingInformation(int num_segm, int num_vws, int num_subs)
{	
  num_subsets = num_subs;
  num_segments = num_segm;
	
  //TODO: restrict to basic view_nums 
  num_views = num_vws;
  num_vs_numbers= num_segments * num_views; 
  initialize();
}

DistributedCachingInformation::DistributedCachingInformation()
{
}


DistributedCachingInformation::~DistributedCachingInformation()
{	
}

void DistributedCachingInformation::initialize()
{
  num_workers= distributed::num_processors-1;
	 
  //initialize vector sizes
  proc_vs_nums.resize(num_workers*num_subsets);
  vs_num_procs.resize(num_vs_numbers);
	 
  for (int i=0; i<num_workers*num_subsets; i++) proc_vs_nums[i] = new vector<ViewSegmentNumbers>(num_vs_numbers);
  for (int i=0; i<num_vs_numbers; i++) vs_num_procs[i]= new vector<int>(num_workers*num_subsets);
	 
  proc_vs_nums_sizes = new int[num_workers*num_subsets];
  vs_num_procs_sizes = new int[num_vs_numbers];
	 
  //initialize sizes
  for (int i=0; i<num_workers*num_subsets;i++) proc_vs_nums_sizes[i]=0;
  for (int i=0; i<num_vs_numbers;i++) vs_num_procs_sizes[i]=0;
	 
  remaining_viewgrams = new int[num_workers*num_subsets];
  processed_count = new int[num_workers*num_subsets];
  for (int i=0; i<num_workers*num_subsets; i++) 
    {
      remaining_viewgrams[i]=0;
      processed_count[i]=0;
    }			 
  processed = new int[num_vs_numbers];
  this->set_all_vs_num_unprocessed();
}

void DistributedCachingInformation::initialize_new_iteration()
{
  set_all_vs_num_unprocessed();
  initialize_counts();
  //	printf("numviews=%i\t num_segments=%i\t num_vs_nums=%i\t", num_views, num_segments, num_vs_numbers);
  //	printf("num_subsets=%i", num_subsets);
}

int DistributedCachingInformation::get_remaining_vs_nums(int proc, int subset_num)
{
  return remaining_viewgrams[(proc-1)+subset_num*num_workers];
}

void DistributedCachingInformation::decrease_remaining_vs_nums(int proc, int subset_num)
{
  remaining_viewgrams[(proc-1)+subset_num*num_workers] = remaining_viewgrams[(proc-1)+subset_num*num_workers]-1;
}

void DistributedCachingInformation::initialize_counts()
{
  //	printf("Initialized remaining viewgrams:\n");
  for (int i=0; i<num_workers*num_subsets; i++) 
    {
      remaining_viewgrams[i]=proc_vs_nums_sizes[i];
      //			printf("Pos %i: %i\n", i, proc_vs_nums_sizes[i]);
    }		
  for (int i=0; i<num_workers*num_subsets; i++) processed_count[i]=0;
}

int DistributedCachingInformation::get_processed_count(int proc, int subset_num)
{
  return processed_count[(proc-1)+subset_num*num_workers];
}


void DistributedCachingInformation::increase_processed_count(int proc, int subset_num)
{
  processed_count[(proc-1)+subset_num*num_workers] = processed_count[(proc-1)+subset_num*num_workers]+1;
}


void DistributedCachingInformation::set_processed(ViewSegmentNumbers vs_num)
{
  int position = vs_num.segment_num()*num_views+vs_num.view_num();
  processed[position]=1;
}


void DistributedCachingInformation::set_all_vs_num_unprocessed()
{
  for (int i=0; i<num_vs_numbers;i++)
    processed[i]=0;
}

bool DistributedCachingInformation::is_processed(ViewSegmentNumbers vs_num)
{
  int position = vs_num.segment_num()*num_views+vs_num.view_num();
  if (processed[position] == 1) return true;
  else return false;
}


void DistributedCachingInformation::add_proc_to_vs_num(ViewSegmentNumbers vs_num, int proc)
{
  //printf("\n\n\n");
	
  const int index1 = vs_num.segment_num()*num_views+vs_num.view_num();
  const int index2 = vs_num_procs_sizes[index1];
	
  //const int view = vs_num.view_num();
  //const int segment = vs_num.segment_num();
	
  //printf("Adding processor %i to (%i,%i) in vs_num_procs at position [%i][%i]\n", proc, view ,segment , index1,  index2);
	
  vs_num_procs[index1]->at(index2) = proc;
	
  vs_num_procs_sizes[index1] += 1; //vs_num_procs_sizes[vs_num.segment_num()*num_views+vs_num.view_num()] + 1;
}

vector<int>* DistributedCachingInformation::get_procs_from_vs_num(ViewSegmentNumbers vs_num)
{
	return vs_num_procs[vs_num.segment_num()*num_views+vs_num.view_num()];
}

void DistributedCachingInformation::add_vs_num_to_proc(int proc, ViewSegmentNumbers vs_num, int subset_num)
{
  const int index1 = proc-1+(subset_num*num_workers);
  const int index2 = proc_vs_nums_sizes[index1];
	
  //const int view = vs_num.view_num();
  //const int segment = vs_num.segment_num();
	
  //printf("For Proc %i Add (%i,%i) to Position [%i][%i]\n", proc, view, segment, index1 ,index2);
  proc_vs_nums[index1]->at(index2) = vs_num;
	
  //	fprintf(stderr, "Added vs_num to proc %i at address %p and %p", proc, &proc_vs_nums[index1], &proc_vs_nums[index1]->at(index2));

  proc_vs_nums_sizes[index1] = proc_vs_nums_sizes[index1] + 1;
}

void DistributedCachingInformation::adjust_counters_for_all_proc(ViewSegmentNumbers vs_num, int proc, int subset_num)
{
  const int index1 = vs_num.segment_num()*num_views+vs_num.view_num();
  const int index2 = vs_num_procs_sizes[index1];
	
  //const int view = vs_num.view_num();
  //const int segment = vs_num.segment_num();
	
  vector<int>* procs = get_procs_from_vs_num(vs_num);
  //	printf("proc %i called adjust counters\n",  proc);
	
  //	printf("vs_num_proc_sizes[%i](%i,%i) = %i\n", index1, view, segment, index2);
  for (int i=0; i<index2;i++)
    { 
      if (procs->at(i)==proc) continue;
      //		printf("decreasing remaining viewgrams of proc %i and subset %i\n",  procs->at(i), subset_num);
      decrease_remaining_vs_nums(procs->at(i), subset_num);
    }
}


ViewSegmentNumbers DistributedCachingInformation::get_oldest_unprocessed_vs_num(int proc, int subset_num)
{
  bool found = false;
  ViewSegmentNumbers vs;
	
  const int index1 = proc-1+(subset_num*num_workers);
  int index2 = 0;
	
  //find unprocessed vs_num
  while (found == false)
    {	
      vs = proc_vs_nums[index1]->at(index2);
      //		fprintf(stderr, "Processor read from other processor %i at address: %p %p", proc, &proc_vs_nums[index1], &proc_vs_nums[index1]->at(index2));
		
      //		printf("For Proc %i Read (%i,%i) from Position [%i][%i]\n", proc, vs.view_num(), vs.segment_num(), index1 ,index2);
      if (is_processed(vs)==false) found = true;
      else index2++;
    }
	 
  return vs;	
}

ViewSegmentNumbers DistributedCachingInformation::get_unprocessed_vs_num(int proc, int subset_num)
{
  //get work from worker with most work left if processor has no own cached work left 
  if (get_remaining_vs_nums(proc, subset_num)==0) return get_vs_num_of_proc_with_most_work_left(proc, subset_num);

  ViewSegmentNumbers vs;
	
  bool found = false;
	
  const int index1 = proc-1+(subset_num*num_workers);
	
  while (found == false)
    {		
      const int processed= get_processed_count(proc, subset_num);
		
      int index2 = proc_vs_nums_sizes[index1] -1 - processed;
		
      vs = proc_vs_nums[index1]->at(index2);
      //		fprintf(stderr, "Processor %i read from address: %p %p\n", proc, &proc_vs_nums[index1], &proc_vs_nums[index1]->at(index2));
      //		printf("For Proc %i Read (%i,%i) from Position [%i][%i]\n", proc, vs.view_num(), vs.segment_num(), index1 ,index2);
		
      increase_processed_count(proc, subset_num);
		
      if (is_processed(vs)==false) found = true;
    }
	
  decrease_remaining_vs_nums(proc, subset_num);
	
  set_processed(vs);
	
  //check whether other processors have this vs_num and adjust counters
  adjust_counters_for_all_proc(vs, proc, subset_num);
	
  return vs;	
}

ViewSegmentNumbers DistributedCachingInformation::get_vs_num_of_proc_with_most_work_left(int proc, int subset_num)
{		 
  int proc_with_max_work=0;
  int max_work=0;
	
  //find processor with most work left
  for (int i=1; i<=num_workers;i++)
    {
      if (i==proc) continue;
      if (get_remaining_vs_nums(i, subset_num)>max_work)
	{
	  max_work=get_remaining_vs_nums(i, subset_num);
	  proc_with_max_work = i;
	} 
    }
	
  ViewSegmentNumbers vs;

  //get vs_number of the processor with most work left
  if (max_work>0) vs = get_oldest_unprocessed_vs_num(proc_with_max_work, subset_num);
  else //all work completed for this iteration 
    {
      //TODO: make sure there never can be a vs_num with values (-1,-1)
      ViewSegmentNumbers vs(-1, -1);
      return vs;
    }
	
  // add the vs_num to own list of vs_nums
  add_vs_num_to_proc(proc, vs, subset_num);
  add_proc_to_vs_num(vs, proc);
	
  increase_processed_count(proc, subset_num);
		
  //decrease_remaining_vs_nums(proc, subset_num);
  set_processed(vs);
	
  //check whether other processors have this vs_num and adjust counters
  adjust_counters_for_all_proc(vs, proc, subset_num);
	
  return vs;
}


END_NAMESPACE_STIR
