//
//
/*
    Copyright (C) 2003- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2014, 2018 University College London
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
  \ingroup normalisation

  \brief Implementation for class stir::BinNormalisation

  \author Kris Thielemans
*/


#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/recon_buildblock/find_basic_vs_nums_in_subsets.h"
#include "stir/RelatedViewgrams.h"
#include "stir/Bin.h"
#include "stir/ProjData.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"
#include "stir/error.h"
#include <boost/format.hpp>

START_NAMESPACE_STIR

BinNormalisation::
BinNormalisation()
  :   _already_set_up(false)
{
}

BinNormalisation::
~BinNormalisation()
{}

Succeeded
BinNormalisation::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_sptr)
{
  _already_set_up = true;
  _proj_data_info_sptr = proj_data_info_sptr->create_shared_clone();
  return Succeeded::yes;  
}

void
BinNormalisation::
check(const ProjDataInfo& proj_data_info) const
{
  if (!this->_already_set_up)
    error("BinNormalisation method called without calling set_up first.");
  if (!(*this->_proj_data_info_sptr >= proj_data_info))
    error(boost::format("BinNormalisation set-up with different geometry for projection data.\nSet_up was with\n%1%\nCalled with\n%2%")
          % this->_proj_data_info_sptr->parameter_info() % proj_data_info.parameter_info());
}
  
// TODO remove duplication between apply and undo by just having 1 functino that does the loops

void 
BinNormalisation::apply(RelatedViewgrams<float>& viewgrams,
			const double start_time, const double end_time) const 
{
  this->check(*viewgrams.get_proj_data_info_sptr());
  for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
  {
    Bin bin(iter->get_segment_num(),iter->get_view_num(), 0,0);
    for (bin.axial_pos_num()= iter->get_min_axial_pos_num(); 
	 bin.axial_pos_num()<=iter->get_max_axial_pos_num(); 
	 ++bin.axial_pos_num())
      for (bin.tangential_pos_num()= iter->get_min_tangential_pos_num(); 
	   bin.tangential_pos_num()<=iter->get_max_tangential_pos_num(); 
	   ++bin.tangential_pos_num())
        (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] /= 
          std::max(1.E-20F, get_bin_efficiency(bin, start_time, end_time));
  }
}

void 
BinNormalisation::
undo(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
{
  this->check(*viewgrams.get_proj_data_info_sptr());
  for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
  {
    Bin bin(iter->get_segment_num(),iter->get_view_num(), 0,0);
    for (bin.axial_pos_num()= iter->get_min_axial_pos_num(); 
	 bin.axial_pos_num()<=iter->get_max_axial_pos_num(); 
	 ++bin.axial_pos_num())
      for (bin.tangential_pos_num()= iter->get_min_tangential_pos_num(); 
	   bin.tangential_pos_num()<=iter->get_max_tangential_pos_num(); 
	   ++bin.tangential_pos_num())
         (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] *= 
	   this->get_bin_efficiency(bin,start_time, end_time);
  }

}

void 
BinNormalisation::
apply(ProjData& proj_data,const double start_time, const double end_time, 
      shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr) const
{
  this->check(*proj_data.get_proj_data_info_sptr());
  if (is_null_ptr(symmetries_sptr))
    symmetries_sptr.reset(new TrivialDataSymmetriesForBins(proj_data.get_proj_data_info_ptr()->create_shared_clone()));

  const std::vector<ViewSegmentNumbers> vs_nums_to_process = 
    detail::find_basic_vs_nums_in_subset(*proj_data.get_proj_data_info_ptr(), *symmetries_sptr,
                                         proj_data.get_min_segment_num(), proj_data.get_max_segment_num(),
                                         0, 1/*subset_num, num_subsets*/);

#ifdef STIR_OPENMP
#pragma omp parallel for  shared(proj_data, symmetries_sptr) schedule(runtime)  
#endif
    // note: older versions of openmp need an int as loop
  for (int i=0; i<static_cast<int>(vs_nums_to_process.size()); ++i)
    {
      const ViewSegmentNumbers vs=vs_nums_to_process[i];
      
      RelatedViewgrams<float> viewgrams;
#ifdef STIR_OPENMP
      // reading/writing to streams is not safe in multi-threaded code
      // so protect with a critical section
      // note that the name of the section has to be same for the get/set
      // function as they're reading from/writing to the same stream
#pragma omp critical (BINNORMALISATION_APPLY__VIEWGRAMS)
#endif
      {
        viewgrams = 
          proj_data.get_related_viewgrams(vs, symmetries_sptr);
      }

      this->apply(viewgrams, start_time, end_time);

#ifdef STIR_OPENMP
#pragma omp critical (BINNORMALISATION_APPLY__VIEWGRAMS)
#endif
      {
        proj_data.set_related_viewgrams(viewgrams);
      }
    }
}

void 
BinNormalisation::
undo(ProjData& proj_data,const double start_time, const double end_time, 
     shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr) const
{
  this->check(*proj_data.get_proj_data_info_sptr());
  if (is_null_ptr(symmetries_sptr))
    symmetries_sptr.reset(new TrivialDataSymmetriesForBins(proj_data.get_proj_data_info_ptr()->create_shared_clone()));

  const std::vector<ViewSegmentNumbers> vs_nums_to_process = 
    detail::find_basic_vs_nums_in_subset(*proj_data.get_proj_data_info_ptr(), *symmetries_sptr,
                                         proj_data.get_min_segment_num(), proj_data.get_max_segment_num(),
                                         0, 1/*subset_num, num_subsets*/);

#ifdef STIR_OPENMP
#pragma omp parallel for  shared(proj_data, symmetries_sptr) schedule(runtime)  
#endif
    // note: older versions of openmp need an int as loop
  for (int i=0; i<static_cast<int>(vs_nums_to_process.size()); ++i)
    {
      const ViewSegmentNumbers vs=vs_nums_to_process[i];
      
      RelatedViewgrams<float> viewgrams;
#ifdef STIR_OPENMP
#pragma omp critical (BINNORMALISATION_UNDO__VIEWGRAMS)
#endif
      {
        viewgrams = 
          proj_data.get_related_viewgrams(vs, symmetries_sptr);
      }

      this->undo(viewgrams, start_time, end_time);

#ifdef STIR_OPENMP
#pragma omp critical (BINNORMALISATION_UNDO__VIEWGRAMS)
#endif
      {
        proj_data.set_related_viewgrams(viewgrams);
      }
    }
}

 
END_NAMESPACE_STIR

