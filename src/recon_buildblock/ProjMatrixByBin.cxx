/*!

  \file
  \ingroup projection
  
  \brief  implementation of the stir::ProjMatrixByBin class 

  \author Nikos Efthimiou
  \author Mustapha Sadki
  \author Kris Thielemans
  \author Robert Twyman
  \author PARAPET project
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2009, Hammersmith Imanet Ltd
    Copyright (C) 2013, 2015, 2022 University College London
    Copyright (C) 2016, University of Hull

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/


#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/TOF_conversions.h"

START_NAMESPACE_STIR

void ProjMatrixByBin::set_defaults()
{
  cache_disabled=false;
  cache_stores_only_basic_bins=true;
  gauss_sigma_in_mm = 0.f;
  r_sqrt2_gauss_sigma = 0.f;
}

void 
ProjMatrixByBin::initialise_keymap()
{
  parser.add_key("disable caching", &cache_disabled);
  parser.add_key("store_only_basic_bins_in_cache", &cache_stores_only_basic_bins);
}

bool
ProjMatrixByBin::post_processing()
{
  return false;
}

ProjMatrixByBin::ProjMatrixByBin()
{ 
  set_defaults();
}
 
void 
ProjMatrixByBin::
enable_cache(const bool v)
{ cache_disabled = !v;}

void
ProjMatrixByBin::
enable_tof(const shared_ptr<const ProjDataInfo>& _proj_data_info_sptr, const bool v)
{
    if (v)
    {
        tof_enabled = true;
        gauss_sigma_in_mm = tof_delta_time_to_mm(proj_data_info_sptr->get_scanner_ptr()->get_timing_resolution()) / 2.355f;
        r_sqrt2_gauss_sigma = 1.0f/ (gauss_sigma_in_mm * static_cast<float>(sqrt(2.0)));
    }
}

void 
ProjMatrixByBin::
store_only_basic_bins_in_cache(const bool v) 
{ cache_stores_only_basic_bins=v;}

bool 
ProjMatrixByBin::
is_cache_enabled() const
{ return !cache_disabled; }

bool 
ProjMatrixByBin::
does_cache_store_only_basic_bins() const
{ return cache_stores_only_basic_bins; }

void 
ProjMatrixByBin::
clear_cache() const
{
#ifdef STIR_OPENMP
#pragma omp critical(PROJMATRIXBYBINCLEARCACHE)
#endif
  for (int i=this->cache_collection.get_min_index();
       i<=this->cache_collection.get_max_index();
       ++i)
    {
      for (int j=this->cache_collection[i].get_min_index();
           j<=this->cache_collection[i].get_max_index();
           ++j)
        {
          this->cache_collection[i][j].clear();
        }
    }
}

/*
void  
ProjMatrixByBin::
reserve_num_elements_in_cache(const std::size_t num_elems)
{
  if ( cache_disabled ) return;
  //cache_collection.reserve(num_elems);
  cache_collection.rehash(ceil(num_elems / cache_collection.max_load_factor()));
}
*/

void
ProjMatrixByBin::
set_up(   
    const shared_ptr<const ProjDataInfo>& proj_data_info_sptr_v,
    const shared_ptr<const DiscretisedDensity<3,float> >& density_info_sptr_v // TODO should be Info only
    )
{
  this->proj_data_info_sptr = proj_data_info_sptr_v;
  this->image_info_sptr.reset(
                              dynamic_cast<const VoxelsOnCartesianGrid<float>* > (density_info_sptr_v->clone() ));
  if (is_cache_enabled())
    {
      const int max_abs_tangential_pos_num =
        std::max(-proj_data_info_sptr->get_min_tangential_pos_num(), proj_data_info_sptr->get_max_tangential_pos_num());
      // next isn't strictly speaking the max, as it could be larger for other segments, but that's pretty unlikely
      const int max_abs_axial_pos_num =
        std::max(-proj_data_info_sptr->get_min_axial_pos_num(0), proj_data_info_sptr->get_max_axial_pos_num(0));
      const int max_abs_timing_pos_num = std::max(-proj_data_info_sptr->get_min_tof_pos_num(), proj_data_info_sptr->get_max_tof_pos_num());

      if ((static_cast<CacheKey>(max_abs_axial_pos_num) >= (static_cast<CacheKey>(1) << axial_pos_bits)) ||
          (static_cast<CacheKey>(max_abs_tangential_pos_num) >= (static_cast<CacheKey>(1) << tang_pos_bits)) ||
          (static_cast<CacheKey>(max_abs_timing_pos_num) >= (static_cast<CacheKey>(1) << timing_pos_bits)))
        error("ProjMatrixByBin: not enough bits reserved for this data in the caching mechanism. You will have to switch caching off. Sorry.");
    }

  const int min_view_num = proj_data_info_sptr->get_min_view_num();
  const int max_view_num = proj_data_info_sptr->get_max_view_num();
  const int min_segment_num = proj_data_info_sptr->get_min_segment_num();
  const int max_segment_num = proj_data_info_sptr->get_max_segment_num();

  if (proj_data_info_sptr->is_tof_data())
	  enable_tof(proj_data_info_sptr,true);
  else
  {
	  tof_enabled = false;
  }

  this->cache_collection.recycle();
  this->cache_collection.resize(min_view_num, max_view_num);
#ifdef STIR_OPENMP
  this->cache_locks.recycle();
  this->cache_locks.resize(min_view_num, max_view_num);
#endif

  for (int view_num=min_view_num; view_num<=max_view_num; ++view_num)
    {
      this->cache_collection[view_num].resize(min_segment_num, max_segment_num);
#ifdef STIR_OPENMP
      this->cache_locks[view_num].resize(min_segment_num, max_segment_num);
      for (int seg_num = min_segment_num; seg_num <=max_segment_num; ++seg_num)
        omp_init_lock(&this->cache_locks[view_num][seg_num]);
#endif
    }

  // Setup the custom erf code
  erf_interpolation.set_num_samples(200000); //200,000 =~12.8MB
  erf_interpolation.set_up();
}


/*!
    \warning Preconditions
    <li>abs(axial_pos_num) fits in 13 (4095) bits
    <li>abs(tangential_pos_num) fits in 10 (1024) bits
    <li>abs(tof_pos_num) fits in 7 bits (127)
  */
ProjMatrixByBin::CacheKey
ProjMatrixByBin::cache_key(const Bin& bin) const
{
  assert(static_cast<CacheKey>(abs(bin.axial_pos_num())) < (static_cast<CacheKey>(1) << axial_pos_bits));
  assert(static_cast<CacheKey>(abs(bin.tangential_pos_num())) < (static_cast<CacheKey>(1) << tang_pos_bits));
  assert(static_cast<CacheKey>(abs(bin.timing_pos_num())) < (static_cast<CacheKey>(1) << timing_pos_bits));

  return static_cast<CacheKey>(
                               (static_cast<CacheKey>(bin.axial_pos_num()>=0?0:1) << (timing_pos_bits + tang_pos_bits + axial_pos_bits + 2))
                               | (static_cast<CacheKey>(abs(bin.axial_pos_num())) << (timing_pos_bits + tang_pos_bits + 2))
                               | (static_cast<CacheKey>(bin.tangential_pos_num()>=0?0:1) << (timing_pos_bits + tang_pos_bits + 1))
                               | (static_cast<CacheKey>(abs(bin.tangential_pos_num())) << (timing_pos_bits+1))
                               | (static_cast<CacheKey>(bin.timing_pos_num()>=0?0:1) << timing_pos_bits)
                               | (static_cast<CacheKey>(abs(bin.timing_pos_num()))));
} 


void  
ProjMatrixByBin::
cache_proj_matrix_elems_for_one_bin(
                                    const ProjMatrixElemsForOneBin& probabilities) const
{ 
  if ( cache_disabled ) return;
  
  //std::cerr << "cached lor size " << probabilities.size() << " capacity " << probabilities.capacity() << std::endl;    
  // insert probabilities into the collection	
  const Bin bin = probabilities.get_bin();
#ifdef STIR_OPENMP
  omp_set_lock(&this->cache_locks[bin.view_num()][bin.segment_num()]);
#endif
  cache_collection[bin.view_num()][bin.segment_num()].insert(MapProjMatrixElemsForOneBin::value_type( cache_key(bin), 
                                                                                                      probabilities));  
#ifdef STIR_OPENMP
  omp_unset_lock(&this->cache_locks[bin.view_num()][bin.segment_num()]);
#endif
}


Succeeded 
ProjMatrixByBin::
get_cached_proj_matrix_elems_for_one_bin(
                                         ProjMatrixElemsForOneBin& probabilities) const
{  
  if ( cache_disabled ) 
    return Succeeded::no;
  
  const Bin bin = probabilities.get_bin();

#ifndef NDEBUG
  if (cache_stores_only_basic_bins)
  {
    // Check that this is a 'basic' coordinate
    Bin bin_copy = bin; 
    assert ( symmetries_sptr->find_basic_bin(bin_copy) == 0);
  }
#endif         
  
  bool found=false;
#ifdef STIR_OPENMP
  omp_set_lock(&this->cache_locks[bin.view_num()][bin.segment_num()]);
#endif

  {
    const_MapProjMatrixElemsForOneBinIterator pos = 
      cache_collection[bin.view_num()][bin.segment_num()].find(cache_key( bin));
  
    if ( pos != cache_collection[bin.view_num()][bin.segment_num()]. end())
      { 
	//cout << Key << " =========>> entry found in cache " <<  endl;
	probabilities = pos->second;
	// note: cannot return from inside an OPENMP critical section
	//return Succeeded::yes;	
	found=true;
      } 
  }
#ifdef STIR_OPENMP
  omp_unset_lock(&this->cache_locks[bin.view_num()][bin.segment_num()]);
#endif
  if (found)
    return Succeeded::yes;	
  else
    {
      //cout << " This entry  is not in the cache :" << Key << endl;	
      return Succeeded::no;
    }
}



//TODO



//////////////////////////////////////////////////////////////////////////  
#if 0
// KT moved here
//! store the projection matrix to the file by rows 
void ProjMatrixByBin::write_to_file_by_bin(
                                      const char * const file_name_without_extension)
{ 
  char h_interfile[256];
  sprintf (h_interfile, "%s.hp", file_name_without_extension );
  FILE * prob_file = fopen (h_interfile , "wb");
  sprintf (h_interfile, "%s.p", file_name_without_extension );
  fstream pout;
  open_write_binary(pout, h_interfile);
  
  // KT tough ! write Symmetries to file!
  // scan_info ==> interfile header 
  
  int t, get_num_delta = 15;// todo change to scan_info.get_num_delta();
  pout.write( (char*)&get_num_delta, sizeof (int));
  t =  proj_data_info_ptr->get_num_views()/4;
  pout.write( (char*)& t,sizeof (int));
  t=  proj_data_info_ptr->get_num_tangential_poss()/2;
  pout.write( (char*)&t, sizeof (int));
  int max_z = image_info.get_max_z();
  pout.write( (char*)& max_z, sizeof(int));
  
  int nviews =  proj_data_info_ptr->get_num_views();
  pout.write( (char*)& nviews, sizeof(int));
  
  //float offset = offset_ring();	pout.write( (char*)& offset, sizeof(float));
  
  for ( int seg = 0; seg <= get_num_delta; ++seg)
    for ( int view = 0 ;view <= proj_data_info_ptr->get_num_views()/4;++view)  
      for ( int bin = 0 ;bin <=  proj_data_info_ptr->get_num_tangential_poss()/2;++bin)  
        for ( int ring = 0; ring <= 0 /*get_num_rings()*/ ;++ring) // only ring 0
        {	    
          ProjMatrixElemsForOneBin ProbForOneBin; 
          get_proj_matrix_elems_for_one_bin(
            ProbForOneBin, 
            seg, 
            view, 
            ring, 
            bin);  
          cout << " get_number_of_elements() " << ProbForOneBin. get_number_of_elements() << endl; 	   	   
          ProbForOneBin.write(pout); 
        }
        pout.close();   
        fclose(prob_file);
        cout << "End of write_to_file_by_bin " << endl; 
}


#endif


END_NAMESPACE_STIR
