/*!

  \file
  \ingroup projection
  
  \brief  implementation of the stir::ProjMatrixByBin class 
    
  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2009, Hammersmith Imanet Ltd
    Copyright (C) 2013, 2015 University College London

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


#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"

// define a local preprocessor symbol to keep code relatively clean
#ifdef STIR_NO_MUTABLE
#define STIR_MUTABLE_CONST
#else
#define STIR_MUTABLE_CONST const
#endif

START_NAMESPACE_STIR

void ProjMatrixByBin::set_defaults()
{
  cache_disabled=false;
  cache_stores_only_basic_bins=true;
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
clear_cache() STIR_MUTABLE_CONST
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
    const shared_ptr<ProjDataInfo>& proj_data_info_sptr,
    const shared_ptr<DiscretisedDensity<3,float> >& /*density_info_ptr*/ // TODO should be Info only
    )
{
  const int min_view_num = proj_data_info_sptr->get_min_view_num();
  const int max_view_num = proj_data_info_sptr->get_max_view_num();
  const int min_segment_num = proj_data_info_sptr->get_min_segment_num();
  const int max_segment_num = proj_data_info_sptr->get_max_segment_num();

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
}


/*!
    \warning Preconditions
    <li>abs(axial_pos_num) fits in 17 bits
    <li>abs(tangential_pos_num) fits in 11 bits   
  */
ProjMatrixByBin::CacheKey
ProjMatrixByBin::cache_key(const Bin& bin) const
{
  assert(static_cast<boost::uint32_t>(abs(bin.axial_pos_num())) < (static_cast<boost::uint32_t>(1)<<18));
  assert(abs(bin.tangential_pos_num()) < (1<<12));
  return (CacheKey)( 
                    (static_cast<boost::uint32_t>(bin.axial_pos_num()>=0?0:1) << 31)
                    | (static_cast<boost::uint32_t>(bin.axial_pos_num())<<13) 
                    | (static_cast<boost::uint32_t>(bin.tangential_pos_num()>=0?0:1) << 12)
                    |  static_cast<boost::uint32_t>(abs(bin.tangential_pos_num())) );    	
} 


void  
ProjMatrixByBin::
cache_proj_matrix_elems_for_one_bin(
                                    const ProjMatrixElemsForOneBin& probabilities) STIR_MUTABLE_CONST
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
