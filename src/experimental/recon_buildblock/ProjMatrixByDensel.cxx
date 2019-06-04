//
//
/*!

  \file
  \ingroup recon_buildblock
  
  \brief  implementation of the ProjMatrixByDensel class 
    
  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project
      
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2004, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/


#include "stir_experimental/recon_buildblock/ProjMatrixByDensel.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneDensel.h"
#include "stir/Succeeded.h"
//#include <iostream>

#ifndef STIR_NO_NAMESPACES
//using std::cout;
//using std::endl;
#endif


START_NAMESPACE_STIR

ProjMatrixByDensel::ProjMatrixByDensel()
{ 
  cache_disabled=false;
}
 
Succeeded 
ProjMatrixByDensel::
get_cached_proj_matrix_elems_for_one_densel(
                                         ProjMatrixElemsForOneDensel& probabilities) const

                                                         
                                                     
{  
  if ( cache_disabled ) 
    return Succeeded::no;
  
  const Densel densel = probabilities.get_densel();

#ifndef NDEBUG
  // Check that this is a 'basic' coordinate
  Densel densel_copy = densel; 
  assert ( !get_symmetries_ptr()->find_basic_densel(densel_copy));     
#endif         
  
	 
  const_MapProjMatrixElemsForOneDenselIterator  pos = 
    cache_collection.find(cache_key( densel));
  
  if ( pos != cache_collection. end())
  { 
    //cout << Key << " =========>> entry found in cache " <<  endl;
    probabilities = pos->second;	 	                    
    return Succeeded::yes;	
  } 
  
  //cout << " This entry  is not in the cache :" << Key << endl;	
  return Succeeded::no;
}



//TODO



//////////////////////////////////////////////////////////////////////////  
#if 0
// KT moved here
//! store the projection matrix to the file by rows 
void ProjMatrixByDensel::write_to_file_by_densel(
                                      const char * const file_name_without_extension)
{ 
  char h_interfile[256];
  sprintf (h_interfile, "%s.hp", file_name_without_extension );
  FILE * prob_file = fopen (h_interfile , "wb");
  sprintf (h_interfile, "%s.p", file_name_without_extension );
  fstream pout;
  open_write_denselary(pout, h_interfile);
  
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
      for ( int densel = 0 ;densel <=  proj_data_info_ptr->get_num_tangential_poss()/2;++densel)  
        for ( int ring = 0; ring <= 0 /*get_num_rings()*/ ;++ring) // only ring 0
        {	    
          ProjMatrixElemsForOneDensel ProbForOneDensel; 
          get_proj_matrix_elems_for_one_densel(
            ProbForOneDensel, 
            seg, 
            view, 
            ring, 
            densel);  
          cout << " get_number_of_elements() " << ProbForOneDensel. get_number_of_elements() << endl; 	   	   
          ProbForOneDensel.write(pout); 
        }
        pout.close();   
        fclose(prob_file);
        cout << "End of write_to_file_by_densel " << endl; 
}


#endif


END_NAMESPACE_STIR
