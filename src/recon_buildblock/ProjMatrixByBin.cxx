//
// $Id$: $Date$
//
/*!

  \file
  \ingroup recon_buildblock
  
  \brief  implementation of the ProjMatrixByBin class 
    
  \author Mustapha Sadki
  \author Kris Thielemans
  \author PARAPET project
      
  \date $Date$
        
  \version $Revision$
*/


#include "recon_buildblock/ProjMatrixByBin.h"
#include "recon_buildblock/ProjMatrixElemsForOneBin.h"
//#include "utilities.h"
//#include <iostream>

#ifndef TOMO_NO_NAMESPACES
//using std::cout;
//using std::endl;
#endif


START_NAMESPACE_TOMO

ProjMatrixByBin::ProjMatrixByBin()
{ 
  cache_disabled=false;
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
  // Check that this is a 'basic' coordinate
  Bin bin_copy = bin; 
  assert ( symmetries_ptr->find_basic_bin(bin_copy) == 0);     
#endif         
  
	 
  const_MapProjMatrixElemsForOneBinIterator  pos = 
    cache_collection.find(cache_key( bin));
  
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


END_NAMESPACE_TOMO
