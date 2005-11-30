/*
 * $Id$
 *
 *
 * $Date$
 *
 * $Revision$
 *
 *
 */


namespace GE_IO {


  //
  // Fill the specified array with pixel/voxel data from the Niff object.
  // Note that the number of dimensions of the array must match the number
  // of dimensions of the Niff object.
  // If the fill succeeds, true is returned.
  //
  
  template<unsigned int num_dimensions> bool 
  Niff::fill_array(stir::Array<num_dimensions, float>& data) const 
    throw (std::out_of_range, std::ios_base::failure) {
  
    bool status = false;

    // Check that the number of dimension of the array is correct.
    if ( num_dimensions == _num_dimensions ) {

      
      


    }
    

    return status;
  }
  
  
  // Fill the specified array with pixel/voxel data from the Niff object.
  // Note that the number of dimensions of the array must match the number
  // of dimensions of the Niff object.
  
  template<unsigned int num_dimensions> bool 
  Niff::fill_array(stir::Array<num_dimensions, double>& data) const
    throw (std::out_of_range, std::ios_base::failure) {
    
    bool status = false;

    return status;
    
  }


        
  
} // End of namespace.
