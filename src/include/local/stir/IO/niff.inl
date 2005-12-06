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

#include "stir/BasicCoordinate.h"
#include "stir/IndexRange.h"


namespace GE_IO {
  


  //
  // Fill the specified array with pixel/voxel data from the Niff object.
  // Note that the number of dimensions of the array must match the number
  // of dimensions of the Niff object.
  // If the fill succeeds, true is returned.
  //
  template<int num_dimensions> inline bool 
  Niff::fill_array(stir::Array<num_dimensions, float>& data) const 
    throw (std::out_of_range, std::ios_base::failure) {

    bool status = false;

    // Check that the number of dimension of the array is correct.
    if ( num_dimensions == _num_dimensions ) {

      // Resize the Array to accomodate the niff data.

      stir::BasicCoordinate<num_dimensions, int> dimension_sizes;

      for(unsigned int dim = 0 ; dim < num_dimensions ; ++dim) {
        dimension_sizes[dim + 1] = this->get_data_size(dim);
      }

      const stir::IndexRange<num_dimensions> index_range(dimension_sizes);

      data.resize(index_range);


      // Copy consecutive pixels directly from niff to Array.

      long total_pixels = this->get_total_pixels();

      typename stir::Array<num_dimensions, float>::full_iterator 
        data_iter = data.begin_all();

      for(long pix_index = 0 ; pix_index < total_pixels ; ++pix_index, ++data_iter) {
        *data_iter = this->pixel(pix_index);
      }

    }

    return status;
  }
  


  
  // Fill the specified array with pixel/voxel data from the Niff object.
  // Note that the number of dimensions of the array must match the number
  // of dimensions of the Niff object.
  //
  template<int num_dimensions> inline bool 
  Niff::fill_array(stir::Array<num_dimensions, double>& data) const
    throw (std::out_of_range, std::ios_base::failure) {
    
    bool status = false;
    
    // Check that the number of dimension of the array is correct.
    if ( num_dimensions == _num_dimensions ) {
      
      // Resize the Array to accomodate the niff data.
      
      stir::BasicCoordinate<num_dimensions, int> dimension_sizes;

      for(unsigned int dim = 0 ; dim < num_dimensions ; ++dim) {
        dimension_sizes[dim + 1] = this->get_data_size(dim);
      }
      
      const stir::IndexRange<num_dimensions> index_range(dimension_sizes);
      
      data.resize(index_range);

      
      // Copy consecutive pixels directly from niff to Array.

      long total_pixels = this->get_total_pixels();
      
      typename stir::Array<num_dimensions, double>::full_iterator 
        data_iter = data.begin_all();
      
      for(long pix_index = 0 ; pix_index < total_pixels ; ++pix_index, ++data_iter) {
        *data_iter = this->pixel(pix_index);
      }
      
    }
    
    return status;
  }


  
  
  
  //
  // Fill the specified array with pixel/voxel data from the Niff object.
  // Note that the number of dimensions of the array must match the number
  // of dimensions of the Niff object.
  //
  // This method assumes the supplied array is already correctly sized.
  // It allows the index ranges to be varied.
  //
  // If the fill succeeds, true is returned.
  //
  
  template<int num_dimensions> inline bool 
  Niff::fill_prepared_array(stir::Array<num_dimensions, float>& data) const 
    throw (std::out_of_range, std::ios_base::failure) {
  
    bool status = false;

    // Check that the number of dimension of the array is correct.
    if ( num_dimensions == _num_dimensions ) {
      
      // Copy consecutive pixels directly from niff to Array.
      long total_pixels = this->get_total_pixels();
      
      typename stir::Array<num_dimensions, float>::full_iterator 
        data_iter = data.begin_all();

      for(long pix_index = 0 ; pix_index < total_pixels ; ++pix_index, ++data_iter) {
        *data_iter = this->pixel(pix_index);
      }
      
    }
    
    return status;
  }
  


  
  // Fill the specified array with pixel/voxel data from the Niff object.
  // Note that the number of dimensions of the array must match the number
  // of dimensions of the Niff object.
  //
  // This method assumes the supplied array is already correctly sized.
  // It allows the index ranges to be varied.
  //
  // If the fill succeeds, true is returned.
  //
  template<int num_dimensions> inline bool 
  Niff::fill_prepared_array(stir::Array<num_dimensions, double>& data) const
    throw (std::out_of_range, std::ios_base::failure) {
    
    bool status = false;
    
    // Check that the number of dimension of the array is correct.
    if ( num_dimensions == _num_dimensions ) {
      
      // Copy consecutive pixels directly from niff to Array.

      long total_pixels = this->get_total_pixels();
      
      typename stir::Array<num_dimensions, double>::full_iterator 
        data_iter = data.begin_all();
      
      for(long pix_index = 0 ; pix_index < total_pixels ; ++pix_index, ++data_iter) {
        *data_iter = this->pixel(pix_index);
      }
      
    }
    
    return status;
  }

  


    
  
} // End of namespace.
