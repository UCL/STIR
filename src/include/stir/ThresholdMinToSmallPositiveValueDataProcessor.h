//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup DataProcessor  
  \brief Declaration of class stir::ThresholdMinToSmallPositiveValueDataProcessor
    
  \author Kris Thielemans
      
*/

#ifndef __stir_ThresholdMinToSmallPositiveValueDataProcessor_H__
#define __stir_ThresholdMinToSmallPositiveValueDataProcessor_H__


#include "stir/RegisteredParsingObject.h"
#include "stir/DataProcessor.h"


START_NAMESPACE_STIR

/*!
  \ingroup DataProcessor  
  \brief A class in the DataProcessor hierarchy for making sure all elements are strictly positive.

  Works by calling threshold_min_to_small_positive_value().
  
  As it is derived from RegisteredParsingObject, it implements all the 
  necessary things to parse parameter files etc.

  \par Parsing parameters

  ; if part of a larger parameter file, you'd probably have something like
  ; data processor type := Chained Data Processor
  Chained Data Processor Parameters:=
    Data Processor to apply first:= data_processor type
    ; parameters for first data processor
    Data Processor to apply second:=data_processor type
    ; parameters for second data processor
  END Chained Data Processor Parameters:=

 */

template <typename DataT>
class ThresholdMinToSmallPositiveValueDataProcessor : 
  public 
    RegisteredParsingObject<
        ThresholdMinToSmallPositiveValueDataProcessor<DataT>,
        DataProcessor<DataT>,
        DataProcessor<DataT>
    >
{
private:
  typedef 
    RegisteredParsingObject<
        ThresholdMinToSmallPositiveValueDataProcessor<DataT>,
        DataProcessor<DataT>,
        DataProcessor<DataT>
    >
    base_type;
public:
  static const char * const registered_name; 
  
  //! Construct by calling set_defaults()
  ThresholdMinToSmallPositiveValueDataProcessor();
    
  
private:
  
  int rim_truncation_image;
  
  virtual void set_defaults();
  virtual void initialise_keymap();
  
  Succeeded virtual_set_up(const DataT&);

  void  virtual_apply(DataT& out_data, const DataT& in_data) const;
  void  virtual_apply(DataT& data) const ;
  
};


END_NAMESPACE_STIR

#endif


