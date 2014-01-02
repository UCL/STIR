/*
    Copyright (C) 2000-2009, Hammersmith Imanet Ltd
    Copyright (C) 2013, University College London
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
  \ingroup numerics
  \brief implements the IR_filters

  \author Charalampos Tsoumpas
  \author Kris Thielemans
*/

START_NAMESPACE_STIR

template <class RandIter1,
          class RandIter2,
          class RandIter3,
          class RandIter4>
void 
inline 
IIR_filter(RandIter1 output_begin_iterator, 
           RandIter1 output_end_iterator,
           const RandIter2 input_begin_iterator, 
           const RandIter2 input_end_iterator,
           const RandIter3 input_factor_begin_iterator,
           const RandIter3 input_factor_end_iterator,
           const RandIter4 pole_begin_iterator,
           const RandIter4 pole_end_iterator,
           const bool if_initial_exists)
{
  // The input should be initialised to 0
  //    if(output_begin_iterator==output_end_iterator)
  //            warning("No output signal is given./n");
#if 1
  if(if_initial_exists==false) 
    *output_begin_iterator=(*input_begin_iterator)*(*input_factor_begin_iterator);
#else
  // an attempt to remove warnings by VC++, but it doesn't work for higher-dimensional arrays
  typedef typename CastScalarForOperation<typename std::iterator_traits<RandIter1>::value_type>::type cast_type;

  if(if_initial_exists==false) 
    *output_begin_iterator=(*input_begin_iterator)*static_cast<cast_type>(*input_factor_begin_iterator);
#endif

  RandIter1 current_output_iterator = output_begin_iterator ;
  RandIter2 current_input_iterator = input_begin_iterator ;
        
  for(++current_output_iterator, ++current_input_iterator; 
      current_output_iterator != output_end_iterator &&
        current_input_iterator != input_end_iterator;
      ++current_output_iterator,        ++current_input_iterator)
    {
      RandIter2 current_current_input_iterator = current_input_iterator;
      for(RandIter3 current_input_factor_iterator = input_factor_begin_iterator;                                
          current_input_factor_iterator != input_factor_end_iterator;
          ++current_input_factor_iterator,--current_current_input_iterator
          )
        {
#if 1
          (*current_output_iterator) += 
            (*current_current_input_iterator) *
            (*current_input_factor_iterator);
#else
          (*current_output_iterator) += 
            (*current_current_input_iterator) *
                                  static_cast<cast_type>(*current_input_factor_iterator);
#endif
          if (current_current_input_iterator==input_begin_iterator)
            break;
        }

      RandIter4 current_pole_iterator = pole_begin_iterator;
      RandIter1 current_feedback_iterator = current_output_iterator ;
                        
      for(--current_feedback_iterator ;
          current_pole_iterator != pole_end_iterator                    
            ;++current_pole_iterator,--current_feedback_iterator)
        {                                                       
          (*current_output_iterator) -= 
#if 1
            (*current_feedback_iterator) *
            (*current_pole_iterator);
#else
            (*current_feedback_iterator) *
                                  static_cast<cast_type>(*current_pole_iterator);
#endif
          if(current_feedback_iterator==output_begin_iterator)
            break;
        }                                       
    }
}

template <class RandIter1,
          class RandIter2,
          class RandIter3>               
void 
inline 
FIR_filter(RandIter1 output_begin_iterator, 
           RandIter1 output_end_iterator,
           const RandIter2 input_begin_iterator, 
           const RandIter2 input_end_iterator,
           const RandIter3 input_factor_begin_iterator,
           const RandIter3 input_factor_end_iterator,
           const bool if_initial_exists)
{
  IIR_filter(output_begin_iterator, output_end_iterator, 
             input_begin_iterator, input_end_iterator, 
             input_factor_begin_iterator, input_factor_end_iterator,
             output_begin_iterator, output_begin_iterator,if_initial_exists);
}

END_NAMESPACE_STIR
