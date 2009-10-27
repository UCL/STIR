//
// $Id$
//
/*
  Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
#ifndef __stir_numerics_BSplines1DRegularGrid__H__
#define __stir_numerics_BSplines1DRegularGrid__H__

/*!
\file 
\ingroup BSpline
\brief Implementation of the B-Splines Interpolation 

\author Charalampos Tsoumpas
\author Kris Thielemans
  
$Date$
$Revision$
*/

#include "stir/numerics/BSplines.h"
#include <vector>

START_NAMESPACE_STIR

namespace BSpline 
{
                
  /*! \ingroup BSpline
    \brief Temporary class for 1D B-splines
        
    This class works with std::vector objects, while BSplinesRegularGrid works with 
    stir::Array objects.
    \todo remove overlap with the n-dimensional version BSplinesRegularGrid 
  */
  template <typename out_elemT, typename in_elemT=out_elemT>
    class BSplines1DRegularGrid
    {
      private:
      typedef typename std::vector<out_elemT>::iterator RandIterOut; 
      int input_size; // create in the constructor 
      double z1;
      double z2;
      double lambda;
        
        
      inline
      out_elemT
      BSplines_product(const int index, const pos_type relative_position, const bool if_deriv) const;

      inline
      void
      set_private_values(BSplineType);

      inline 
      out_elemT
      compute_BSplines_value(const pos_type relative_position, const bool if_deriv) const;

      // sadly,VC6.0 needs definition of template members in the class definition
      template <class IterT>
      inline
      void
      set_coef(IterT input_begin_iterator, IterT input_end_iterator)
      {         
        BSplines1DRegularGrid::input_size = input_end_iterator - input_begin_iterator;
        BSplines_coef_vector.resize(input_size);
        BSplines_coef(BSplines_coef_vector.begin(),BSplines_coef_vector.end(), 
                      input_begin_iterator, input_end_iterator, z1, z2, lambda);                                
      }
        
      public:
      std::vector<out_elemT> BSplines_coef_vector;  
      BSplineType spline_type;
        
        
      //! default constructor: no input
      inline BSplines1DRegularGrid();
        
      //! constructor given a vector as an input, estimates the Coefficients 
      inline explicit BSplines1DRegularGrid(const std::vector<in_elemT> & input_vector);
                
      //! constructor given a begin_ and end_ iterator as input, estimates the Coefficients 
      template <class IterT>
      inline BSplines1DRegularGrid(const IterT input_begin_iterator, 
                                   const IterT input_end_iterator)
      {  
        set_private_values(cubic);              
        set_coef(input_begin_iterator, input_end_iterator);
      }
      //! constructor given a begin_ and end_ iterator as input, estimates the Coefficients 
      template <class IterT>
      inline BSplines1DRegularGrid(const IterT input_begin_iterator, 
                                   const IterT input_end_iterator, const BSplineType this_type)
      {  
        set_private_values(this_type);          
        set_coef(input_begin_iterator, input_end_iterator);
      }  
        
      inline
      BSplines1DRegularGrid(const std::vector<in_elemT> & input_vector, const BSplineType this_type); 
        
      //! destructor
      inline ~BSplines1DRegularGrid();
        
      inline 
      out_elemT
      BSplines(const pos_type relative_position) const;
        
      inline 
      out_elemT
      BSplines_1st_der(const pos_type relative_position) const;
        
      //! same as BSplines()
      inline
      const out_elemT 
      operator() (const pos_type relative_position) const;
        
      inline
      const std::vector<out_elemT> 
      BSplines_output_sequence(RandIterOut output_relative_position_begin_iterator,  //relative_position might be better float
                               RandIterOut output_relative_position_end_iterator);
      inline
      const std::vector<out_elemT> 
      BSplines_output_sequence(std::vector<pos_type> output_relative_position);
    };
} // end BSpline namespace

END_NAMESPACE_STIR

#include "stir/numerics/BSplines1DRegularGrid.inl"

#endif
