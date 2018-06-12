//
//
/*!
  \file
  \ingroup ClearPET_utilities
  \brief Preliminary code to handle listmode events 
    
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd

    This software is distributed under the terms 
    of the GNU Lesser General  Public Licence (LGPL)
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_CListEventDataLMF_H__
#define __stir_listmode_CListEventDataLMF_H__

//! Class for storing and using a coincidence event from a listmode file
/*! \ingroup ClearPET_utilities
 */
class CListEventDataLMF 
{
 public:  
  inline bool is_prompt() const { return true; } // TODO
  inline Succeeded set_prompt(const bool prompt = true) // TODO
  { return Succeeded::no; }
  
  inline LORAs2Points<float> get_LOR() const
    { return this->lor; }


  CartesianCoordinate3D<float> pos1() const
    { return lor.p1(); }
  CartesianCoordinate3D<float>& pos1()
    { return lor.p1(); }
  CartesianCoordinate3D<float> pos2() const
    { return lor.p2(); }
  CartesianCoordinate3D<float>& pos2()
    { return lor.p1(); }

 private:
  LORAs2Points<float> lor;
}; /*-coincidence event*/ 

END_NAMESPACE_STIR

#endif
