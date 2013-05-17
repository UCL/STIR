//
// $Id$
//
/*
    Copyright (C) 2005 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2013-01-01 - $Date$, Kris Thielemans
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
#ifndef __stir_MultipleProjData__H__
#define __stir_MultipleProjData__H__

/*!
  \file
  \ingroup data_buildblock
  \brief Declaration of class stir::MultipleProjData
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"
//#include "stir/Scanner.h"
#include <vector>

START_NAMESPACE_STIR

class MultipleProjData
{
public:

  MultipleProjData() {};
#if 0
  MultipleProjData(const shared_ptr<Scanner>& scanner_sptr)
  {
    _scanner_sptr=scanner_sptr;
  }
#endif
  unsigned
    get_num_proj_data() const
  {
    return static_cast<unsigned>(this->_proj_datas.size());
  }


  //! resize to new number of projection data
  /*! This acts like std::vector::resize(), i.e. if the new size is smaller than the previous size,
    the last elements are deleted. If the new size is larger than the previous size, the new
    elements are assigned with null pointers. In the latter case, you need to use set_proj_data_sptr()
  */
  void resize(const unsigned new_size)
  {
    this->_proj_datas.resize(new_size);
  }

  //! set projection data for a particular index
  /*! \arg proj_data_sptr projection data (already fully initialised)
      \arg index number of the data (needs to be between 1 and get_num_proj_data())
  */
  void 
    set_proj_data_sptr(const shared_ptr<ProjData >& proj_data_sptr, 
		       const unsigned int index);
  /*!
    \warning The index starts from 1
  */
  const ProjData & 
    operator[](const unsigned int index) const 
    { 
      assert(index>=1);
      assert(index<= this->get_num_proj_data());
      return *this->_proj_datas[index-1]; 
    }
  /*!
    \warning The index starts from 1
  */
  const ProjData & 
    get_proj_data(const unsigned int index) const 
    { return (*this)[index]; }

  /*!
    \warning The index starts from 1
  */
  shared_ptr<ProjData> 
    get_proj_data_sptr(const unsigned int index) const 
    {
      assert(index>=1);
      assert(index<= this->get_num_proj_data());
      return this->_proj_datas[index-1]; 
    }

  const ProjDataInfo *
    get_proj_data_info_ptr() const;
  // return get_proj_data_sptr(1))->get_proj_data_info_ptr()

  /*! \deprecated */
  unsigned int get_num_gates() const
  {
    return static_cast<unsigned int>(_proj_datas.size());
  }

protected:
  std::vector<shared_ptr<ProjData > > _proj_datas;
  //shared_ptr<Scanner> _scanner_sptr;
};

END_NAMESPACE_STIR
#endif
