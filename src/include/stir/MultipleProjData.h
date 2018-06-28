/*
    Copyright (C) 2005 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2013, Kris Thielemans
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
#ifndef __stir_MultipleProjData__H__
#define __stir_MultipleProjData__H__

/*!
  \file
  \ingroup data_buildblock
  \brief Declaration of class stir::MultipleProjData
  \author Kris Thielemans
*/
#include "stir/ProjData.h"
#include "stir/IO/ExamData.h"
#include "stir/shared_ptr.h"
#include "stir/Array.h"
#include "stir/is_null_ptr.h"
//#include "stir/Scanner.h"
#include <vector>

START_NAMESPACE_STIR

class MultipleProjData : public ExamData
{
public:

  MultipleProjData():ExamData() {};

  MultipleProjData(const shared_ptr<ExamInfo>& exam_info_sptr)
    :ExamData(exam_info_sptr)
  {
  }

  //!
  //! \brief MultipleProjData
  //! \param exam_info_sptr
  //! \param num_gates
  //! \author Nikos Efthimiou
  //! \details Convinience constructor which sets the number of gates.
  //! \warning The _proj_datas have been resized, but are still empty.
  MultipleProjData(const shared_ptr<ExamInfo>& exam_info_sptr,
                   const int num_gates);

  static
  shared_ptr<MultipleProjData>
  read_from_file(const std::string &parameter_file);

  //N.E.14/07/16 Inherited from ExamData
  // //! Get a pointer to the exam information
//  const ExamInfo*
//    get_exam_info_ptr() const
//  {
//    return this->_exam_info_sptr.get();
//  }

//  //! Get a shared pointer to the exam information
//  shared_ptr<ExamInfo>
//    get_exam_info_sptr() const
//  {
//    return this->_exam_info_sptr;
//  }

  unsigned
    get_num_proj_data() const
  {
    return static_cast<unsigned>(this->_proj_datas.size());
  }

  //!
  //! \brief get_projData_size
  //! \return The size of the projdata[0]
  //!
  std::size_t get_proj_data_size() const
  {
      return _proj_datas.at(0)->size_all();
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

  //!
  //! \brief copy_to
  //! \param full_iterator of some array
  //! \details Copy all data to an array.
  //! \author Nikos Efthimiou
  //! \warning Full::iterator should be supplied.
  template < typename iterT>
  void copy_to(iterT array_iter)
  {
      for ( std::vector<shared_ptr<ProjData> >::iterator it = _proj_datas.begin();
            it != _proj_datas.end(); ++it)
      {
          if ( is_null_ptr( *(it)))
              error("Dynamic ProjData have not been properly allocated.Abort.");

          const std::size_t num_bins = (*it)->copy_to(array_iter);
          std::advance(array_iter, num_bins);
      }
  }

  //!
  //! \brief fill_from
  //! \param full_iterator of some array
  //! \details Fills all ProjData from a 2D array.
  //! \author Nikos Efthimiou
  //! \warning Full::iterator should be supplied.
  template <typename iterT>
  void fill_from(iterT array_iter)
  {
      long int cur_pos = 0;
      for (std::vector<shared_ptr<ProjData> >::iterator it = _proj_datas.begin();
           it != _proj_datas.end(); ++it)
      {
          if ( is_null_ptr( *(it)))
              error("Dynamic ProjData have not been properly allocated.Abort.");

          cur_pos = (*it)->fill_from(array_iter);
          std::advance(array_iter, cur_pos);
      }
  }

  //!
  //! \brief size_all
  //! \return
  //! \author Nikos Efthimiou
  //! \details Returns the total size of the object
  std::size_t size_all() const
  {
      std::size_t size = 0;
      for (std::size_t i_gate = 0; i_gate < this->get_num_gates(); i_gate++)
          size += _proj_datas.at(i_gate)->size_all();

      return size;
  }

protected:
  std::vector<shared_ptr<ProjData > > _proj_datas;
  //shared_ptr<Scanner> _scanner_sptr;
 protected:
  //N.E:14/07/16 Inherited from ExamData.
//  shared_ptr<ExamInfo> _exam_info_sptr;
};

END_NAMESPACE_STIR
#endif
