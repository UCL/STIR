/*
    Copyright (C) 2005 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2013, Kris Thielemans
    Copyright (C) 2013, 2016-2020 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
#include "stir/ExamData.h"
#include "stir/shared_ptr.h"
#include "stir/Array.h"
#include "stir/is_null_ptr.h"
#include "stir/copy_fill.h"
//#include "stir/Scanner.h"
#include <vector>

START_NAMESPACE_STIR

class MultipleProjData : public ExamData
{
public:

  MultipleProjData():ExamData() {}

  MultipleProjData(const shared_ptr<const ExamInfo>& exam_info_sptr)
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
  MultipleProjData(const shared_ptr<const ExamInfo>& exam_info_sptr,
                   const int num_gates);

  static
  unique_ptr<MultipleProjData>
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

  const shared_ptr<const ProjDataInfo>
    get_proj_data_info_sptr() const;
  // return get_proj_data_sptr(1))->get_proj_data_info_sptr()

  /*! \deprecated */
  unsigned int get_num_gates() const
  {
    return static_cast<unsigned int>(_proj_datas.size());
  }

  //!
  //! \brief copy_to
  //! \param array_iter an iterator to an array or other object (which has to be pre-allocated)
  //! \details Copy all data by incrementing \c array_iter.
  //! \author Nikos Efthimiou
  template < typename iterT>
  iterT copy_to(iterT array_iter) const
  {
    for ( std::vector<shared_ptr<ProjData> >::const_iterator it = _proj_datas.begin();
            it != _proj_datas.end(); ++it)
      {
          if ( is_null_ptr( *(it)))
              error("Dynamic/gated ProjData have not been properly allocated. Abort.");

          array_iter = stir::copy_to(*(*it), array_iter);
      }
      return array_iter;
  }

  //!
  //! \brief fill_from
  //! \param array_iter output iterator, e.g. of some array
  //! \details Fills all ProjData from the iterator (which has to fit the size)
  //! \author Nikos Efthimiou
  template <typename iterT>
  void fill_from(iterT array_iter)
  {
      for (std::vector<shared_ptr<ProjData> >::iterator it = _proj_datas.begin();
           it != _proj_datas.end(); ++it)
      {
          if ( is_null_ptr( *(it)))
              error("Dynamic ProjData have not been properly allocated.Abort.");

          array_iter = (*it)->fill_from(array_iter);
      }
  }

  //!
  //! \brief Returns the total number of elements in the object
  //! \author Nikos Efthimiou
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


//! Copy all bins to a range specified by an iterator
/*! 
  \ingroup copy_fill
  \return \a iter advanced over the range (as std::copy)
  
  \warning there is no range-check on \a iter
*/
template<>
struct CopyFill<MultipleProjData>
{ template < typename iterT>
    static
iterT copy_to(const MultipleProjData& stir_object, iterT iter)
{
  //std::cerr<<"Using MultipleProjData::copy_to\n";
  return stir_object.copy_to(iter);
}
};

//! set all elements of a MultipleProjData  from an iterator
/*!  
   Implementation that resorts to MultipleProjData::fill_from
   \warning there is no size/range-check on \a iter
*/
template < typename iterT>
void fill_from(MultipleProjData& stir_object, iterT iter, iterT /*iter_end*/)
{
  return stir_object.fill_from(iter);
}

END_NAMESPACE_STIR
#endif
