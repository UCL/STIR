//
//
#ifndef __stir_modelling_ParametricDiscretisedDensity_H__
#define __stir_modelling_ParametricDiscretisedDensity_H__
/*
    Copyright (C) 2006 - 2011, Hammersmith Imanet Ltd
    Copyright (C) 2019, University College London
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
  \ingroup modelling
  \brief Declaration of class stir::ParametricDiscretisedDensity
  \author Kris Thielemans
  \author Richard Brown
 
*/

#include "stir/DiscretisedDensity.h"
#include "stir/NestedIterator.h"
// for ParametricVoxelsOnCartesianGrid typedef 
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/modelling/KineticParameters.h"
START_NAMESPACE_STIR
template <typename DiscDensT>
class ParametricDiscretisedDensity;

/// Forward declaration of dynamic image
class DynamicDiscretisedDensity;

//! A helper class to find the type of a 'single' image for a corresponding parametric image.
template <typename DiscDensT>
struct Parametric2Single;

template <typename DiscDensT>
struct Parametric2Single<ParametricDiscretisedDensity<DiscDensT> >
{
  typedef typename Parametric2Single<DiscDensT>::type type;
};

template <int num_parameters, typename elemT>
struct Parametric2Single<VoxelsOnCartesianGrid<KineticParameters<num_parameters, elemT> > >
{
  typedef VoxelsOnCartesianGrid<elemT> type;
};

//! Class to store parametric images
/*! \ingroup modelling
  \param DiscDensT should be a class for storing an image, e.g. VoxelsOnCartesianGrid<KineticParameters<..> 

  This class stores parametric images as an image of parameters (as opposed to a sequence of images, one
  for each parameter). However, it provides various functions to get single images out, e.g. corresponding
  to a single parameter, or by applying a function on all parameters.
*/
template <typename DiscDensT>
class ParametricDiscretisedDensity:
public DiscDensT
{
 private:
  //  typedef DiscretisedDensity<num_dimensions, KinParsT> base_type;
  typedef DiscDensT base_type;
 public:
  //! A typedef that can be used what the base of the hierarchy is
  /*! For these purposes, we don't use DiscDensT (even though it's the base_type).

      This typedef is used in write_to_file().
  */
  typedef ParametricDiscretisedDensity hierarchy_base_type;

  typedef typename Parametric2Single<DiscDensT>::type SingleDiscretisedDensityType;

  typedef typename base_type::full_iterator full_densel_iterator;
  typedef typename base_type::const_full_iterator const_full_densel_iterator;

  typedef NestedIterator<full_densel_iterator, BeginEndFunction<full_densel_iterator> > full_iterator;
  typedef NestedIterator<const_full_densel_iterator, ConstBeginEndFunction<const_full_densel_iterator> >
    const_full_iterator;

  //! A static member to read an image from file
  static ParametricDiscretisedDensity * read_from_file(const std::string& filename);

  //! Get number of parameters in a single densel
  static unsigned int
    get_num_params();

  ParametricDiscretisedDensity(const base_type& density)
    : base_type(density)
    {}

  /// Create blank parametric image from a dynamic image
  ParametricDiscretisedDensity(const DynamicDiscretisedDensity& dyn_im);

  // implementation works, although only for VoxelsOnCartesianGrid , but not needed for now
  // ParametricDiscretisedDensity(const VectorWithOffset<shared_ptr<SingleDiscretisedDensityType> > & densities);

  // implementation works, but not needed for now
  // void update_parametric_image(const VectorWithOffset<shared_ptr<SingleDiscretisedDensityType> > &  densities);

  void update_parametric_image(const SingleDiscretisedDensityType &  single_density, const unsigned int param_num);

  full_iterator begin_all()
    { return full_iterator(base_type::begin_all(), base_type::end_all()); }

  const_full_iterator begin_all_const() const
    { return const_full_iterator(base_type::begin_all_const(), base_type::end_all_const()); }

  full_iterator end_all()
    { return full_iterator(base_type::end_all(), base_type::end_all()); }

  const_full_iterator end_all_const() const
    { return const_full_iterator(base_type::end_all_const(), base_type::end_all_const()); }

  const_full_iterator begin_all() const
    { return this->begin_all_const(); }

  const_full_iterator end_all() const
    { return this->end_all_const(); }

  full_densel_iterator begin_all_densel()
    { return base_type::begin_all(); }

  const_full_densel_iterator begin_all_densel_const() const
    { return base_type::begin_all_const(); }

  full_densel_iterator end_all_densel()
    { return base_type::end_all(); }

  const_full_densel_iterator end_all_densel_const() const
    { return base_type::end_all_const(); }

  const_full_densel_iterator begin_all_densel() const
    { return this->begin_all_densel_const(); }

  const_full_densel_iterator end_all_densel() const
    { return this->end_all_densel_const(); }

  //! Allocate a new object with same characteristics as the current one.
  virtual ParametricDiscretisedDensity* get_empty_copy() const;

  //! Allocate a new object which is a copy of the current one.
  virtual ParametricDiscretisedDensity* clone() const;

  //! construct a single image by applying a function object on each KineticParameter 
  template <class KPFunctionObject>
    void 
    construct_single_density_using_function(SingleDiscretisedDensityType& density, KPFunctionObject f) const;

  template <class KPFunctionObject>
    const SingleDiscretisedDensityType
    construct_single_density_using_function(KPFunctionObject f) const;

  //! construct a single image corresponding to the parameter with index \c i
   //@{
  void 
    construct_single_density(SingleDiscretisedDensityType& density, const int i) const;

  const SingleDiscretisedDensityType
    construct_single_density(const int index) const;
#if 0   //!< Implementation of non-const functions - which should be able to update a single parameter of a parametric image.

  template <class KPFunctionObject>
    void 
    construct_single_density_using_function(SingleDiscretisedDensityType& density, KPFunctionObject f) ;
  
  template <class KPFunctionObject>
    SingleDiscretisedDensityType &
    construct_single_density_using_function(KPFunctionObject f) ;
  
  void 
    construct_single_density(SingleDiscretisedDensityType& density, const int i) ;
  
  SingleDiscretisedDensityType & // Maybe this should be a reference...
    construct_single_density(const int index);
#endif
   //@}
};

//! Convenience typedef for base-type of Cartesian Voxelised Parametric Images with just two parameters 
typedef VoxelsOnCartesianGrid<KineticParameters<2,float> >
           ParametricVoxelsOnCartesianGridBaseType;


//! Convenience typedef for Cartesian Voxelised Parametric Images with just two parameters 
typedef ParametricDiscretisedDensity<ParametricVoxelsOnCartesianGridBaseType>
   ParametricVoxelsOnCartesianGrid;


END_NAMESPACE_STIR
//#include "stir/modelling/ParametricDiscretisedDensity.inl"

#endif //__stir_modelling_ParametricDiscretisedDensity_H__
