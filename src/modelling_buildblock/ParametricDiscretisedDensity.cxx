//
// $Id$
//
/*
    Copyright (C) 2006 - $Date$, Hammersmith Imanet Ltd
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
 
  $Date$
  $Revision$
*/

#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/KineticParameters.h"
#include "boost/lambda/lambda.hpp"
#include "stir/DynamicDiscretisedDensity.h"
#include <iostream>

START_NAMESPACE_STIR

//#define TEMPLATE template <int num_dimensions, typename KinParsT>
#define TEMPLATE template <typename DiscDensityT>
//#define ParamDiscDensity ParametricDiscretisedDensity<num_dimensions, KinParsT>
#define ParamDiscDensity ParametricDiscretisedDensity<DiscDensityT>

/////////////////////////////////////////////////////////////////////////////////////
TEMPLATE
unsigned int
ParamDiscDensity::
get_num_params()
{
  // somewhat naughty trick to get elemT of DiscDensityT
  typedef typename DiscDensityT::full_value_type KinParsT;
  const KinParsT dummy;
  return dummy.size();
}

#if 0
// implementation works, although only for VoxelsOnCartesianGrid , but not needed for now
TEMPLATE
ParamDiscDensity::
ParametricDiscretisedDensity(const VectorWithOffset<shared_ptr<SingleDiscretisedDensityType> > &  densities)
  // TODO this will only work for VoxelsOnCartesianGrid
  :    base_type(densities[1]->get_index_range(),
                 densities[1]->get_origin(), 
                 densities[1]->get_grid_spacing())
{

  assert(densities.size()==this->get_num_params()); 
  for (unsigned f=1; f<= this->get_num_params(); ++f)
    {
      assert(densities[1]->get_grid_spacing()==densities[f]->get_grid_spacing());
      assert(densities[1]->get_index_range()==densities[f]->get_index_range());
      assert(densities[1]->get_origin()==densities[f]->get_origin());
    }
  
  for (unsigned f=1; f<= this->get_num_params(); ++f)
    {
#if 1
      // for some reason, the following gives a segmentation fault in gcc 4.1 optimised mode.
      // Maybe because we're calling a member function in the constructor?
      this->update_parametric_image(*densities[f], f);
#else
      // alternative (untested)
      const SingleDiscretisedDensityType& current_density =
        dynamic_cast<SingleDiscretisedDensityType const&>(*densities[f]);
      typename SingleDiscretisedDensityType::const_full_iterator single_density_iter =
        current_density.begin_all();
      const typename SingleDiscretisedDensityType::const_full_iterator end_single_density_iter =
        current_density.end_all();
      typename ParamDiscDensity::full_densel_iterator parametric_density_iter =
        this->begin_all_densel();

      while (single_density_iter!=end_single_density_iter)
        {         (*parametric_density_iter)[f] = *single_density_iter;
          ++single_density_iter; ++parametric_density_iter;
        }
#endif
    } 
}
#endif

#if 0
// implementation works, but not needed for now
TEMPLATE
void
ParamDiscDensity::
update_parametric_image(const VectorWithOffset<shared_ptr<SingleDiscretisedDensityType> > &  densities)
{
  assert(densities.size()==this->get_num_params());
  for (unsigned f=1; f<= this->get_num_params(); ++f)
    {
      assert(densities[1]->get_grid_spacing()==densities[f]->get_grid_spacing());
      assert(densities[1]->get_index_range()==densities[f]->get_index_range());
      assert(densities[1]->get_origin()==densities[f]->get_origin());
    }
  for (unsigned f=1; f<= this->get_num_params(); ++f)
    {
      this->update_parametric_image(*densities[f], f);
    } 

}
#endif

TEMPLATE
void
ParamDiscDensity::
update_parametric_image(const SingleDiscretisedDensityType &  single_density, const unsigned int param_num)
{
  assert(param_num<=this->get_num_params()); 
  assert(single_density.get_index_range() == this->get_index_range());

  const unsigned int f=param_num;
  typename SingleDiscretisedDensityType::const_full_iterator single_density_iter =
    single_density.begin_all();
  const typename SingleDiscretisedDensityType::const_full_iterator end_single_density_iter =
    single_density.end_all();
  typename ParamDiscDensity::full_densel_iterator parametric_density_iter =
    this->begin_all_densel();
  while (single_density_iter!=end_single_density_iter)
    {     
      if (parametric_density_iter == this->end_all_densel())
        error("update ITER");
      //(*parametric_density_iter)[f] = *single_density_iter;
      const float tmp = *single_density_iter;
      (*parametric_density_iter)[f] = tmp;
      ++single_density_iter; ++parametric_density_iter;
    } 
  // TODO Currently need this to avoid segmentation fault with 4.1...
  // std::cerr << " Done\n";
}

TEMPLATE
ParamDiscDensity *
ParamDiscDensity::
read_from_file(const std::string& filename)
{
  // TODO this will only work for elemT==float
  shared_ptr<DynamicDiscretisedDensity > multi_sptr =
    DynamicDiscretisedDensity::read_from_file(filename);

  using namespace boost::lambda;
  
  // somewhat naughty trick to get elemT of DiscDensityT
  typedef typename DiscDensityT::full_value_type KinParsT;

  // check size
  {
    KinParsT dummy;
    const unsigned num_pars = dummy.size();

    if (num_pars != multi_sptr->get_num_time_frames())
      error("I expect %d 'time frames' when reading %s. Exiting",
            num_pars, filename.c_str());
  }

  if (dynamic_cast<const VoxelsOnCartesianGrid<float> * >(&(*multi_sptr)[1])==0)
    error("ParametricDiscretisedDensity::read_from_file only supports VoxelsOnCartesianGrid");

  CartesianCoordinate3D<float> grid_spacing =
    static_cast<const VoxelsOnCartesianGrid<float> *>(&(*multi_sptr)[1])->get_grid_spacing();
  // TODO this will only work for VoxelsOnCartesianGrid
  ParamDiscDensity * parametric_density_ptr =
    new ParamDiscDensity(DiscDensityT((*multi_sptr)[1].get_index_range(),
                                      (*multi_sptr)[1].get_origin(), 
                                      grid_spacing));
  
  for (unsigned f=1; f<= multi_sptr->get_num_time_frames(); ++f)
    {
      const SingleDiscretisedDensityType& current_density =
        dynamic_cast<SingleDiscretisedDensityType const&>((*multi_sptr)[f]);
      typename SingleDiscretisedDensityType::const_full_iterator single_density_iter =
        current_density.begin_all();
      const typename SingleDiscretisedDensityType::const_full_iterator end_single_density_iter =
        current_density.end_all();
      typename ParamDiscDensity::full_densel_iterator parametric_density_iter =
        parametric_density_ptr->begin_all_densel();

      while (single_density_iter!=end_single_density_iter)
        {
          (*parametric_density_iter)[f] = *single_density_iter;
          ++single_density_iter; ++parametric_density_iter;
        }
    }
  return parametric_density_ptr;
}


TEMPLATE
ParamDiscDensity *
ParamDiscDensity::
get_empty_copy() const
{
  // TODO maybe this can be done smarter by using base_type::get_empty_copy. Doesn't matter too much though.
  ParamDiscDensity * res = this->clone();
  typename ParamDiscDensity::iterator parametric_density_iter =
    res->begin();
  while (parametric_density_iter!=res->end())
    {
      assign(*parametric_density_iter++, 0);
    }
  return res;
}

TEMPLATE
ParamDiscDensity *
ParamDiscDensity::
clone() const
{
  return new ParamDiscDensity(*this);
}

TEMPLATE
template <class KPFunctionObject>
void 
ParamDiscDensity::
construct_single_density_using_function(typename ParamDiscDensity::SingleDiscretisedDensityType& density, KPFunctionObject f) const
{
  std::transform(this->begin_all_densel(),
                 this->end_all_densel(),
                 density.begin_all(),
                 f);
}
                 

TEMPLATE
template <class KPFunctionObject>
const typename ParamDiscDensity::SingleDiscretisedDensityType
ParamDiscDensity::
construct_single_density_using_function(KPFunctionObject f) const
{
  // TODO this will only work for VoxelsOnCartesianGrid
  SingleDiscretisedDensityType
    density(this->get_index_range(),
            this->get_origin(), 
            this->get_grid_spacing());
  this->construct_single_density_using_function(density, f);
  return density;
}

TEMPLATE
void 
ParamDiscDensity::
construct_single_density(typename ParamDiscDensity::SingleDiscretisedDensityType& density, const int index) const
{
  using namespace boost::lambda;

  // TODO this will only work for elemT==float
  this->construct_single_density_using_function(density, ret<float>(_1[index]));
}
                 

TEMPLATE
const typename ParamDiscDensity::SingleDiscretisedDensityType
ParamDiscDensity::
construct_single_density(const int index) const
{
  using namespace boost::lambda;
  // TODO this will only work for elemT==float
  return this->construct_single_density_using_function(ret<float>(_1[index]));
}

///////////////////////////////
#if 0   //!< Implementation of non-const functions - which should be able to update a single parameter of a parametric image.
TEMPLATE
template <class KPFunctionObject>
void 
ParamDiscDensity::
construct_single_density_using_function(typename ParamDiscDensity::SingleDiscretisedDensityType& density, KPFunctionObject f) 
{
  std::transform(this->begin_all_densel(),
                 this->end_all_densel(),
                 density.begin_all(),
                 f);
}
                 

TEMPLATE
template <class KPFunctionObject>
typename ParamDiscDensity::SingleDiscretisedDensityType &
ParamDiscDensity::
construct_single_density_using_function(KPFunctionObject f) 
{
  // TODO this will only work for VoxelsOnCartesianGrid
  SingleDiscretisedDensityType
    density(this->get_index_range(),
            this->get_origin(), 
            this->get_grid_spacing());
  this->construct_single_density_using_function(density, f);
  return density;
}

TEMPLATE
void 
ParamDiscDensity::
construct_single_density(typename ParamDiscDensity::SingleDiscretisedDensityType& density, const int index) 
{
  using namespace boost::lambda;

  // TODO this will only work for elemT==float
  this->construct_single_density_using_function(density, ret<float>(_1[index]));
}

TEMPLATE
typename ParamDiscDensity::SingleDiscretisedDensityType &
ParamDiscDensity::
construct_single_density(const int index) 
{
  using namespace boost::lambda;
  // TODO this will only work for elemT==float
  return this->construct_single_density_using_function(ret<float>(_1[index]));
}

#endif 


#undef ParamDiscDensity
#undef TEMPLATE
// instantiations

// template class ParametricDiscretisedDensity<3,KineticParameters<NUM_PARAMS,float> >;
 template class ParametricDiscretisedDensity<ParametricVoxelsOnCartesianGridBaseType >; 

END_NAMESPACE_STIR


