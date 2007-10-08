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

#include "local/stir/modelling/ParametricDiscretisedDensity.h"
#include "local/stir/modelling/KineticParameters.h"
#include "boost/lambda/lambda.hpp"

#include "local/stir/DynamicDiscretisedDensity.h"

START_NAMESPACE_STIR

//#define TEMPLATE template <int num_dimensions, typename KinParsT>
#define TEMPLATE template <typename DiscDensityT>
//#define ParamDiscDensity ParametricDiscretisedDensity<num_dimensions, KinParsT>
#define ParamDiscDensity ParametricDiscretisedDensity<DiscDensityT>
#define NUM_PARAMS 2

/////////////////////////////////////////////////////////////////////////////////////


TEMPLATE
ParamDiscDensity::
ParametricDiscretisedDensity(const VectorWithOffset<SingleDiscretisedDensityType> &  densities)
{
  using namespace boost::lambda;

  assert(densities.size()==NUM_PARAMS); // Warning: NUM_PARAMS is not defined as unsigned integer
  for (unsigned f=1; f<= NUM_PARAMS; ++f)
    {
      assert(densities[1].get_grid_spacing()==densities[f].get_grid_spacing());
      assert(densities[1].get_index_range()==densities[f].get_index_range());
      assert(densities[1].get_origin()==densities[f].get_origin());
    }
  // somewhat naughty trick to get elemT of DiscDensityT
  typedef typename DiscDensityT::full_value_type KinParsT;
  CartesianCoordinate3D<float> grid_spacing =
    dynamic_cast<const VoxelsOnCartesianGrid<float>&>(densities[1]).get_grid_spacing();
  // TODO this will only work for VoxelsOnCartesianGrid
  ParamDiscDensity * parametric_density_ptr =
    new ParamDiscDensity(DiscDensityT(densities[1].get_index_range(),
				      densities[1].get_origin(), 
				      grid_spacing));
  
  for (unsigned f=1; f<= NUM_PARAMS; ++f)
    {
      const SingleDiscretisedDensityType& current_density =
	dynamic_cast<SingleDiscretisedDensityType const&>(densities[f]);
      typename SingleDiscretisedDensityType::const_full_iterator single_density_iter =
	current_density.begin_all();
      const typename SingleDiscretisedDensityType::const_full_iterator end_single_density_iter =
	current_density.end_all();
      typename ParamDiscDensity::full_densel_iterator parametric_density_iter =
	parametric_density_ptr->begin_all_densel();

      while (single_density_iter!=end_single_density_iter)
	{	  (*parametric_density_iter)[f] = *single_density_iter;
	  ++single_density_iter; ++parametric_density_iter;
	}
    } 
}

TEMPLATE
ParamDiscDensity
ParamDiscDensity::
create_parametric_image(const VectorWithOffset<shared_ptr<SingleDiscretisedDensityType> > &  densities) const
{
  using namespace boost::lambda;

  assert(densities.size()==NUM_PARAMS); // Warning: NUM_PARAMS is not defined as unsigned integer
  for (unsigned f=1; f<= NUM_PARAMS; ++f)
    {
      assert(densities[1]->get_grid_spacing()==densities[f]->get_grid_spacing());
      assert(densities[1]->get_index_range()==densities[f]->get_index_range());
      assert(densities[1]->get_origin()==densities[f]->get_origin());
    }
  // somewhat naughty trick to get elemT of DiscDensityT
  typedef typename DiscDensityT::full_value_type KinParsT;
  shared_ptr<ParamDiscDensity> parametric_density_ptr = this->clone();

  for (unsigned f=1; f<= NUM_PARAMS; ++f)
    {
      const SingleDiscretisedDensityType& current_density =
	dynamic_cast<SingleDiscretisedDensityType const&>(*densities[f]);
      typename SingleDiscretisedDensityType::const_full_iterator single_density_iter =
	current_density.begin_all();
      const typename SingleDiscretisedDensityType::const_full_iterator end_single_density_iter =
	current_density.end_all();
      typename ParamDiscDensity::full_densel_iterator parametric_density_iter =
	parametric_density_ptr->begin_all_densel();

      while (single_density_iter!=end_single_density_iter)
	{	  (*parametric_density_iter)[f] = *single_density_iter;
	  ++single_density_iter; ++parametric_density_iter;
	}
    } 
  return *parametric_density_ptr;
}

TEMPLATE
ParamDiscDensity
ParamDiscDensity::
update_parametric_image(const VectorWithOffset<shared_ptr<SingleDiscretisedDensityType> > &  densities)
{
  using namespace boost::lambda;

  assert(densities.size()==NUM_PARAMS); // Warning: NUM_PARAMS is not defined as unsigned integer
  for (unsigned f=1; f<= NUM_PARAMS; ++f)
    {
      assert(densities[1]->get_grid_spacing()==densities[f]->get_grid_spacing());
      assert(densities[1]->get_index_range()==densities[f]->get_index_range());
      assert(densities[1]->get_origin()==densities[f]->get_origin());
    }
  // somewhat naughty trick to get elemT of DiscDensityT
  typedef typename DiscDensityT::full_value_type KinParsT;

  for (unsigned f=1; f<= NUM_PARAMS; ++f)
    {
      const SingleDiscretisedDensityType& current_density =
	dynamic_cast<SingleDiscretisedDensityType const&>(*densities[f]);
      typename SingleDiscretisedDensityType::const_full_iterator single_density_iter =
	current_density.begin_all();
      const typename SingleDiscretisedDensityType::const_full_iterator end_single_density_iter =
	current_density.end_all();
      typename ParamDiscDensity::full_densel_iterator parametric_density_iter =
	this->begin_all_densel();

      while (single_density_iter!=end_single_density_iter)
	{	  (*parametric_density_iter)[f] = *single_density_iter;
	  ++single_density_iter; ++parametric_density_iter;
	}
    } 
  return *this;
}

TEMPLATE
ParamDiscDensity
ParamDiscDensity::
update_parametric_image(const shared_ptr<SingleDiscretisedDensityType> &  single_density_sptr, const unsigned int param_num)
{
  using namespace boost::lambda;

  assert(param_num<=NUM_PARAMS); // Warning: NUM_PARAMS is not defined as unsigned integer

  // somewhat naughty trick to get elemT of DiscDensityT
  typedef typename DiscDensityT::full_value_type KinParsT;

  const unsigned int f=param_num;
  const SingleDiscretisedDensityType& current_density =
    dynamic_cast<SingleDiscretisedDensityType const&> (*single_density_sptr);
  typename SingleDiscretisedDensityType::const_full_iterator single_density_iter =
    current_density.begin_all();
  const typename SingleDiscretisedDensityType::const_full_iterator end_single_density_iter =
    current_density.end_all();
  typename ParamDiscDensity::full_densel_iterator parametric_density_iter =
    this->begin_all_densel();
  
  while (single_density_iter!=end_single_density_iter)
    {	  
      (*parametric_density_iter)[f] = *single_density_iter;
      ++single_density_iter; ++parametric_density_iter;
    } 
  return *this;
}

#if 0
// ChT::ToDo: make a class which will update the single parameter, using a reference to the output...
TEMPLATE
typename ParamDiscDensity::SingleDiscretisedDensityType &
ParamDiscDensity::
update_parametric_image(const unsigned int param_num)
{
  using namespace boost::lambda;

  assert(param_num<=NUM_PARAMS);

  // somewhat naughty trick to get elemT of DiscDensityT
  typedef typename DiscDensityT::full_value_type KinParsT;

  const unsigned int f=param_num;
  SingleDiscretisedDensityType & current_density =
    dynamic_cast<SingleDiscretisedDensityType const&> (this->construct_single_density(f));
  typename SingleDiscretisedDensityType::const_full_iterator single_density_iter =
    current_density.begin_all();
  const typename SingleDiscretisedDensityType::const_full_iterator end_single_density_iter =
    current_density.end_all();
  typename ParamDiscDensity::full_densel_iterator parametric_density_iter =
    this->begin_all_densel();
  
  while (single_density_iter!=end_single_density_iter)
    {	  
      (*parametric_density_iter)[f] = *single_density_iter;
      ++single_density_iter; ++parametric_density_iter;
    } 
  return current_density;
}
#endif

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
    dynamic_cast<const VoxelsOnCartesianGrid<float>&>((*multi_sptr)[1]).get_grid_spacing();
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
  ParamDiscDensity * res =
    this->clone();
  std::fill(res->begin_all(), res->end_all(), 0);
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


#undef NUM_PARAMS
#undef ParamDiscDensity
#undef TEMPLATE

// template class ParametricDiscretisedDensity<3,KineticParameters<NUM_PARAMS,float> >; 
 template class ParametricDiscretisedDensity<ParametricVoxelsOnCartesianGridBaseType >; 

END_NAMESPACE_STIR


