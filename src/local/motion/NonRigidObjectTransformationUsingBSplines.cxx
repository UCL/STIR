//
// $Id$
//
/*
    Copyright (C) 2005- $Date$ , Hammersmith Imanet Ltd
    For internal GE use only
*/
/*!
  \file
  \ingroup motion

  \brief Implementation of class stir::NonRigidObjectTransformationUsingBSplines

  \author  Kris Thielemans
  $Date$
  $Revision$
*/

#include "local/stir/motion/NonRigidObjectTransformationUsingBSplines.h"
#include "stir/stream.h"//xxx
#include "stir/numerics/determinant.h"
#include "stir/IndexRange2D.h"
#include <iostream>

// for ncat
#include <string>
#include <cstring>
#include "stir/CartesianCoordinate3D.h"
#include <fstream>
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"

// for binary file
#include "stir/VoxelsOnCartesianGrid.h"

START_NAMESPACE_STIR

template <>
const char * const 
NonRigidObjectTransformationUsingBSplines<3,float>::registered_name = "BSplines transformation"; 


//////////// functions for reading NCAT transformations ///////////////////////
// (shouldn't be in this file)
static
Succeeded
parse_line(const string& deformation_field_from_NCAT_file,
	   istream& ncat_file, 
	   CartesianCoordinate3D<int>& current_voxel,
	   CartesianCoordinate3D<float>& current_displacement)
{
  std::string line;
  std::getline(ncat_file, line);
  if (!ncat_file)
    {
      warning("Error reading line in NCAT file %s", deformation_field_from_NCAT_file.c_str());
      return Succeeded::no;
    }
  const std::string::size_type position =
    line.find("FRAME");
  CartesianCoordinate3D<float> new_voxel_coords;
  CartesianCoordinate3D<float> current_voxel_coords;
  int frame_num1, frame_num2;
  if (
      std::sscanf(line.c_str() + position,
		  "FRAME%d %f %f %f FRAME%d %f %f %f VECTOR %f %f %f",
		  &frame_num1,
		  &current_voxel_coords.x(), &current_voxel_coords.y(), &current_voxel_coords.z(),
		  &frame_num2,
		  &new_voxel_coords.x(), &new_voxel_coords.y(), &new_voxel_coords.z(),
		  &current_displacement.x(), &current_displacement.y(), &current_displacement.z())
      != 11)
    {
      warning("Error parsing line in NCAT file %s:\n\"%s\"\nstart position %d\ntext to parse:\n%s", 
	      deformation_field_from_NCAT_file.c_str(),
	      line.c_str(),
	      position,
	      line.c_str() + position);
      return Succeeded::no;
    }
  if (norm(new_voxel_coords - current_voxel_coords - current_displacement) > .1)
    {
      warning("Error in line in NCAT file %s: inconsistent coordinates\n\"%s\"", 
	      deformation_field_from_NCAT_file.c_str(),
	      line.c_str());
      std::cerr << new_voxel_coords << current_voxel_coords << current_displacement
		<< new_voxel_coords - current_voxel_coords - current_displacement << std::endl;
      return Succeeded::no;
    }
    current_voxel = round(current_voxel_coords);
    if (norm(BasicCoordinate<3,float>(current_voxel) - current_voxel_coords) > .01)
    {
      warning("Error in line in NCAT file %s: ORIG voxel coordinates are expected to be on the grid\n\"%s\"", 
	      deformation_field_from_NCAT_file.c_str(),
	      line.c_str());
      return Succeeded::no;
    }
    return Succeeded::yes;
}

static  
Succeeded
set_deformation_field_from_NCAT_file(DeformationFieldOnCartesianGrid<3,float>& deformation_field,
				     const string& deformation_field_from_NCAT_file,
				     const CartesianCoordinate3D<int>& image_size,
				     const CartesianCoordinate3D<float>& grid_spacing,
				     const CartesianCoordinate3D<float>& origin)
{
  std::ifstream ncat_file(deformation_field_from_NCAT_file.c_str());
  if (!ncat_file)
    {
      warning("Error opening NCAT file %s", deformation_field_from_NCAT_file.c_str());
      return Succeeded::no;
    }
  // skip first line
  {
    std::string line;
    std::getline(ncat_file, line);
  }
  // allocate deformation_field
  deformation_field[1].grow(IndexRange<3>(image_size));
  deformation_field[2].grow(IndexRange<3>(image_size));
  deformation_field[3].grow(IndexRange<3>(image_size));
  // note: constructor sets all deformations to 0

  std::cerr << "\nstart parsing  NCAT" << std::endl;
  CartesianCoordinate3D<int> current_voxel;
  CartesianCoordinate3D<float> current_displacement;
  while (ncat_file)
    {
      if (parse_line(deformation_field_from_NCAT_file,
		     ncat_file, 
		     current_voxel,
		     current_displacement)
	  != Succeeded::yes)
	return Succeeded::no;
      const CartesianCoordinate3D<float> current_displacement_in_mm = 
	current_displacement * grid_spacing;
      if (current_voxel[1] < deformation_field[1].get_min_index() ||
	  current_voxel[1] > deformation_field[1].get_max_index() ||
	  current_voxel[2] < deformation_field[1][current_voxel[1]].get_min_index() ||
	  current_voxel[2] > deformation_field[1][current_voxel[1]].get_max_index() ||
	  current_voxel[3] < deformation_field[1][current_voxel[1]][current_voxel[2]].get_min_index() ||
	  current_voxel[3] > deformation_field[1][current_voxel[1]][current_voxel[2]].get_max_index())
	{
	  std::cerr << "\nCoordinates out of range : " << current_voxel << current_displacement << std::endl;
	  return Succeeded::no;
	}
      deformation_field[1][current_voxel] = current_displacement_in_mm.z();
      deformation_field[2][current_voxel] = current_displacement_in_mm.y();
      deformation_field[3][current_voxel] = current_displacement_in_mm.x();
      if (ncat_file.eof())
	break;
    }
  std::cerr << "\nend parsing  NCAT" << std::endl;
  return Succeeded::yes;
}
/////////////// end of NCAT parsing stuff //////////////


/////////////// binary /////////////////////////////////
static 
Succeeded
set_deformation_field_from_file(DeformationFieldOnCartesianGrid<3,float>& deformation_field,
				const std::string& deformation_field_from_file_x,
				const std::string& deformation_field_from_file_y,
				const std::string& deformation_field_from_file_z,
				CartesianCoordinate3D<float>& grid_spacing,
				CartesianCoordinate3D<float>& origin)
{
  shared_ptr<DiscretisedDensity<3,float> > image_sptr =
    DiscretisedDensity<3,float>::read_from_file(deformation_field_from_file_z);
  if (is_null_ptr(image_sptr))
    {
      warning("Error reading %s", deformation_field_from_file_z.c_str());
      return Succeeded::no;
    }
  VoxelsOnCartesianGrid<float> const * voxels_ptr =
    dynamic_cast<VoxelsOnCartesianGrid<float> const *>(image_sptr.get());
  if (is_null_ptr(voxels_ptr))
    {
      warning("Error reading %s: should be of type VoxelsOnCartesianGrid", deformation_field_from_file_z.c_str());
      return Succeeded::no;
    }
  deformation_field[1] = *image_sptr;
  origin = image_sptr->get_origin();
  grid_spacing = voxels_ptr->get_grid_spacing();

    DiscretisedDensity<3,float>::read_from_file(deformation_field_from_file_y);
  if (is_null_ptr(image_sptr))
    {
      warning("Error reading %s", deformation_field_from_file_y.c_str());
      return Succeeded::no;
    }
  deformation_field[2] = *image_sptr;

    DiscretisedDensity<3,float>::read_from_file(deformation_field_from_file_x);
  if (is_null_ptr(image_sptr))
    {
      warning("Error reading %s", deformation_field_from_file_x.c_str());
      return Succeeded::no;
    }
  deformation_field[3] = *image_sptr;

  return Succeeded::yes;
}

template <int num_dimensions, class elemT>
void
NonRigidObjectTransformationUsingBSplines<num_dimensions,elemT>::
set_defaults()
{
  this->_bspline_type = BSpline::cubic;
}


template <int num_dimensions, class elemT>
void 
NonRigidObjectTransformationUsingBSplines<num_dimensions,elemT>::
initialise_keymap()
{
  this->deformation_field_sptr = new DeformationFieldOnCartesianGrid<num_dimensions,elemT>;
  this->parser.add_key("grid spacing", &this->_grid_spacing);
  this->parser.add_key("origin", &this->_origin);
  this->parser.add_key("deformation field", this->deformation_field_sptr.get());
  this->parser.add_key("deformation field from NCAT file", &this->_deformation_field_from_NCAT_file);
  this->parser.add_key("deformation field from NCAT x-size", &this->_deformation_field_from_NCAT_size.x());
  this->parser.add_key("deformation field from NCAT y-size", &this->_deformation_field_from_NCAT_size.y());
  this->parser.add_key("deformation field from NCAT z-size", &this->_deformation_field_from_NCAT_size.z());

  this->parser.add_key("deformation field from file x-component", &this->_deformation_field_from_file_x);
  this->parser.add_key("deformation field from file y-component", &this->_deformation_field_from_file_y);
  this->parser.add_key("deformation field from file z-component", &this->_deformation_field_from_file_z);
  this->parser.add_key("bspline order", &this->_bspline_order);
  this->parser.add_start_key("BSplines Transformation Parameters");
  this->parser.add_stop_key("End BSplines Transformation Parameters");
}

template <int num_dimensions, class elemT>
void 
NonRigidObjectTransformationUsingBSplines<num_dimensions,elemT>::
set_key_values()
{
  // TODO following is only correct if bspline coefficients are equal to samples
  for (int i=1; i<=num_dimensions; ++i)
    (*this->deformation_field_sptr)[i] =
      this->interpolator[i].get_coefficients();

  this->_bspline_order = static_cast<int>(this->_bspline_type);
}
    
template <int num_dimensions, class elemT>
bool 
NonRigidObjectTransformationUsingBSplines<num_dimensions,elemT>::
post_processing()
{
  this->_bspline_type = static_cast<BSpline::BSplineType>(this->_bspline_order);

  if ((*this->deformation_field_sptr)[1].size()==0)
    {
      if (this->_deformation_field_from_NCAT_file.size() == 0)
	{
	  if (this->_deformation_field_from_file_z.size() == 0)
	    {
	      warning("NonRigidObjectTransformationUsingBSplines:\n"
		      "you need to set either deformation_field or deformation_field_from_NCAT_file");
	      return false;
	    }
	  else
	    {
	      if (set_deformation_field_from_file(*(this->deformation_field_sptr),
						  this->_deformation_field_from_file_x,
						  this->_deformation_field_from_file_y,
						  this->_deformation_field_from_file_z,
						  this->_grid_spacing,
						  this->_origin) 
		  == Succeeded::no)
		return false;
	}

	}
      else
	{
	  if (set_deformation_field_from_NCAT_file(*(this->deformation_field_sptr),
						   this->_deformation_field_from_NCAT_file,
						   this->_deformation_field_from_NCAT_size,
						   this->_grid_spacing,
						   this->_origin) 
	      == Succeeded::no)
	    return false;
	}
    }

  std::cerr << "\nStarting to compute interpolators";
  for (int i=1; i<=num_dimensions; ++i)
    this->interpolator[i] = 
      BSpline::BSplinesRegularGrid<num_dimensions,elemT,elemT>((*this->deformation_field_sptr)[i],this->_bspline_type);
  std::cerr << "\nDone computing interpolators";
  // deallocate data for deformation field
  // at present, have to do this by assigning an object as opposed to 0
  // in case we want to parse twice
  // WARNING: do not reassign a new pointer, as the keymap stores a pointer to the deformation_field object
  *(this->deformation_field_sptr)  = DeformationFieldOnCartesianGrid<num_dimensions,elemT>();

  return false;
}


template <int num_dimensions, class elemT>
NonRigidObjectTransformationUsingBSplines<num_dimensions,elemT>::
NonRigidObjectTransformationUsingBSplines()
{
  this->set_defaults();
}


template <int num_dimensions, class elemT>
BasicCoordinate<num_dimensions,elemT>
NonRigidObjectTransformationUsingBSplines<num_dimensions,elemT>::
transform_point(const BasicCoordinate<num_dimensions,elemT>& point) const
{
  // note: current Bspline needs double here
  const BasicCoordinate<num_dimensions,double> point_in_grid_coords =
    BasicCoordinate<num_dimensions,double>((point - this->_origin)/this->_grid_spacing);
  BasicCoordinate<num_dimensions,elemT> result;
  for (int i=1; i<=num_dimensions; ++i)
    result[i]= this->interpolator[i](point_in_grid_coords);
  return result + point;
}

template <int num_dimensions, class elemT>
float
NonRigidObjectTransformationUsingBSplines<num_dimensions,elemT>::
jacobian(const BasicCoordinate<num_dimensions,elemT>& point) const
{
  // note: current Bspline needs double here
  const BasicCoordinate<num_dimensions,double> point_in_grid_coords =
    BasicCoordinate<num_dimensions,double>((point - this->_origin)/this->_grid_spacing);
  Array<2,float> jacobian_matrix(IndexRange2D(1,num_dimensions,1,num_dimensions));
  for (int i=1; i<=num_dimensions; ++i)
    {
      BasicCoordinate<num_dimensions,elemT> gradient =
	this->interpolator[i].gradient(point_in_grid_coords)/
	this->_grid_spacing;
      gradient[i] += 1; // take into account that we're only modelling deformation.
      std::copy(gradient.begin(), gradient.end(), jacobian_matrix[i].begin());
    }
  return 
    determinant(jacobian_matrix);
}

////////////////////// instantiations
template class NonRigidObjectTransformationUsingBSplines<3,float>;
END_NAMESPACE_STIR
