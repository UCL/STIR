/*
    Copyright (C) 2020, University College London
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
  \ingroup recon_test
  \brief Test class for reconstructions
  \author Kris Thielemans
*/

#include "stir/RunTests.h"
#include "stir/Shape/EllipsoidalCylinder.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/ProjDataInMemory.h"
#include "stir/recon_buildblock/Reconstruction.h"
#include "stir/IO/read_from_file.h"
#include "stir/IO/write_to_file.h"
#include "stir/ArrayFunction.h"

START_NAMESPACE_STIR

/*!
  \ingroup recon_test
  \brief Base class for simple test on reconstruction
*/
template <class TargetT>
class ReconstructionTests : public RunTests
{
public:
  //! Constructor that can take some input data to run the test with
  explicit inline
    ReconstructionTests(const std::string &proj_data_filename = "",
                        const std::string & density_filename = "");

  virtual ~ReconstructionTests() {}

  //! creates input
  /*! sets \c _proj_data_sptr and \c _input_density_sptr from
    filenames or defaults if the filename is empty.

    \c _proj_data_sptr is constructed by forward projecting \c _input_density_sptr
  */
  virtual inline void construct_input_data();
  //! creates the reconstruction object
  /*! has to set \c _recon_sptr */
  virtual inline void construct_reconstructor() = 0;
  //! perform reconstruction
  virtual inline shared_ptr<TargetT> reconstruct();
  //! compares output and input
  /*! voxel-wise comparison */
  virtual inline void compare(const shared_ptr<const TargetT> output_sptr);

protected:
  std::string _proj_data_filename;
  std::string _input_density_filename;
  shared_ptr<ProjDataInMemory> _proj_data_sptr;
  shared_ptr<TargetT> _input_density_sptr;
  shared_ptr<Reconstruction<TargetT> > _recon_sptr;
};

template <class TargetT>
ReconstructionTests<TargetT>::
ReconstructionTests(const std::string &proj_data_filename,
                    const std::string & density_filename) :
    _proj_data_filename(proj_data_filename),
    _input_density_filename(density_filename)
{
}


template <class TargetT>
void
ReconstructionTests<TargetT>::
construct_input_data()
{ 
  if (this->_proj_data_filename.empty())
    {
      // construct a small scanner and sinogram
      shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));
      scanner_sptr->set_num_rings(5);
      shared_ptr<ProjDataInfo> proj_data_info_sptr(
        ProjDataInfo::ProjDataInfoCTI(scanner_sptr, 
                                      /*span=*/3, 
                                      /*max_delta=*/2,
                                      /*num_views=*/128,
                                      /*num_tang_poss=*/128));
      shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
      _proj_data_sptr.reset(new ProjDataInMemory (exam_info_sptr, proj_data_info_sptr));
    }
  else
    {
      shared_ptr<ProjData> proj_data_sptr =
        ProjData::read_from_file(this->_proj_data_filename);
      _proj_data_sptr.reset(new ProjDataInMemory (*proj_data_sptr));
    }

  if (this->_input_density_filename.empty())
    {
      CartesianCoordinate3D<float> origin (0,0,0);    
      const float zoom=1.F;
      
      shared_ptr<VoxelsOnCartesianGrid<float> >
        vox_sptr(new VoxelsOnCartesianGrid<float>(this->_proj_data_sptr->get_exam_info_sptr(),
                                                  *this->_proj_data_sptr->get_proj_data_info_sptr(),
                                                  zoom,origin));

      // create very long cylinder, such that we don't have to think about origin
      EllipsoidalCylinder cylinder(/*length_z*/1000.F,
                                   /*radius_y*/100.F,
                                   /*radius_x*/90.F,
                                   CartesianCoordinate3D<float>(0.F,0.F,0.F));
      cylinder.construct_volume(*vox_sptr, CartesianCoordinate3D<int>(2,2,2));
      this->_input_density_sptr = vox_sptr;
    }
  else
    {
      shared_ptr<TargetT> aptr(read_from_file<TargetT>(this->_input_density_filename));
      this->_input_density_sptr = aptr;
    }

  // forward project
  {
    shared_ptr<ProjMatrixByBin> PM_sptr(new ProjMatrixByBinUsingRayTracing);    
    shared_ptr<ForwardProjectorByBin> fwd_proj_sptr =
      MAKE_SHARED<ForwardProjectorByBinUsingProjMatrixByBin>(PM_sptr);
    fwd_proj_sptr->set_up(this->_proj_data_sptr->get_proj_data_info_sptr(),
                          this->_input_density_sptr);
    fwd_proj_sptr->set_input(*this->_input_density_sptr);
    fwd_proj_sptr->forward_project(*this->_proj_data_sptr);
  }
}

template <class TargetT>
shared_ptr<TargetT>
ReconstructionTests<TargetT>::
reconstruct()
{
  this->_recon_sptr->set_input_data(this->_proj_data_sptr);
  this->_recon_sptr->set_disable_output(true);
  if (this->_recon_sptr->set_up(this->_input_density_sptr)==Succeeded::no)
    error("recon::set_up() failed");
  
  shared_ptr<TargetT> output_sptr(this->_input_density_sptr->get_empty_copy());
  if (this->_recon_sptr->reconstruct(output_sptr)==Succeeded::no)
    error("recon::reconstruct() failed");
  return output_sptr;
}

template <class TargetT>
void
ReconstructionTests<TargetT>::
compare(const shared_ptr<const TargetT> output_sptr)
{
  shared_ptr<TargetT> diff_sptr(output_sptr->clone());
  *diff_sptr -= *this->_input_density_sptr;
  const float diff_min = diff_sptr->find_min();
  const float diff_max = diff_sptr->find_max();
  const float max_input = this->_input_density_sptr->find_max();
  in_place_abs(*diff_sptr);
  const float mean_abs_error=diff_sptr->sum() / this->_input_density_sptr->size_all();
  std::cerr << "Reconstruction diff relative range: "
            << "[" << diff_min/max_input << ", " << diff_max/max_input << "]\n"
            << "mean abs diff normalised was " << mean_abs_error/max_input << "\n";
  if (!check_if_less(-0.3F, diff_min/max_input, "relative diff min") ||
      !check_if_less(diff_max/max_input, .3F, "relative diff max") ||
      !check_if_less(mean_abs_error/max_input, .01F, "relative mean abs diff"))
    {
      write_to_file("test_recon_output.hv", *output_sptr);
      write_to_file("test_recon_original.hv", *this->_input_density_sptr);
      write_to_file("test_recon_diff.hv", *diff_sptr);
    }
}

END_NAMESPACE_STIR
