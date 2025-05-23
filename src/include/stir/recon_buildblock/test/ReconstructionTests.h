/*
    Copyright (C) 2020, University College London
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0
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
#include "stir/SeparableGaussianImageFilter.h"
#include "stir/Verbosity.h"

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

  //! default proj_data_info
  virtual inline std::unique_ptr<ProjDataInfo>
    construct_default_proj_data_info_uptr() const;

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
  /*! Uses \c target_sptr as initialisation, and updates it
      \see Reconstruction::reconstruct(shared_ptr<TargetT>&)
  */
  virtual inline void reconstruct(shared_ptr<TargetT> target_sptr);
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
std::unique_ptr<ProjDataInfo>
ReconstructionTests<TargetT>::
construct_default_proj_data_info_uptr() const
{
  // construct a small scanner and sinogram
  shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));
  scanner_sptr->set_num_rings(5);
  std::unique_ptr<ProjDataInfo> proj_data_info_uptr(
        ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                                      /*span=*/3,
                                      /*max_delta=*/4,
                                      /*num_views=*/128,
                                      /*num_tang_poss=*/128));
  return proj_data_info_uptr;
}

template <class TargetT>
void
ReconstructionTests<TargetT>::
construct_input_data()
{ 
  Verbosity::set(1);
  if (this->_proj_data_filename.empty())
    {
      shared_ptr<ProjDataInfo> proj_data_info_sptr(this->construct_default_proj_data_info_uptr());
      shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
      exam_info_sptr->imaging_modality = ImagingModality::PT;
      _proj_data_sptr.reset(new ProjDataInMemory (exam_info_sptr, proj_data_info_sptr));

      std::cerr << "Will run tests with projection data with the following settings:\n"
                << proj_data_info_sptr->parameter_info();
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
      const float zoom=.7F;

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

      // filter it a bit to avoid too high frequency stuff creating trouble in the comparison
      SeparableGaussianImageFilter<float> filter;
      filter.set_fwhms(make_coordinate(10.F,10.F,10.F));
      filter.set_up(*vox_sptr);
      filter.apply(*vox_sptr);
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
void
ReconstructionTests<TargetT>::
reconstruct(shared_ptr<TargetT> target_sptr)
{
  this->_recon_sptr->set_input_data(this->_proj_data_sptr);
  this->_recon_sptr->set_disable_output(true);
  // set a prefix anyway, as some reconstruction algorithms write some files even with disabled output
  this->_recon_sptr->set_output_filename_prefix("test_recon_" + this->_recon_sptr->method_info());
  if (this->_recon_sptr->set_up(target_sptr)==Succeeded::no)
    error("recon::set_up() failed");
  
  if (this->_recon_sptr->reconstruct(target_sptr)==Succeeded::no)
    error("recon::reconstruct() failed");

  std::cerr << "\n================================\nReconstruction "
            << this->_recon_sptr->method_info()
            << " finished!\n\n";
}

template <class TargetT>
void
ReconstructionTests<TargetT>::
compare(const shared_ptr<const TargetT> output_sptr)
{
  if (!check(this->_input_density_sptr->has_same_characteristics(*output_sptr),
             "output image has wrong characteristics"))
    return;

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
      const std::string prefix = "test_recon_" + this->_recon_sptr->method_info();
      write_to_file(prefix + "_output.hv", *output_sptr);
      write_to_file(prefix + "_original.hv", *this->_input_density_sptr);
      write_to_file(prefix + "_diff.hv", *diff_sptr);
    }
}

END_NAMESPACE_STIR
