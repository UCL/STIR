// %W%: %E%
#ifndef __FBP2DReconstruction_H__
#define __FBP2DReconstruction_H__
/*!
  \file 
  \ingroup recon_buildblock
 
  \brief declares the FBP2DReconstruction class

  \author Kris Thielemans
  \author PARAPET project

  %E%
  %I%
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/Reconstruction.h"
#include "local/stir/FBP2D/RampFilter.h"
#include "stir/recon_buildblock/ReconstructionParameters.h"

START_NAMESPACE_STIR

template <typename elemT> class Segment;
template <int num_dimensions, typename elemT> class DiscretisedDensity;
template <typename T> class shared_ptr;

//! Reconstruction class for 2D Filtered Back Projection
class FBP2DReconstruction : public Reconstruction
{

public:
    FBP2DReconstruction(const Segment<float>& direct_sinos, const RampFilter& f);

    Succeeded reconstruct(shared_ptr<DiscretisedDensity<3,float> > const &);

    virtual string method_info() const;
 
    virtual string parameter_info();


protected:
    RampFilter filter;

private:
  const Segment<float>& direct_sinos;
  ReconstructionParameters parameters;

  virtual ReconstructionParameters& params();
  virtual const ReconstructionParameters& params() const;

};




END_NAMESPACE_STIR

    
#endif

