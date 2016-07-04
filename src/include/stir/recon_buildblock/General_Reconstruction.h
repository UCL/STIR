#ifndef __stir_recon_buildblock_General_Reconstruction_H__
#define __stir_recon_buildblock_General_Reconstruction_H__
/*!
  \file
  \ingroup recon_buildblock
  \brief Definition of class stir::General_Reconstruction

  \author Nikos Efthimiou
*/

#include "stir/shared_ptr.h"
#include "stir/DiscretisedDensity.h"
#include "stir/ProjData.h"
#include "stir/ParsingObject.h"
#include <vector>
#include "stir/CartesianCoordinate3D.h"
#include "Reconstruction.h"


START_NAMESPACE_STIR

class General_Reconstruction : public ParsingObject
{
public:
    //!
    //! \brief General_Reconstuction
    //! \details Default constructor
    General_Reconstruction();

    virtual Succeeded process_data();
protected:

    void set_defaults();
    void initialise_keymap();
    bool post_processing();

private:

    shared_ptr < Reconstruction < DiscretisedDensity < 3, float > > >
        recon_sptr;

};

END_NAMESPACE_STIR

#endif
