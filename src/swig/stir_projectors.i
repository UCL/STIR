/// projectors
%shared_ptr(stir::ForwardProjectorByBin);

%shared_ptr(stir::RegisteredParsingObject<stir::ForwardProjectorByBinUsingProjMatrixByBin,
    stir::ForwardProjectorByBin>);
%shared_ptr(stir::ForwardProjectorByBinUsingProjMatrixByBin);
%shared_ptr(stir::BackProjectorByBin);
%shared_ptr(stir::RegisteredParsingObject<stir::BackProjectorByBinUsingProjMatrixByBin,
    stir::BackProjectorByBin>);
%shared_ptr(stir::BackProjectorByBinUsingProjMatrixByBin);


%shared_ptr(stir::ProjMatrixByBin);
%shared_ptr(stir::RegisteredParsingObject<
	      stir::ProjMatrixByBinUsingRayTracing,
              stir::ProjMatrixByBin,
              stir::ProjMatrixByBin
	    >);
%shared_ptr(stir::ProjMatrixByBinUsingRayTracing);

%include "stir/recon_buildblock/ForwardProjectorByBin.h"
%include "stir/recon_buildblock/BackProjectorByBin.h"

%include "stir/recon_buildblock/ProjMatrixByBin.h"

%template (internalRPProjMatrixByBinUsingRayTracing) stir::RegisteredParsingObject<
	      stir::ProjMatrixByBinUsingRayTracing,
              stir::ProjMatrixByBin,
              stir::ProjMatrixByBin
  >;

%include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"

%template (internalRPForwardProjectorByBinUsingProjMatrixByBin)  
  stir::RegisteredParsingObject<stir::ForwardProjectorByBinUsingProjMatrixByBin,
     stir::ForwardProjectorByBin>;
%include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"


%template (internalRPBackProjectorByBinUsingProjMatrixByBin)  
  stir::RegisteredParsingObject<stir::BackProjectorByBinUsingProjMatrixByBin,
     stir::BackProjectorByBin>;
%include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"

%shared_ptr(stir::ProjectorByBinPair);
// explicitly ignore constructor, because SWIG tries to instantiate the abstract class otherwise
%ignore stir::ProjectorByBinPair::ProjectorByBinPair();
%shared_ptr(stir::RegisteredParsingObject<
        stir::ProjectorByBinPairUsingProjMatrixByBin,
              stir::ProjectorByBinPair,
              stir::ProjectorByBinPair>);
%shared_ptr(stir::ProjectorByBinPairUsingProjMatrixByBin)
%include "stir/recon_buildblock/ProjectorByBinPair.h"
%template(internalRPProjectorByBinPairUsingProjMatrixByBin) stir::RegisteredParsingObject<
        stir::ProjectorByBinPairUsingProjMatrixByBin,
              stir::ProjectorByBinPair,
              stir::ProjectorByBinPair>;
%include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"

#ifdef HAVE_parallelproj
%shared_ptr(stir::RegisteredParsingObject<stir::ForwardProjectorByBinParallelproj,
    stir::ForwardProjectorByBin>);
%shared_ptr(stir::ForwardProjectorByBinParallelproj);
%shared_ptr(stir::RegisteredParsingObject<stir::BackProjectorByBinParallelproj,
    stir::BackProjectorByBin>);
%shared_ptr(stir::BackProjectorByBinParallelproj);

%template (internalRPForwardProjectorByBinParallelproj)  
  stir::RegisteredParsingObject<stir::ForwardProjectorByBinParallelproj,
     stir::ForwardProjectorByBin>;
%include "stir/recon_buildblock/Parallelproj_projector/ForwardProjectorByBinParallelproj.h"

%template (internalRPBackProjectorByBinParallelproj)  
  stir::RegisteredParsingObject<stir::BackProjectorByBinParallelproj,
     stir::BackProjectorByBin>;
%include "stir/recon_buildblock/Parallelproj_projector/BackProjectorByBinParallelproj.h"
#endif
