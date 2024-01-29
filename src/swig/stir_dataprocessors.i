#ifdef STIRSWIG_SHARED_PTR
#define elemT float
%shared_ptr(stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >)
%shared_ptr(stir::RegisteredParsingObject<
             stir::ChainedDataProcessor<stir::DiscretisedDensity<3,elemT> >,
             stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >,
	    stir::DataProcessor<stir::DiscretisedDensity<3,elemT> > >)
%shared_ptr(stir::ChainedDataProcessor<stir::DiscretisedDensity<3,elemT> >)

%shared_ptr(stir::RegisteredParsingObject<stir::SeparableCartesianMetzImageFilter<elemT>,
	    stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >,
	    stir::DataProcessor<stir::DiscretisedDensity<3,elemT> > >)
%shared_ptr(stir::SeparableCartesianMetzImageFilter<elemT>)

%shared_ptr(stir::RegisteredParsingObject<stir::SeparableGaussianImageFilter<elemT>,
        stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >,
        stir::DataProcessor<stir::DiscretisedDensity<3,elemT> > >)
%shared_ptr(stir::SeparableGaussianImageFilter<elemT>)

%shared_ptr(stir::RegisteredParsingObject<stir::SeparableConvolutionImageFilter<elemT>,
        stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >,
        stir::DataProcessor<stir::DiscretisedDensity<3,elemT> > >)
%shared_ptr(stir::SeparableConvolutionImageFilter<elemT>)

%shared_ptr(stir::RegisteredParsingObject<stir::TruncateToCylindricalFOVImageProcessor<elemT>,
        stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >,
        stir::DataProcessor<stir::DiscretisedDensity<3,elemT> > >)
%shared_ptr(stir::TruncateToCylindricalFOVImageProcessor<elemT>)

%shared_ptr(stir::RegisteredParsingObject<stir::HUToMuImageProcessor<stir::DiscretisedDensity<3,elemT> >,
	    stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >,
	    stir::DataProcessor<stir::DiscretisedDensity<3,elemT> > >)
%shared_ptr(stir::HUToMuImageProcessor<stir::DiscretisedDensity<3,elemT> >)
#undef elemT
#endif

%include "stir/DataProcessor.h"
%include "stir/ChainedDataProcessor.h"
%include "stir/SeparableCartesianMetzImageFilter.h"
%include "stir/SeparableGaussianImageFilter.h"
%include "stir/SeparableConvolutionImageFilter.h"
%include "stir/TruncateToCylindricalFOVImageProcessor.h"
%include "stir/HUToMuImageProcessor.h"

#define elemT float
%template(DataProcessor3DFloat) stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >;
%template(RPChainedDataProcessor3DFloat) stir::RegisteredParsingObject<
             stir::ChainedDataProcessor<stir::DiscretisedDensity<3,elemT> >,
   stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >,
   stir::DataProcessor<stir::DiscretisedDensity<3,elemT> > >;
%template(ChainedDataProcessor3DFloat) stir::ChainedDataProcessor<stir::DiscretisedDensity<3,elemT> >;

%template(RPSeparableCartesianMetzImageFilter3DFloat) stir::RegisteredParsingObject<
             stir::SeparableCartesianMetzImageFilter<elemT>,
             stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >,
             stir::DataProcessor<stir::DiscretisedDensity<3,elemT> > >;
%template(SeparableCartesianMetzImageFilter3DFloat) stir::SeparableCartesianMetzImageFilter<elemT>;

%template(RPSeparableGaussianImageFilter3DFloat) stir::RegisteredParsingObject<
        stir::SeparableGaussianImageFilter<elemT>,
        stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >,
stir::DataProcessor<stir::DiscretisedDensity<3,elemT> > >;
%template(SeparableGaussianImageFilter3DFloat) stir::SeparableGaussianImageFilter<elemT>;

%template(RPSeparableConvolutionImageFilter3DFloat) stir::RegisteredParsingObject<
        stir::SeparableConvolutionImageFilter<elemT>,
        stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >,
stir::DataProcessor<stir::DiscretisedDensity<3,elemT> > >;
%template(SeparableConvolutionImageFilter3DFloat) stir::SeparableConvolutionImageFilter<elemT>;

%template(RPTruncateToCylindricalFOVImageProcessor3DFloat) stir::RegisteredParsingObject<
        stir::TruncateToCylindricalFOVImageProcessor<elemT>,
        stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >,
stir::DataProcessor<stir::DiscretisedDensity<3,elemT> > >;
%template(TruncateToCylindricalFOVImageProcessor3DFloat) stir::TruncateToCylindricalFOVImageProcessor<elemT>;

%template(RPHUToMuImageProcessor3DFloat) stir::RegisteredParsingObject<
             stir::HUToMuImageProcessor<stir::DiscretisedDensity<3,elemT> >,
             stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >,
             stir::DataProcessor<stir::DiscretisedDensity<3,elemT> > >;

%template(HUToMuImageProcessor3DFloat) stir::HUToMuImageProcessor<stir::DiscretisedDensity<3,elemT> >;
#undef elemT
