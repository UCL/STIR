#ifdef STIRSWIG_SHARED_PTR
#define elemT float
%shared_ptr(stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >)
%shared_ptr(stir::RegisteredParsingObject<
             stir::ChainedDataProcessor<stir::DiscretisedDensity<3,elemT> >,
             stir::DataProcessor<DiscretisedDensity<3,elemT> >,
	    stir::DataProcessor<DiscretisedDensity<3,elemT> > >)
%shared_ptr(stir::ChainedDataProcessor<stir::DiscretisedDensity<3,elemT> >)

%shared_ptr(stir::RegisteredParsingObject<stir::SeparableCartesianMetzImageFilter<elemT>,
	    stir::DataProcessor<DiscretisedDensity<3,elemT> >,
	    stir::DataProcessor<DiscretisedDensity<3,elemT> > >)
%shared_ptr(stir::SeparableCartesianMetzImageFilter<elemT>)

%shared_ptr(stir::RegisteredParsingObject<stir::SeparableGaussianImageFilter<elemT>,
        stir::DataProcessor<DiscretisedDensity<3,elemT> >,
        stir::DataProcessor<DiscretisedDensity<3,elemT> > >)
%shared_ptr(stir::SeparableGaussianImageFilter<elemT>)

%shared_ptr(stir::RegisteredParsingObject<stir::SeparableConvolutionImageFilter<elemT>,
        stir::DataProcessor<DiscretisedDensity<3,elemT> >,
        stir::DataProcessor<DiscretisedDensity<3,elemT> > >)
%shared_ptr(stir::SeparableConvolutionImageFilter<elemT>)

#ifdef HAVE_JSON
%shared_ptr(stir::RegisteredParsingObject<stir::HUToMuImageProcessor<DiscretisedDensity<3,elemT> >,
	    stir::DataProcessor<DiscretisedDensity<3,elemT> >,
	    stir::DataProcessor<DiscretisedDensity<3,elemT> > >)
%shared_ptr(stir::HUToMuImageProcessor<DiscretisedDensity<3,elemT> >)
#endif
#undef elemT
#endif

%include "stir/DataProcessor.h"
%include "stir/ChainedDataProcessor.h"
%include "stir/SeparableCartesianMetzImageFilter.h"
%include "stir/SeparableGaussianImageFilter.h"
%include "stir/SeparableConvolutionImageFilter.h"
#ifdef HAVE_JSON
%include "stir/HUToMuImageProcessor.h"
#endif

#define elemT float
%template(DataProcessor3DFloat) stir::DataProcessor<stir::DiscretisedDensity<3,elemT> >;
%template(RPChainedDataProcessor3DFloat) stir::RegisteredParsingObject<
             stir::ChainedDataProcessor<stir::DiscretisedDensity<3,elemT> >,
             stir::DataProcessor<DiscretisedDensity<3,elemT> >,
             stir::DataProcessor<DiscretisedDensity<3,elemT> > >;
%template(ChainedDataProcessor3DFloat) stir::ChainedDataProcessor<stir::DiscretisedDensity<3,elemT> >;

%template(RPSeparableCartesianMetzImageFilter3DFloat) stir::RegisteredParsingObject<
             stir::SeparableCartesianMetzImageFilter<elemT>,
             stir::DataProcessor<DiscretisedDensity<3,elemT> >,
             stir::DataProcessor<DiscretisedDensity<3,elemT> > >;
%template(SeparableCartesianMetzImageFilter3DFloat) stir::SeparableCartesianMetzImageFilter<elemT>;

%template(RPSeparableGaussianImageFilter3DFloat) stir::RegisteredParsingObject<
        stir::SeparableGaussianImageFilter<elemT>,
        stir::DataProcessor<DiscretisedDensity<3,elemT> >,
stir::DataProcessor<DiscretisedDensity<3,elemT> > >;
%template(SeparableGaussianImageFilter3DFloat) stir::SeparableGaussianImageFilter<elemT>;

%template(RPSeparableConvolutionImageFilter3DFloat) stir::RegisteredParsingObject<
        stir::SeparableConvolutionImageFilter<elemT>,
        stir::DataProcessor<DiscretisedDensity<3,elemT> >,
stir::DataProcessor<DiscretisedDensity<3,elemT> > >;
%template(SeparableConvolutionImageFilter3DFloat) stir::SeparableConvolutionImageFilter<elemT>;

#ifdef HAVE_JSON
%template(RPHUToMuImageProcessor3DFloat) stir::RegisteredParsingObject<
             stir::HUToMuImageProcessor<DiscretisedDensity<3,elemT> >,
             stir::DataProcessor<DiscretisedDensity<3,elemT> >,
             stir::DataProcessor<DiscretisedDensity<3,elemT> > >;

%template(HUToMuImageProcessor3DFloat) stir::HUToMuImageProcessor<DiscretisedDensity<3,elemT> >;
#endif
#undef elemT
