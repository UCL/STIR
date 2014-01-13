#
# $Id$
#
# Author Kris Thielemans
# Copyright 2004- $Date$ Hammersmith Imanet Ltd
# This file is part of STIR, and is distributed under the 
# terms of the GNU Lesser General Public Licence (LGPL) Version 2.1.


all: STIR-glossary.pdf STIR-general-overview.pdf \
     STIR-developers-overview.pdf \
     STIR-UsersGuide.pdf \
     STIR-version2-vs-version1.pdf

clean:
	$(RM) -rf STIR-*.pdf *.log *.dvi *.aux *.toc *.out STIR-UsersGuide/

doc_as_html: STIR-UsersGuide/homepage.html	


.PHONY: all clean doc_as_html

# rule to make pdf from tex source
# note: need to run twice to get page refs etc ok.
%.pdf: %.tex
	pdflatex $<
	pdflatex $<

STIR-UsersGuide/homepage.html: STIR-UsersGuide.pdf
	rm -rf STIR-UsersGuide;
	latex2html -no_auto_link -split 7 -local_icons STIR-UsersGuide.tex; 
	pushd STIR-UsersGuide/;
	rm *.pl *.tex *.log WARNINGS;
	cp STIR-UsersGuide.html index.html
	cp STIR-UsersGuide.html homepage.html
	popd

