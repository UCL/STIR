VERSION=1.3
destination=krthie@shell.sf.net:stir/htdocs/
mkdir -p ~/distrib
cd ~/distrib
# somehow make release_${VERSION}.htm
#cvs -d parapet:/usr/local/cvsroot checkout parapet
cd parapet
cvs up -dP
cd PPhead
rm -rf local include/local

# update VERSION.txt
echo $VERSION > VERSION.txt
cvs commit -m "- updated for release of version $VERSION" VERSION.txt
# update LICENSE.txt
#check directories etc
#first remove list of files at the end
#then
find . -name "*[xhlc]" -print | xargs grep -l PARAPET >>LICENSE.txt 
cvs commit  -m "- updated for release of version $VERSION" LICENSE.txt

cvs2cl.pl -F trunk
cp ChangeLog ~/distrib
mv  ChangeLog ~/parapet/documentation/distrib

# make doxygen
dox
rm -rf ~/parapet/documentation/distrib/doxy
mv ../documentation/distrib/doxy ~/parapet/documentation/distrib
# make documentation PDFs BY HAND

cd ~/parapet/documentation/distrib
zip -r STIR_doc_${VERSION}.zip *.pdf ChangeLog doxy
mv STIR_doc_${VERSION}.zip ~/distrib/

cd ~/distrib
rm -f parapet/all.zip parapet/VCprojects.zip
WORKSPACE=~/distrib/parapet/PPhead zipit
WORKSPACE=~/distrib/parapet/PPhead zipproj
rm -rf PPhead STIR
unzip parapet/VCprojects.zip
rm parapet/VCprojects.zip
mv PPhead STIR
zip -r VCprojects_${VERSION}.zip STIR
rm -rf STIR
unzip parapet/all.zip
rm parapet/all.zip
mv PPhead STIR
cp ~/distrib/ChangeLog STIR
zip -r -l STIR_${VERSION}.zip STIR


# recon_test_pack
 cd parapet/
rm -rf recon_test_pack/CVS
zip -r ../recon_test_pack_${VERSION}.zip recon_test_pack
tar zcvf ../recon_test_pack_${VERSION}.tar.gz recon_test_pack

# put it all there
rsync  ~/lln/ecat/VC/ecat.dsp ~/lln/ecat.tar.gz \
    STIR_${VERSION}.zip VCprojects_${VERSION}.zip \
    recon_test_pack_${VERSION}.tar.gz recon_test_pack_${VERSION}.zip \
    ${destination}registered
rsync ChangeLog release_${VERSION}.htm STIR_doc_${VERSION}.zip  ${destination}documentation

exit 
# remote
VERSION=1.3
rm  recon_test_pack.tar.gz STIR.zip VCprojects.zip recon_test_pack.zip 
ln -s STIR_${VERSION}.zip STIR.zip 
ln -s VCprojects_${VERSION}.zip  VCprojects.zip
ln -s recon_test_pack_${VERSION}.tar.gz  recon_test_pack.tar.gz 
ln -s recon_test_pack_${VERSION}.zip recon_test_pack.zip
cd ../documentation
rm STIR_doc.zip
ln -s STIR_doc_${VERSION}.zip STIR_doc.zip 
rm -fr doxy

unzip STIR_doc
 chmod go+r STIR_doc_${VERSION}.zip
