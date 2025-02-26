{ stdenv
, lib
, fetchFromGitHub
, cmake , boost , itk , swig, libGL
, buildPython ? true, python ? null, numpy ? null
, buildLibs ? true,
}:

let
  ver = "6.2.0";
  # Uncomment and update this if you want to build from GitHub
  #sha256 = "0fzfm3ki0v2g09jqjakrq3asz69snmnydm3iwcqi6k9rabgn4g7z";

in stdenv.mkDerivation {
  name = "stir-${ver}";

  src = lib.cleanSource ./.;
  # Uncomment and update this if you want to build from GitHub
#   src = fetchFromGitHub {
#     owner = "UCL";
#     repo = "STIR";
#     rev = "rel_${ver}";
#     inherit sha256;
#   };

  buildInputs = [ boost cmake itk /*openmpi*/ ];
  propagatedBuildInputs = [ swig ]
    ++ lib.optional buildPython [ python numpy ];

  # This is a hackaround because STIR requires source available at runtime.
  setSourceRoot = ''
    actualSourceRoot=;
    for i in *;
    do
        if [ -d "$i" ]; then
            case $dirsBefore in
                *\ $i\ *)

                ;;
                *)
                    if [ -n "$actualSourceRoot" ]; then
                        echo "unpacker produced multiple directories";
                        exit 1;
                    fi;
                    actualSourceRoot="$i"
                ;;
            esac;
        fi;
    done;

    # "Install" location for source
    sourceRoot=$prefix/src
    mkdir -p $sourceRoot
    # Put the actual source there
    cp -r $actualSourceRoot -T $sourceRoot
  '';
  cmakeFlags = [
    "-DBUILD_TESTING=ON"
    "-DGRAPHICS=PGM"
    "-DSTIR_MPI=OFF"
    "-DSTIR_OPENMP=${if stdenv.isDarwin then "OFF" else "ON"}"
   ] ++ lib.optionals (buildLibs) [
    "-DBUILD_SHARED_LIBS=ON"
   ] ++ lib.optionals (buildPython) [
    "-DBUILD_SWIG_PYTHON=ON"
   ] ++ lib.optionals (!stdenv.hostPlatform.isDarwin) [
    "-DOPENGL_INCLUDE_DIR=${lib.getInclude libGL}/include"  # Borrowed from vtk
  ];
  preConfigure = lib.optionalString buildPython ''
    cmakeFlags="-DPYTHON_DEST=$out/${python.sitePackages} $cmakeFlags"
  '';
  postInstall = ''
    # add scripts to bin
    find $src/scripts -type f ! -path "*maintenance*" -name "*.sh"  -exec cp -fn {} $out/bin \;
    find $src/scripts -type f ! -path "*maintenance*" ! -name "*.*" -exec cp -fn {} $out/bin \;
  '';

  pythonPath = "";  # Makes python.buildEnv include libraries
  enableParallelBuilding = true;

  meta = with lib; {
    description = "STIR - Software for Tomographic Image Reconstruction";
    homepage = http://stir.sourceforge.net;
    license = with licenses; [ lgpl21 gpl2 free ];  # free = custom PARAPET license
  };
}
