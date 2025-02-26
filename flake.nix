{
  description = "Software for Tomographic Image Reconstruction";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      nixpkgsFor = forAllSystems (system: import nixpkgs { inherit system; });
    in
    {
      # STIR Builds available: nix run /path/to/stir#stir-nolibs
      packages = forAllSystems (system:
        let pkgs = nixpkgsFor.${system}; in with pkgs; rec {
          stir = callPackage ./default.nix { inherit (python3Packages) python numpy; };
          stir-nolibs = callPackage ./default.nix { buildLibs = false; buildPython = false; };
          default = stir;
        }
      );

      # Development environment
      devShells = forAllSystems (system:
        let
          pkgs = nixpkgsFor.${system};
          # Which STIR package to use
          package = self.packages.${system}.stir-nolibs;

        in {
          default = pkgs.mkShell {
            buildInputs = package.buildInputs ++ [
              # Extra dependencies for the shell
              pkgs.cmakeCurses
            ];
            nativeBuildInputs = package.nativeBuildInputs or [];

            shellHook = ''
              echo "Development shell for STIR is ready!"
            '';
          };
        }
      );
    };
}
