let
  pkgs = import <nixpkgs> {
      config = {
       allowUnfree = true;
       cudaSupport = true;
      };
  };
in
pkgs.mkShell {
  packages = [
    pkgs.uv
    pkgs.python311
  ];

  nativeBuildInputs = with pkgs.buildPackages; [
      cudaPackages_12.cudatoolkit
      python311
      cudaPackages_12.cudatoolkit
      python311Packages.pytorch-bin
    ];

  env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.libz
  ];

  shellHook = ''
    export PATH="${pkgs.python311}/bin:$PATH"
    export CUDA_PATH=${pkgs.cudatoolkit}
    if [ ! -d ".venv" ]; then
      echo "Creating uv virtual environment..."
      uv venv
    fi
    source .venv/bin/activate
    echo "uv virtual environment activated."
  '';

  GOOGLE_API_KEY="AIzaSyA3FVpKidQ4dNt3MeoljHBOATwgCuNYeLc";
}
