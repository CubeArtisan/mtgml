{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem
      (system:
      let
        pkgs = import "${nixpkgs}" {
          inherit system;
          config.allowUnfree = true;
        };
        cudaPackages = pkgs.cudaPackages_11_8;
        pythonPackages = pkgs.python310Packages;
      in
        {
          devShell = pkgs.mkShell {
            name = "mtgml";
            buildInputs = [
              cudaPackages.cudatoolkit
              cudaPackages.cudnn
              cudaPackages.cutensor
              cudaPackages.libcublas
              cudaPackages.tensorrt
              pkgs.linuxPackages.nvidia_x11
              pkgs.poetry
              pythonPackages.python
              pythonPackages.tensorboard-data-server
              pkgs.docker
              pkgs.docker-compose
              pkgs.pkg-config
              pkgs.cmake
              pkgs.ninja
              pkgs.yajl
              pkgs.zlib
              pkgs.awscli2
              pkgs.google-cloud-sdk
            ];

            shellHook = ''
              export LD_LIBRARY_PATH=${cudaPackages.cudatoolkit}/lib:${cudaPackages.cudatoolkit.lib}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${cudaPackages.cudnn}/lib:${cudaPackages.cutensor}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${cudaPackages.cudatoolkit}/nvvm/libdevice:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${cudaPackages.libcublas}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${cudaPackages.tensorrt}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.yajl}/lib:$LD_LIBRARY_PATH

              # export TF_GPU_ALLOCATOR=cuda_malloc_async
              export XLA_FLAGS="--xla_gpu_enable_fast_min_max --xla_gpu_cuda_data_dir=${cudaPackages.cudatoolkit}"
              export TF_XLA_FLAGS="--tf_xla_enable_lazy_compilation  --tf_xla_async_compilation"
              export TF_GPU_THREAD_MODE=gpu_private

              poetry env use 3.10
              poetry install --sync
              export VENV_DIR=$(poetry env info --path)
              source $VENV_DIR/bin/activate
              python setup.py install --build-type Release
            '';
          };
        }
      );
}
