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
        pythonPackages = pkgs.python311Packages;
      in
        {
          devShell = pkgs.mkShell {
            name = "mtgml";
            buildInputs = [
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              pkgs.cudaPackages.cutensor
              pkgs.cudaPackages.libcublas
              pkgs.cudaPackages.tensorrt
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
              export CUDATOOLKIT=${pkgs.cudaPackages.cudatoolkit}
              export CUDATOOLKIT_LIB=${pkgs.cudaPackages.cudatoolkit.lib}
              export CUDA_DIR=$CUDATOOLKIT_LIB
              export CUDNN=${pkgs.cudaPackages.cudnn}
              export CUTENSOR=${pkgs.cudaPackages.cutensor}
              export CUBLAS=${pkgs.cudaPackages.libcublas}
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudatoolkit.lib}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudnn}/lib:${pkgs.cudaPackages.cutensor}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/nvvm/libdevice:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.libcublas}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.tensorrt}/lib:${pkgs.linuxPackages.nvidia_x11}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.yajl}/lib:$LD_LIBRARY_PATH

              # export TF_GPU_ALLOCATOR=cuda_malloc_async
              export XLA_FLAGS="--xla_gpu_enable_fast_min_max --xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}"
              export TF_XLA_FLAGS="--tf_xla_cpu_global_jit --tf_xla_enable_lazy_compilation  --tf_xla_async_compilation"
              export TF_GPU_THREAD_MODE=gpu_private

              poetry env use 3.11
              export VENV_DIR=$(poetry env info --path)
              source $VENV_DIR/bin/activate
              poetry install --sync
              python setup.py install --build-type Release
            '';
          };
        }
      );
}
