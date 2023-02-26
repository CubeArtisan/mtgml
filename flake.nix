{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    poetry2nix.url = "github:nix-community/poetry2nix";
    poetry2nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nixpkgs, flake-utils, poetry2nix, ... }:
    flake-utils.lib.eachDefaultSystem
      (system:
      let
        pkgs = import "${nixpkgs}" {
          inherit system;
          config.allowUnfree = true;
          overlays = [ poetry2nix.overlay ];
        };
        libclang = pkgs.llvmPackages_14.libclang;
        pythonPackages = pkgs.python310Packages;
      in
        {
          devShell = pkgs.mkShell {
            name = "mtgdraftbots-model";
            buildInputs = [
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              pkgs.cudaPackages.cutensor
              pkgs.cudaPackages.libcublas
              pkgs.cudaPackages.tensorrt
              pkgs.docker
              pkgs.linuxPackages.nvidia_x11
              poetry2nix.packages.${system}.poetry
              pythonPackages.python
              pythonPackages.tensorboard-data-server
              pkgs.docker-compose
              pkgs.pkg-config
              pkgs.docker
              pkgs.cmake
              pkgs.ninja
              pkgs.clang_12
              pkgs.yajl
              pkgs.zlib
              pkgs.awscli2
            ];

            shellHook = ''
              export LIBCLANG_PATH="${libclang}/lib";
              export CUDATOOLKIT=${pkgs.cudaPackages.cudatoolkit}
              export CUDATOOLKIT_LIB=${pkgs.cudaPackages.cudatoolkit.lib}
              export CUDA_DIR=$CUDATOOLKIT_LIB
              export CUDNN=${pkgs.cudaPackages.cudnn}
              export CUTENSOR=${pkgs.cudaPackages.cutensor}
              export CUBLAS=${pkgs.cudaPackages.libcublas}
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudatoolkit.lib}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudnn}/lib:${pkgs.cudaPackages.cutensor}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/nvvm/libdevice:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.libcublas}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.tensorrt}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.yajl}/lib:$LD_LIBRARY_PATH
              export TF_GPU_ALLOCATOR=cuda_malloc_async
              export XLA_FLAGS="--xla_gpu_enable_fast_min_max --xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}"
              export TF_XLA_FLAGS="--tf_xla_cpu_global_jit --tf_xla_enable_lazy_compilation  --tf_xla_async_compilation"
              # export TF_XLA_FLAGS="$TF_XLA_FLAGS --tf_mlir_enable_mlir_bridge --tf_mlir_enable_merge_control_flow_pass"
              export TF_GPU_THREAD_MODE=gpu_private
              export CUDA_AVAILABLE_DEVICES=-1
              export VENV_DIR=$(poetry env info --path)
              if [ ! -d "$VENV_DIR" ]; then
                poetry env use 3.10
                poetry install
                export VENV_DIR=$(poetry env info --path)
                source $VENV_DIR/bin/activate
                python setup.py install --build_type Release
              else
                source $VENV_DIR/bin/activate
              fi
            '';
          };
        }
      );
}
