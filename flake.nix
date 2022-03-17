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
        venvDir = "./.venv";
        defaultShellPath = pkgs.lib.makeBinPath [ pkgs.bash pkgs.coreutils pkgs.findutils pkgs.gnugrep pkgs.gnused pkgs.which ];
        cacheRequirements = pkgs.lib.readFile ./requirements.txt;
      in
        {
          devShell = pkgs.mkShell {
            name = "mtgdraftbots-model";
            buildInputs = [
              pkgs.clang_12
              pkgs.llvmPackages_12.libclang
              pkgs.cudaPackages.cudatoolkit_11_4
              pkgs.cutensor_cudatoolkit_11_4
              pkgs.cudnn_8_3_cudatoolkit_11_4
              pkgs.linuxPackages.nvidia_x11
              pkgs.python39Packages.python
              pkgs.cmake
              pkgs.ninja
              pkgs.pkg-config
              pkgs.yajl
              pkgs.zlib
              pkgs.docker
              pkgs.docker-compose
            ];

            shellHook = ''
              export LIBCLANG_PATH="${pkgs.llvmPackages_12.libclang}/lib";
              export CUDATOOLKIT=${pkgs.cudaPackages.cudatoolkit_11_4}
              export CUDATOOLKIT_LIB=${pkgs.cudaPackages.cudatoolkit_11_4.lib}
              export CUDNN=${pkgs.cudnn_8_3_cudatoolkit_11_4}
              export CUTENSOR=${pkgs.cutensor_cudatoolkit_11_4}
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit_11_4}/lib:${pkgs.cudaPackages.cudatoolkit_11.lib}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.cudnn_8_3_cudatoolkit_11_4}/lib:${pkgs.cutensor_cudatoolkit_11_4}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit_11_4}/nvvm/libdevice:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.zlib.out}/lib:$LD_LIBRARY_PATH
              SOURCE_DATE_EPOCH=$(date +%s)
              export TF_GPU_ALLOCATOR=cuda_malloc_async
              export XLA_FLAGS="--xla_gpu_enable_fast_min_max --xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit_11_4}"
              export TF_XLA_FLAGS="--tf_xla_cpu_global_jit --tf_xla_enable_lazy_compilation  --tf_xla_async_compilation"
              # export TF_XLA_FLAGS="$TF_XLA_FLAGS --tf_mlir_enable_mlir_bridge --tf_mlir_enable_merge_control_flow_pass"
              export TF_GPU_THREAD_MODE=gpu_private

              if [ ! -d "${venvDir}" ]; then
                echo "Creating new venv environment in path: '${venvDir}'"
                ${pkgs.python39Packages.python.interpreter} -m venv "${venvDir}"
                source "${venvDir}/bin/activate"
                pip install --upgrade wheel setuptools pip
                pip install -r requirements.txt
              else
                source "${venvDir}/bin/activate"
              fi

              # Under some circumstances it might be necessary to add your virtual
              # environment to PYTHONPATH, which you can do here too;
              # export PYTHONPATH=$PWD/${venvDir}/${pkgs.python38Packages.python.sitePackages}/:$PYTHONPATH

              unset SOURCE_DATE_EPOCH
            '';
          };
        }
      );
}
