{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    pythonOnNix.url = "github:on-nix/python";
    pythonOnNix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nixpkgs, flake-utils, pythonOnNix, ... }:
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
        pythonOnNixSystem = pythonOnNix.lib.${system};
        env = pythonOnNixSystem.python39Env {
          name = "MtgML";
          projects = {
            async-generator = "1.10";
            flask = "2.0.1";
            google-auth = "2.1.0";
            google-auth-oauthlib = "0.4.6";
            # jsonslicer = "0.1.7";
            # keras = "2.8.0";
            matplotlib = "3.5.1";
            numpy = "1.21.3";
            pybind11 = "2.7.1";
            pyyaml = "6.0";
            # scikit-build = "0.12.0";
            setuptools = "58.2.0";
            sortedcontainers = "2.4.0";
            # tensorboard = "2.8.0";
            # tensorboard-plugin-profile = "2.5.0";
            # tensorflow = "2.8.0";
            tqdm = "4.62.3";
            zstandard = "0.15.2";
          };
        };
      in
        {
          devShell = pkgs.mkShell {
            name = "mtgdraftbots-model";
            buildInputs = [
              pkgs.clang_12
              pkgs.llvmPackages_12.libclang
              pkgs.cudaPackages.cudatoolkit
              pkgs.cudaPackages.cudnn
              pkgs.cudaPackages.nccl
              pkgs.cudaPackages.cutensor
              pkgs.cudaPackages.cuda_cupti
              pkgs.linuxPackages.nvidia_x11
              pkgs.cmake
              pkgs.ninja
              pkgs.pkg-config
              pkgs.yajl
              pkgs.zlib
              pkgs.docker
              pkgs.docker-compose
              env
            ];

            shellHook = ''
              export LIBCLANG_PATH="${pkgs.llvmPackages_12.libclang}/lib";
              export CUDATOOLKIT=${pkgs.cudaPackages.cudatoolkit}
              export CUDATOOLKIT_LIB=${pkgs.cudaPackages.cudatoolkit.lib}
              export CUDNN=${pkgs.cudaPackages.cudnn}
              export CUTENSOR=${pkgs.cudaPackages.cutensor}
              export NCCL=${pkgs.cudaPackages.nccl.out}
              export CUPTI=${pkgs.cudaPackages.cuda_cupti}
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudatoolkit.lib}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudnn}/lib:${pkgs.cudaPackages.cutensor}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudatoolkit}/nvvm/libdevice:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.zlib.out}/lib:$LD_LIBRARY_PATH
              export LD_LIBRARY_PATH=${pkgs.cudaPackages.nccl.out}/lib:${pkgs.cudaPackages.cuda_cupti}/lib:$LD_LIBRARY_PATH
              SOURCE_DATE_EPOCH=$(date +%s)
              export TF_GPU_ALLOCATOR=cuda_malloc_async
              export XLA_FLAGS="--xla_gpu_enable_fast_min_max --xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}"
              export TF_XLA_FLAGS="--tf_xla_cpu_global_jit --tf_xla_enable_lazy_compilation  --tf_xla_async_compilation"
              # export TF_XLA_FLAGS="$TF_XLA_FLAGS --tf_mlir_enable_mlir_bridge --tf_mlir_enable_merge_control_flow_pass"
              export TF_GPU_THREAD_MODE=gpu_private
            '';
          };
        }
      );
}
