#!/bin/bash

set -ex

install_ubuntu() {
    apt-get update
    apt-get install -y wget

    DEB_ROCM_REPO=http://repo.radeon.com/rocm/apt/debian
    # Add rocm repository
    wget -qO - $DEB_ROCM_REPO/rocm.gpg.key | apt-key add -
    echo "deb [arch=amd64] $DEB_ROCM_REPO xenial main" > /etc/apt/sources.list.d/rocm.list
    apt-get update --allow-insecure-repositories

    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
                   rocm-dev \
                   rocm-libs \
                   rocm-utils \
                   rocfft \
                   miopen-hip \
                   miopengemm \
                   rocblas \
                   hipblas \
                   rocrand \
                   hcsparse \
                   rocm-profiler \
                   cxlactivitylogger

    pushd $HOME
    # install hcrng
    curl https://s3.amazonaws.com/ossci-linux/hcrng-master-a8c6a0b-Linux.deb -o hcrng.deb
    dpkg -i hcrng.deb
    rm hcrng.deb

    # hotfix a bug in hip's cmake files, this has been fixed in
    # https://github.com/ROCm-Developer-Tools/HIP/pull/516 but for
    # some reason it has not included in the latest rocm release
    if [[ -f /opt/rocm/hip/cmake/FindHIP.cmake ]]; then
        sudo sed -i 's/\ -I${dir}/\ $<$<BOOL:${dir}>:-I${dir}>/' /opt/rocm/hip/cmake/FindHIP.cmake
    fi
    
    # HIP has a bug that drops DEBUG symbols in generated MakeFiles.
    # https://github.com/ROCm-Developer-Tools/HIP/pull/588
    if [[ -f /opt/rocm/hip/cmake/FindHIP.cmake ]]; then
        sudo sed -i 's/set(_hip_build_configuration "${CMAKE_BUILD_TYPE}")/string(TOUPPER _hip_build_configuration "${CMAKE_BUILD_TYPE}")/' /opt/rocm/hip/cmake/FindHIP.cmake
    fi
}

install_centos() {
    echo "Not implemented yet"
    exit 1
}
 
install_hip_thrust() {
    # Needed for now, will be replaced soon
    git clone --recursive https://github.com/ROCmSoftwarePlatform/Thrust.git /data/Thrust
    rm -rf /data/Thrust/thrust/system/cuda/detail/cub-hip
    git clone --recursive https://github.com/ROCmSoftwarePlatform/cub-hip.git /data/Thrust/thrust/system/cuda/detail/cub-hip
    cd /data/Thrust/thrust/system/cuda/detail/cub-hip && git checkout hip_port_1.7.4_caffe2 && cd -
}

# This will be removed after merging an upcoming PR.
install_hcrng() {
    mkdir -p /opt/rocm/debians
    curl https://s3.amazonaws.com/ossci-linux/hcrng-master-a8c6a0b-Linux.deb -o /opt/rocm/debians/hcrng.deb 
    dpkg -i /opt/rocm/debians/hcrng.deb
}

# Install Python packages depending on the base OS
if [ -f /etc/lsb-release ]; then
  install_ubuntu
elif [ -f /etc/os-release ]; then
  install_centos
else
  echo "Unable to determine OS..."
  exit 1
fi

install_hip_thrust
install_hcrng
