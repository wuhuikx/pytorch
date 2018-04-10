#!/usr/bin/env bash

set -ex

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

pushd $SCRIPTPATH
source ./build_common.sh

echo "Building Torch"

mkdir -p $LIBTORCH_BUILDPATH
pushd $LIBTORCH_BUILDPATH

cmake -DNO_CUDA:BOOL=$NO_CUDA \
      -DATEN_PATH=$PYTORCHPATH/aten/ \
      -DATEN_BUILD_PATH=$ATEN_BUILDPATH \
      -DCMAKE_BUILD_TYPE:STRING=$BUILD_TYPE \
      -DCMAKE_INSTALL_PREFIX:STRING=$INSTALL_PREFIX \
      -DCMAKE_INSTALL_MESSAGE=NEVER \
      -DVERBOSE=${VERBOSE:-0} \
      -DWerror=${WERROR:-0} \
      -G "$GENERATE" \
      $SCRIPTPATH/libtorch
$MAKE -j "$JOBS"

popd
popd
