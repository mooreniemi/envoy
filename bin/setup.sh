# basic setup I did to get torch working
# see also https://github.com/LaurentMazare/tch-rs
#
# specifically checked version in torch-sys here:
# https://docs.rs/crate/torch-sys/0.3.0/source/build.rs

cd $HOME

wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.7.0.zip

unzip libtorch-cxx11-abi-shared-with-deps-1.7.0.zip

export LIBTORCH=$HOME/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
