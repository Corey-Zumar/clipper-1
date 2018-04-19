#!/usr/bin/env bash

set -u
set -e
set -o pipefail

unset CDPATH
# one-liner from http://stackoverflow.com/a/246128
# Determines absolute path of the directory containing
# the script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Let the user start this script from anywhere in the filesystem.


CLIPPER_ROOT=$DIR/..

pip2 install -e $CLIPPER_ROOT/clipper_admin

pip2 install -e $CLIPPER_ROOT/model_composition/containerized_utils

pip2 install -e $CLIPPER_ROOT/model_composition/single_proc_utils
