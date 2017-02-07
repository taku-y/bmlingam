#!/bin/bash
# This script execute processes for running test of BMLiNGAM.
# This script will run in a Docker container.
# It is assumed that /home/jovyan/work is mounted on the root directory of the
# bmlingam repo.

# Switch environment
if [ $# -ge 1 ]; then
    if [ $1 = "py27" ]; then
        source activate py27
    fi
fi

# 1/3 Make wheel of BMLiNGAM
sh scripts/docker/make_wheel.sh

# 2/3 Install BMLiNGAM
pushd dist
WHEEL=`ls .`
echo $WHEEL
pip install --user $WHEEL
popd
export PATH=/home/jovyan/.local/bin:$PATH

# 3/3 run commands in tutorial
pushd /home/jovyan/work
mkdir -p tmp
export LOGFILE=/home/jovyan/work/tmp/log.txt

echo "$ bmlingam-make-testdata" > $LOGFILE
bmlingam-make-testdata >> $LOGFILE
echo >> $LOGFILE

echo "$ head -6 sampledata.csv" >> $LOGFILE
head -6 sampledata.csv >> $LOGFILE
echo >> $LOGFILE

echo "$ mkdir result1" >> $LOGFILE
echo "$ bmlingam-causality sampledata.csv --sampling_mode=cache_mp4 --result_dir=tmp/result1" >> $LOGFILE
mkdir -p tmp/result1
bmlingam-causality sampledata.csv --sampling_mode=cache_mp4 --result_dir=tmp/result1 >> $LOGFILE
echo >> $LOGFILE

echo "bmlingam-coeff tmp/result1/x1_src_x2_dst.bmlm.pklz" >> $LOGFILE
bmlingam-coeff tmp/result1/x1_src_x2_dst.bmlm.pklz >> $LOGFILE
echo >> $LOGFILE

popd
