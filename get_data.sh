#!/usr/bin/env bash
# File       : deploy.sh
# Created    : Thu Jan 06 2022 09:27:58 PM (-0500)
# Author     : Sherry Tang
# Description: Script for downloading data
# Copyright 2022 Harvard University. All Rights Reserved.

DST="$1"
DATA_TYPE="$2"
OS_TYPE="$3"
mkdir -p ./${DST}
mkdir -p ./${DST}/azcopy
cd ./${DST}

AZCOPY_URL=`curl -s -D- "https://aka.ms/downloadazcopy-v10-${OS_TYPE}" | grep Location`
PREFIX='Location: '
AZCOPY_URL=${AZCOPY_URL/#$PREFIX}
AZCOPY_URL=`echo ${AZCOPY_URL} | tr -d '\r'`
echo 'Downloading: '
echo ${AZCOPY_URL}

if [[ "$OS_TYPE" == "mac" ]]; then
	curl -o azcopy.zip "${AZCOPY_URL}"
	unzip azcopy.zip -d ./azcopy
fi
if [[ "$OS_TYPE" == "linux" ]]; then
	curl -o azcopy.tar.gz "${AZCOPY_URL}"
	tar -xf azcopy.tar.gz -C ./azcopy
fi
AZCOPY_DIR=`ls ./azcopy`

# get spatial index file
./azcopy/${AZCOPY_DIR}/azcopy copy --recursive "https://lilablobssc.blob.core.windows.net/lcmcvpr2019/cvpr_chesapeake_landcover/spatial_index.geojson" "."

if [[ "$DATA_TYPE" == "train" ]]; then
	./azcopy/${AZCOPY_DIR}/azcopy copy --recursive "https://lilablobssc.blob.core.windows.net/lcmcvpr2019/cvpr_chesapeake_landcover/md_1m_2013_extended-debuffered-train_tiles" "."
fi
if [[ "$DATA_TYPE" == "val" ]]; then
	./azcopy/${AZCOPY_DIR}/azcopy copy --recursive "https://lilablobssc.blob.core.windows.net/lcmcvpr2019/cvpr_chesapeake_landcover/md_1m_2013_extended-debuffered-val_tiles" "."
fi
if [[ "$DATA_TYPE" == "test" ]]; then
	./azcopy/${AZCOPY_DIR}/azcopy copy --recursive "https://lilablobssc.blob.core.windows.net/lcmcvpr2019/cvpr_chesapeake_landcover/md_1m_2013_extended-debuffered-test_tiles" "."
fi

if [[ "$DST" == "data_reorg" ]]; then
	cd ./*${DATA_TYPE}*
	find . -name '*_naip-old*' -delete
	find . -name '*-leaf*' -delete
	find . -name '*_nlcd.tif' -delete
	mkdir -p dataset lb_building lb_land_cover
	mv *_lc.tif lb_land_cover
	mv *_buildings* lb_building
	mv *-new* dataset
fi
