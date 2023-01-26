# Description
This is the main repository that keeps track of our working progress for the Satellite project. It contains codes, scripts, documents, etc, that are deliverables and helpers for the team.

# Scripts
## get_data.sh
### How to use

`OS_TYPE` = `mac` or `linux`

`DST` = `data` or `data_chesapeakeiclr`

Downloading into `DST` = `data_chesapeakeiclr` keeps original data folder structure to use with `ChesapeakeICLR` module.

For downloading training dataset:

```
bash get_data.sh [DST] train [OS_TYPE]
```

For downloading validation dataset:
```
bash get_data.sh [DST] val [OS_TYPE]
```

For downloading test dataset:
```
bash get_data.sh [DST] test [OS_TYPE]
```

The directory for the downloaded images will similar to the following:
```
.
├── LICENSE
├── README.md
├── data
│   ├── azcopy
│   │   └── azcopy_darwin_amd64_10.16.0
│   │       ├── NOTICE.txt
│   │       └── azcopy
│   ├── azcopy.zip
│   └── md_1m_2013_extended-debuffered-test_tiles
│       ├── m_3807502_se_18_1_landsat-leaf-off.tif
│       ├── m_3807502_se_18_1_lc.tif
│       ├── m_3807502_se_18_1_nlcd.tif
│       ├── m_3807541_sw_18_1_buildings.tif
│       ├── ...
```
