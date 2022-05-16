# labutils

## How to setup the zbatlas folder

the folder should contain the trasformation files we have downloaded preiously:
```bash
$ ls zbatlas/
live2fixed_0GenericAffine.mat
live2fixed_1InverseWarp.nii.gz
live2fixed_1Warp.nii.gz
MPIN-Atlas__Reference_brains__Fixed__ERK.nrrd
MPIN-Atlas__Reference_brains__Fixed__HuCnlsGCaMP.nrrd
MPIN-Atlas__Reference_brains__Fixed__HuC.nrrd
MPIN-Atlas__Reference_brains__Fixed__SYP.nrrd
MPIN-Atlas__Reference_brains__Live__HuCH2BGCaMP.nrrd
url
```
url is a text file containing the new api url of the zbatlas. The contents should be as follows:

```
http://18.185.7.131:8080
```

After this instantiating for the first time an atlas object it will download all the remaining necesary data:

```python
from labutils.zbatlas import MPIN_Atlas
atlas = MPIN_Atlas('path/to/zbatlas')
```

## How to use the fish classes

To use the fish classes first you will need to setup your data as follows:

```bash
├── 20211005_DARK1  # <- diectory for each fish, can have any name
│   ├── T000        # can have many tseries with different names
│   ├── T001
│   ├── T002
│   ├── T003
│   ├── T004
│   ├── T005
│   └── Z           # Z stack to align everything must be called Z and be unique 
├── 20211005_LIGHT1
│   ├── T000
│   ├── T001
│   ├── T002
│   ├── T003
│   └── Z
├── _20211123_DARK1_G7
│   ├── baseline
│   └── Z
├── _20211123_DARK2_G7
│   ├── baseline
│   └── Z
```
T series and Z stack should be just out of thorlab data.
If the T series has already been segmented with suite2p the suite2p folder should be put inside.

Then to load data:
```python
from labutils.zbatlas import MPIN_Atlas
from labutils.thorio import Fish
atlas = MPIN_Atlas('path/to/zbatlas')
myfish = Fish('path/to/20211005_LIGHT1', atlas, md={'gcamp': 6, tseries:['T000', 'T001', 'T002', 'T003' ]})  # for example
# it will create a md file with some data for each fish
myfish.Z  # to access the Z stack
myfish.Ts[0] # to open the first Tseries
myfish.Ts[0].cells # to access the time evolution of neurons from suite2p (will be a little slow to load data the first time, or will also run s2p if it wasnàt done)
myfish.Ts[0].atlaspos # to access position of cells but in the atlas space (will align Zstack to atlas if not already done)
```
If ants was already run for a zstack you can move the transform files in the Z folder with the names
```
z2live_0GenericAffine.mat
z2live_1InverseWarp.nii.gz
z2live_1Warp.nii.gz
```
