# Third-party notices

## Real-ESRGAN

The compact `SRVGGNetCompact` architecture and official weak-denoise v3 weights
come from [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), licensed
under the BSD 3-Clause License. The local integration loads weights with
`weights_only=True` and does not execute the upstream inference scripts.

## AMD FidelityFX CAS

The contrast-adaptive sharpening stage is a CPU adaptation of the algorithm
described by [AMD FidelityFX CAS](https://github.com/GPUOpen-Effects/FidelityFX-CAS).
FidelityFX CAS is distributed under the MIT License.

## Torph

Interface text and numeric transitions use
[lochie/torph](https://github.com/lochie/torph), version 0.1.3, distributed under
the MIT License. The package is stored locally and the application does not load it
from a CDN.
