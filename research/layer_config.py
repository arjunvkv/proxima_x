from dataclasses import dataclass


@dataclass(frozen=True)
class LayerConfig:
    ecdf: bool = True
    ranking: bool = True
    rotation: bool = True
    h20: bool = True
    execution: bool = True
    fusion_kernel: bool = True
    doa: bool = True
    afl: bool = True
    cal: bool = True
    fwo: bool = True
    rsl: bool = True
    rtd: bool = True
    tca: bool = True
    cwf: bool = True
    cdm: bool = True
    drl: bool = True
    mso: bool = True
    lct: bool = True
    ssol: bool = True
