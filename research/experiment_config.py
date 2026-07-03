from research.layer_config import LayerConfig


FULL_V4 = LayerConfig()

NO_AFL = LayerConfig(afl=False)
NO_CAL = LayerConfig(cal=False)
NO_FWO = LayerConfig(fwo=False)

NO_CAL_FWO = LayerConfig(cal=False, fwo=False)

NO_DRL_MSO = LayerConfig(drl=False, mso=False)

NO_LCT_SSOL = LayerConfig(lct=False, ssol=False)

HMS24_MINIMAL = LayerConfig(
    doa=True,
    afl=False,
    cal=False,
    fwo=False,
    rsl=False,
    rtd=False,
    tca=False,
    cwf=False,
    cdm=False,
    drl=False,
    mso=False,
    lct=True,
    ssol=False,
)

EXPERIMENTS = {
    "FULL_V4": FULL_V4,
    "NO_AFL": NO_AFL,
    "NO_CAL": NO_CAL,
    "NO_FWO": NO_FWO,
    "NO_CAL_FWO": NO_CAL_FWO,
    "NO_DRL_MSO": NO_DRL_MSO,
    "NO_LCT_SSOL": NO_LCT_SSOL,
    "HMS24_MINIMAL": HMS24_MINIMAL,
}
