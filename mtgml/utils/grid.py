def interpolate(pickNum, numPicks, packNum, numPacks):
    fpackIdx = 3 * packNum / numPacks
    fpickIdx = 15 * pickNum / numPicks
    floorpackIdx = min(2, int(fpackIdx))
    ceilpackIdx = min(2, floorpackIdx + 1)
    floorpickIdx = min(14, int(fpickIdx))
    ceilpickIdx = min(14, floorpickIdx + 1)
    modpackIdx = fpackIdx - floorpackIdx
    modpickIdx = fpickIdx - floorpickIdx
    coords = ((floorpackIdx, floorpickIdx), (floorpackIdx, ceilpickIdx), (ceilpackIdx, floorpickIdx), (ceilpackIdx, ceilpickIdx))
    weights = ((1 - modpackIdx) * (1 - modpickIdx), (1 - modpackIdx) * modpickIdx, modpackIdx * (1 - modpickIdx), modpackIdx * modpickIdx)
    return coords, weights
