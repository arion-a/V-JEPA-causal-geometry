"""Scene and parameter names shared by every module."""

from enum import StrEnum


class Scene(StrEnum):
    FREEFALL = "freefall"
    BOUNCE = "bounce"
    INCLINE = "incline"


class Param(StrEnum):
    G = "g"
    E = "e"
    MU = "mu"


OBSERVABLE = {
    Scene.FREEFALL: (Param.G,),
    Scene.BOUNCE: (Param.G, Param.E),
    Scene.INCLINE: (Param.G, Param.MU),
}
