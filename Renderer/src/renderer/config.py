import yaml
from pydantic import BaseModel
import hashlib
import json
from typing import Tuple

class ReflectanceConfig(BaseModel):
    model: str
    eps: float

class AlbedoConfig(BaseModel):
    clamp_min: float
    clamp_max: float
    procedural_mean: float
    procedural_scale: float
    fbm_octaves: int
    fbm_sigma: float

class SensorConfig(BaseModel):
    jitter_sigma: float
    secondary_eps: float
    signal_scale: float
    psf_sigma: float
    read_noise_e: float
    gain: float
    offset: float
    bits: int
    to_8bit: bool

class RandomizeConfig(BaseModel):
    azimuth: Tuple[float, float]
    elevation_eq: Tuple[float, float]
    elevation_polar: Tuple[float, float]
    albedo_scale: Tuple[float, float]
    additive_noise_sigma: Tuple[float, float]
    roughness_amp: Tuple[float, float]
    psf_sigma: Tuple[float, float]
    read_noise_e: Tuple[float, float]
    secondary_eps: Tuple[float, float]
    scale_ratio: Tuple[float, float]
    rotation: Tuple[float, float]

class RenderConfig(BaseModel):
    reflectance: ReflectanceConfig
    albedo: AlbedoConfig
    sensor: SensorConfig
    randomize: RandomizeConfig
    
def load_config(path: str) -> RenderConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return RenderConfig(**data)

def config_hash(cfg: RenderConfig) -> str:
    d = cfg.model_dump()
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
