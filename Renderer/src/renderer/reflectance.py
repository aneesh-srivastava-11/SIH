import numpy as np

def sun_vector(az_rad: float, el_rad: float) -> np.ndarray:
    """(3,) Returns s = (sin A cos e, cos A cos e, sin e)"""
    return np.array([np.sin(az_rad) * np.cos(el_rad),
                     np.cos(az_rad) * np.cos(el_rad),
                     np.sin(el_rad)], dtype=np.float32)

def cos_incidence(normals: np.ndarray, s: np.ndarray) -> np.ndarray:
    """(H,W) µ₀ = n·s, clipped ≥0"""
    mu0 = np.sum(normals * s, axis=-1)
    return np.clip(mu0, 0.0, None)

def cos_emission(normals: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(H,W) µ = n·v, clipped ≥0"""
    mu = np.sum(normals * v, axis=-1)
    return np.clip(mu, 0.0, None)

def phase_angle(s: np.ndarray, v: np.ndarray) -> np.ndarray | float:
    """α = arccos(s·v)"""
    dot = np.sum(s * v, axis=-1) if isinstance(s, np.ndarray) and isinstance(v, np.ndarray) else np.dot(s, v)
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)

def lunar_lambert_L(alpha_rad: float, table) -> np.ndarray | float:
    """Phase-dependent blend weight L(α) ∈ [0,1]. Table-driven, monotonic interp."""
    # The table is a callable interpolator or dummy function for testing.
    return table(alpha_rad)

def reflectance(normals: np.ndarray, s: np.ndarray, v: np.ndarray, albedo: np.ndarray, L_table, model="lunar_lambert", eps=1e-6):
    """
    model ∈ {"lambert", "lommel_seeliger", "lunar_lambert"}  ← Ablation C switch
    
    lunar_lambert:  R = A · [ 2·L(α)·µ₀/(µ₀+µ)  +  (1−L(α))·µ₀ ]
    lommel_seeliger: R = A · 2·µ₀/(µ₀+µ)
    lambert:         R = A · µ₀
    """
    mu0 = cos_incidence(normals, s)
    mu = cos_emission(normals, v)
    alpha = phase_angle(s, v)
    
    if model == "lambert":
        return albedo * mu0
        
    denominator = mu0 + mu
    # Guard against division by zero at grazing angles
    valid = denominator > eps
    
    ls_term = np.zeros_like(mu0)
    ls_term[valid] = (2.0 * mu0[valid]) / denominator[valid]
    
    if model == "lommel_seeliger":
        return albedo * ls_term
        
    elif model == "lunar_lambert":
        L_alpha = lunar_lambert_L(alpha, L_table)
        R = albedo * (L_alpha * ls_term + (1.0 - L_alpha) * mu0)
        return R
        
    else:
        raise ValueError(f"Unknown reflectance model: {model}")
