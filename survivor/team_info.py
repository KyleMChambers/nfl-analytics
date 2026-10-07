"""Stadium location, time zone (standard UTC offset), climate, and roof for every NFL team (incl. relocated codes)."""

# abbr: (lat, lon, utc_offset, climate, indoor)
#   climate: "warm" (hot most of the year), "cold" (cold winters), "mild"
#   indoor:  True for domes / covered or retractable-roof stadiums
TEAM_INFO = {
    "ARI": (33.528, -112.263, -7, "warm", True),  "ATL": (33.755, -84.401, -5, "warm", True),
    "BAL": (39.278, -76.623, -5, "cold", False),  "BUF": (42.774, -78.787, -5, "cold", False),
    "CAR": (35.226, -80.853, -5, "mild", False),  "CHI": (41.862, -87.617, -6, "cold", False),
    "CIN": (39.095, -84.516, -5, "cold", False),  "CLE": (41.506, -81.700, -5, "cold", False),
    "DAL": (32.748, -97.093, -6, "warm", True),   "DEN": (39.744, -105.020, -7, "cold", False),
    "DET": (42.340, -83.046, -5, "cold", True),   "GB":  (44.501, -88.062, -6, "cold", False),
    "HOU": (29.685, -95.411, -6, "warm", True),   "IND": (39.760, -86.164, -5, "cold", True),
    "JAX": (30.324, -81.637, -5, "warm", False),  "KC":  (39.049, -94.484, -6, "cold", False),
    "LV":  (36.091, -115.184, -8, "warm", True),  "LAC": (33.953, -118.339, -8, "warm", True),
    "LA":  (33.953, -118.339, -8, "warm", True),  "MIA": (25.958, -80.239, -5, "warm", False),
    "MIN": (44.974, -93.258, -6, "cold", True),   "NE":  (42.091, -71.264, -5, "cold", False),
    "NO":  (29.951, -90.081, -6, "warm", True),   "NYG": (40.813, -74.074, -5, "cold", False),
    "NYJ": (40.813, -74.074, -5, "cold", False),  "PHI": (39.901, -75.168, -5, "cold", False),
    "PIT": (40.447, -80.016, -5, "cold", False),  "SF":  (37.403, -121.970, -8, "mild", False),
    "SEA": (47.595, -122.332, -8, "mild", False), "TB":  (27.976, -82.503, -5, "warm", False),
    "TEN": (36.166, -86.771, -6, "mild", False),  "WAS": (38.908, -76.864, -5, "cold", False),
    # older team codes in historical data
    "STL": (38.633, -90.188, -6, "cold", True),   "SD":  (32.783, -117.120, -8, "warm", False),
    "OAK": (37.751, -122.201, -8, "mild", False),
}


def miles_between(a: str, b: str) -> float:
    from math import radians, sin, cos, asin, sqrt
    la1, lo1, *_ = TEAM_INFO[a]
    la2, lo2, *_ = TEAM_INFO[b]
    la1, lo1, la2, lo2 = map(radians, (la1, lo1, la2, lo2))
    h = sin((la2 - la1) / 2) ** 2 + cos(la1) * cos(la2) * sin((lo2 - lo1) / 2) ** 2
    return 3959 * 2 * asin(sqrt(h))
