import pymap3d as pm


def ecef_to_ned(gps_ecef, llh0, interp_ftns):
    ned_coords = {}
    lat0, lon0, h0, *_ = llh0
    for tns in interp_ftns:
        if tns in gps_ecef:
            x, y, z = gps_ecef[tns]
            n, e, d = pm.ecef2ned(x, y, z, lat0, lon0, h0)
            ned_coords[tns] = [n, e, d]
    return ned_coords

