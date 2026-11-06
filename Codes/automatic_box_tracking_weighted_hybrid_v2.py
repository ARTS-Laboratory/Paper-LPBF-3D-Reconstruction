"""LPBF wall tracking: memory-efficient weighted centroid + gradient cross-check.

Fixes:
  * Weighting search includes both wall edges and OUTSIDE background pixels.
    A +/-14-pixel search could be wholly within a 61-pixel-wide wall.
  * Analyze a narrow, deskewed 16-bit strip, never a whole-image float64 frame.
  * REVIEW frames retain provisional drift numbers, but validated drift is blank.
  * Use an independent gradient detector as a fallback and flag disagreement.

Install: pip install numpy opencv-python matplotlib
Run: python automatic_box_tracking_weighted_hybrid_v2.py

IMPORTANT: Set ROOT, REFERENCE, OUTPUT, NOMINAL_WIDTH_MM.  Ensure the blue
reference, camera and optical images are spatially registered.  The nominal-
width pixel scale is NOT independently measured dimensional calibration.
"""
from __future__ import annotations

from pathlib import Path
import csv
import gc
import re

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------- USER SETTINGS ---------------------------
ROOT = Path(r"C:\Users\killedar\OneDrive - University of South Carolina\GitHub\Paper-LPBF-3D-Reconstruction\Test_01\processed")
REFERENCE = Path(r"C:\Users\killedar\OneDrive - University of South Carolina\GitHub\Paper-LPBF-3D-Reconstruction\Codes\Thin_wall_drift_tracking\Edge_tracking_output_0.8mm\reference_0.8mm.png")
OUTPUT = Path(r"C:\Users\killedar\OneDrive - University of South Carolina\GitHub\Paper-LPBF-3D-Reconstruction\Codes\Thin_wall_drift_tracking\Edge_tracking_output_0.8mm\weighted_hybrid_outputs_v2")
NOMINAL_WIDTH_MM = 0.8  # change to 0.8/0.7/0.6 for those samples
CALIBRATED_MM_PER_PX = None   # override if independently established
NOMINAL_LENGTH_MM = 40.0
EXPECTED_LAYER_COUNT = 40

TRACK_MODE = "hybrid"  # hybrid | weighted | edges
BLUR_KERNEL = 5
PROFILE_SIGMA = 2.0
TRIM_TOP, TRIM_BOTTOM = 15, 15

# The weighted search MUST extend beyond half of the 61-px reference width.
BACKGROUND_MARGIN_PX = 14    # actual background outside wall on each side
TRACKING_ALLOWANCE_PX = 14  # possible lateral step within a search
BACKGROUND_EDGE_FRACTION = 0.13
RESPONSE_THRESHOLD_FRACTION = 0.12
MIN_RESPONSE_SNR = 2.0
USE_SPATIAL_PRIOR = False
SPATIAL_PRIOR_SIGMA_PX = 32.0

EDGE_SEARCH_PX = 16
WIDTH_TOLERANCE_FRACTION = 0.40
WIDTH_TOLERANCE_MIN_PX = 3.0
METHOD_AGREEMENT_PX = 3.0
WEIGHTED_CENTER_FRACTION = 0.75
MAX_CENTER_STEP_PX = 12.0  # REVIEW threshold, NOT physical drift limit

# A REVIEW measurement can guide tracking as a *provisional* observation if
# its step is not excessive; do not count it as validated drift.
ADVANCE_ON_SMALL_REVIEW = True
SAVE_FULL_FRAME_OVERLAYS = False   # memory-expensive; strip overlays always saved
PLOT_PROFILE_EVERY = 1            # 0 disables
PANEL_SIZE = 7
Y_PREVIEW_MARGIN_PX = 18
STRIP_EXTRA_MARGIN_PX = 18

OUTPUT.mkdir(parents=True, exist_ok=True)


def discover_layers():
    if not ROOT.is_dir():
        raise FileNotFoundError(f"Missing directory: {ROOT}")
    found = []
    for p in ROOT.iterdir():
        m = re.fullmatch(r"optical_(\d+)\.png", p.name, re.IGNORECASE)
        if p.is_file() and m:
            found.append((int(m.group(1)), p))
    found.sort()
    if not found:
        raise FileNotFoundError(f"No optical_<number>.png images in {ROOT}")
    return [p for _, p in found]


def read_gray(path):
    """Read native grayscale; U16 remains U16 (lossless). No huge float arrays."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"Could not load: {path}")
    if im.ndim == 2:
        return im
    if im.shape[2] == 4:
        return cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def display_u8(img):
    """Fast U16->U8 only for SMALL strips, without large float64 allocations."""
    if img.dtype == np.uint8:
        return img
    # Quantiles on at most ~20,000 sample pixels, not a full 20-Mpixel image.
    stride = max(1, int(np.sqrt(img.size/20000)))
    sample = img[::stride, ::stride]
    lo, hi = np.percentile(sample, [0.5, 99.5])
    scale = 255.0 / max(float(hi-lo), 1.0)
    return cv2.convertScaleAbs(img, alpha=scale, beta=-float(lo)*scale)


def plot_slanted_box(img, center, width, y0, y1, geom, label, color=(255, 0, 0)):
    slope, mid = geom["slope"], geom["y_mid"]
    points = []
    for yy, sign in ((y0,-1),(y0,1),(y1,1),(y1,-1)):
        points.append([int(round(center+sign*width/2+slope*(yy-mid))), int(yy)])
    cv2.polylines(img, [np.asarray(points,np.int32)], True, color, 2, cv2.LINE_AA)
    cv2.putText(img, label, (max(5,min(img.shape[1]-250,points[0][0])),max(25,y0-14)),
                cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
    return img


def reference_geometry(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (85,40,40), (140,255,255))
    cv2.imwrite(str(OUTPUT / "reference_blue_mask.png"), mask)
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        raise RuntimeError("Insufficient reference blue pixels; inspect mask")
    counts = np.bincount(ys, minlength=mask.shape[0])
    left = np.full(mask.shape[0],mask.shape[1],dtype=int)
    right = np.full(mask.shape[0],-1,dtype=int)
    np.minimum.at(left,ys,xs)
    np.maximum.at(right,ys,xs)
    valid = (counts>=2)&(counts<=20)&(right-left>3)
    y = np.flatnonzero(valid).astype(float)
    l = left[valid].astype(float)
    r = right[valid].astype(float)
    if len(y) < 60:
        raise RuntimeError("Not enough rows containing two blue guides")
    sep = r-l
    valid = np.abs(sep-np.median(sep)) <= max(4.,.35*np.median(sep))
    y,l,r = y[valid],l[valid],r[valid]
    if len(y) < 60:
        raise RuntimeError("Blue-guide spacing inconsistent")
    slope = float((np.polyfit(y,l,1)[0]+np.polyfit(y,r,1)[0])/2)
    mid = float((y.min()+y.max())/2)
    center = float(np.median((l+r)/2-slope*(y-mid)))
    width = float(np.median(r-l)-1)
    if width <= 2:
        raise RuntimeError("Reference width invalid")
    geom = dict(slope=slope,y_mid=mid,center=center,width_px=width,
                y0=int(y.min()),y1=int(y.max())+1)
    preview = plot_slanted_box(bgr,center,width,geom["y0"],geom["y1"],geom,"Reference",(0,255,0))
    cv2.imwrite(str(OUTPUT/"reference_detected_slanted_guides.png"),preview)
    return geom


def extract_deskewed_strip(gray,geom,expected_center):
    """Warp only a narrow vertical stripe from a large U16 image.

    Deskewed global coord x' = x - slope*(y-y_mid).
    Local dest coord (u,v) maps back to original source:
         x_src = x_start+u+slope*(y_start+v-y_mid)
         y_src = y_start+v
    WARP_INVERSE_MAP specifies destination->source transform.
    """
    h,w = gray.shape[:2]
    half = int(np.ceil(geom["width_px"]*0.5 + BACKGROUND_MARGIN_PX +
                       TRACKING_ALLOWANCE_PX + EDGE_SEARCH_PX +
                       STRIP_EXTRA_MARGIN_PX))
    x_start = int(np.floor(expected_center-half))
    width = 2*half+2
    y_start = max(0,geom["y0"]-Y_PREVIEW_MARGIN_PX)
    y_end = min(h,geom["y1"]+Y_PREVIEW_MARGIN_PX)
    slope = geom["slope"]
    mat = np.array([[1.0,slope,x_start+slope*(y_start-geom["y_mid"])],
                    [0.0,1.0,float(y_start)]],dtype=np.float32)
    strip = cv2.warpAffine(gray,mat,(width,y_end-y_start),
                           flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return strip,x_start,y_start


def make_profile(strip,geom,y_start):
    y0 = geom["y0"]+TRIM_TOP-y_start
    y1 = geom["y1"]-TRIM_BOTTOM-y_start
    if not (0 <= y0 < y1 <= strip.shape[0]):
        raise RuntimeError("Invalid vertical ROI after trimming")
    roi = strip[y0:y1]
    blur = cv2.GaussianBlur(roi,(BLUR_KERNEL,BLUR_KERNEL),0)
    # float64 not required, and roi is only about 200 columns wide
    p = blur.mean(axis=0,dtype=np.float32)
    return cv2.GaussianBlur(p.reshape(1,-1),(0,0),sigmaX=PROFILE_SIGMA).ravel()


def weighted_centroid(profile,expected_center,ref_width):
    """Robust centroid from wall-versus-EXTERIOR background pixels."""
    radius = .5*ref_width+BACKGROUND_MARGIN_PX+TRACKING_ALLOWANCE_PX
    x0 = max(0,int(np.floor(expected_center-radius)))
    x1 = min(len(profile)-1,int(np.ceil(expected_center+radius)))
    if x1-x0 < 10:
        return None
    x = np.arange(x0,x1+1,dtype=np.float64)
    p = profile[x0:x1+1].astype(np.float64,copy=False) # ~100 pixels only
    q = min(len(p)//5,max(3,int(round(BACKGROUND_EDGE_FRACTION*len(p)))))
    exterior = np.r_[p[:q],p[-q:]]
    bg = float(np.median(exterior))
    bg_noise = float(1.4826*np.median(np.abs(exterior-bg)))
    signed = p-bg
    # Use polarity of strongest signed excursion, not abs of everything.
    # This limits distant backgrounds / opposite-polarity artifacts.
    polarity = 1 if np.max(signed) >= abs(np.min(signed)) else -1
    response = np.maximum(polarity*signed,0.)
    peak = float(np.max(response))
    if not np.isfinite(peak) or peak < 1e-8:
        return None
    threshold = max(RESPONSE_THRESHOLD_FRACTION*peak,1.5*bg_noise)
    weights = np.maximum(response-threshold,0.)
    if USE_SPATIAL_PRIOR:
        weights *= np.exp(-.5*((x-expected_center)/SPATIAL_PRIOR_SIGMA_PX)**2)
    total = float(weights.sum())
    if total <= 1e-12:
        return None
    c = float(np.dot(x,weights)/total)
    spread = float(np.sqrt(np.dot(weights,(x-c)**2)/total))
    snr = float(peak/max(bg_noise,1e-7))
    outer = max(2,min(4,len(weights)//9))
    edge_fraction = float((weights[:outer].sum()+weights[-outer:].sum())/total)
    good = bool(snr >= MIN_RESPONSE_SNR and edge_fraction < .30)
    return dict(center=c,spread=spread,snr=snr,good=good,
                background=bg,edge_fraction=edge_fraction,
                threshold=threshold,polarity=polarity)


def detect_gradient_edges(profile,expected_center,ref_width):
    g = np.gradient(profile.astype(np.float32))
    n = len(profile)
    el,er = expected_center-ref_width/2,expected_center+ref_width/2
    il = np.arange(max(1,int(el-EDGE_SEARCH_PX)),min(n-1,int(el+EDGE_SEARCH_PX)+1))
    ir = np.arange(max(1,int(er-EDGE_SEARCH_PX)),min(n-1,int(er+EDGE_SEARCH_PX)+1))
    if not len(il) or not len(ir):
        return None
    tol = max(WIDTH_TOLERANCE_MIN_PX,ref_width*WIDTH_TOLERANCE_FRACTION)
    local = g[max(0,int(el-EDGE_SEARCH_PX)):min(n,int(er+EDGE_SEARCH_PX)+1)]
    scale = max(float(np.median(np.abs(local))),1e-7)
    best = None
    for pol in (1,-1):
        ls = il[np.argsort(pol*g[il])[-min(12,len(il)):]]
        rs = ir[np.argsort(-pol*g[ir])[-min(12,len(ir)):]]
        for l in ls:
            for r in rs:
                ww = float(r-l)
                if ww <= 2 or abs(ww-ref_width)>tol:
                    continue
                c = float((l+r)/2)
                strength = float(pol*(g[l]-g[r]))
                if strength <= 0:
                    continue
                score = strength/scale-.75*abs(ww-ref_width)/tol-.7*abs(c-expected_center)/max(EDGE_SEARCH_PX,1)
                if best is None or score>best["score"]:
                    best = dict(center=c,left=float(l),right=float(r),width_px=ww,
                                score=score,polarity=pol)
    return best


def estimate(profile,expected_center,ref_width):
    wc = weighted_centroid(profile,expected_center,ref_width)
    ed = detect_gradient_edges(profile,expected_center,ref_width)
    wgood = wc is not None and wc["good"]
    egood = ed is not None
    agreement = (abs(wc["center"]-ed["center"]) if wc is not None and egood else np.nan)
    chosen,status,source = np.nan,"FAILED_NO_RESPONSE","none"
    if TRACK_MODE == "hybrid":
        if wgood and egood and agreement<=METHOD_AGREEMENT_PX:
            chosen = WEIGHTED_CENTER_FRACTION*wc["center"]+(1-WEIGHTED_CENTER_FRACTION)*ed["center"]
            status,source="OK","hybrid"
        elif wgood and egood:
            # Do not silently combine contradictory results.
            chosen,status,source = ed["center"],"REVIEW_METHOD_DISAGREEMENT","edge"
        elif egood:
            chosen,status,source = ed["center"],"REVIEW_WEIGHTED_WEAK","edge"
        elif wgood:
            chosen,status,source = wc["center"],"REVIEW_EDGE_NOT_FOUND","weighted"
        elif wc is not None:
            chosen,status,source = wc["center"],"REVIEW_BOTH_WEAK","weighted"
    elif TRACK_MODE == "weighted":
        if wc is not None:
            chosen,source=wc["center"],"weighted"
            status="OK" if wgood else "REVIEW_WEIGHTED_WEAK"
    elif TRACK_MODE == "edges":
        if egood:
            chosen,status,source=ed["center"],"OK","edge"
    else:
        raise ValueError(f"Unknown TRACK_MODE: {TRACK_MODE}")
    step = float(chosen-expected_center) if np.isfinite(chosen) else np.nan
    if np.isfinite(step) and abs(step)>MAX_CENTER_STEP_PX:
        status="REVIEW_LARGE_STEP"
    validated = bool(np.isfinite(chosen) and status=="OK")
    advance = bool(validated or (ADVANCE_ON_SMALL_REVIEW and np.isfinite(step) and
                                abs(step)<=MAX_CENTER_STEP_PX))
    return dict(center=float(chosen),status=status,source=source,
                step=step,agreement=agreement,validated=validated,advance=advance,
                weighted=wc,edge=ed)


def save_profile_plot(profile,expected_center,result,x_start,dest):
    x0=max(0,int(expected_center-75));x1=min(len(profile),int(expected_center+76))
    xx=np.arange(x0,x1)+x_start
    fig,ax=plt.subplots(figsize=(8,4.3))
    ax.plot(xx,profile[x0:x1],lw=1.2,label="Smoothed grayscale profile")
    ax.axvline(expected_center+x_start,color="gray",linestyle=":",label="Prior center")
    if result["weighted"] is not None:
        ax.axvline(result["weighted"]["center"]+x_start,ls="--",label="Weighted")
    if result["edge"] is not None:
        ax.axvline(result["edge"]["center"]+x_start,ls="--",label="Gradient edge")
    if np.isfinite(result["center"]):
        ax.axvline(result["center"]+x_start,ls="-",label="Selected")
    ax.set(xlabel="Deskewed global x (pixels)",ylabel="Mean grayscale intensity",title=result["status"])
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest,dpi=140);plt.close(fig)


def save_panel(crops,titles,path):
    if not crops:return
    fig,axes=plt.subplots(1,len(crops),figsize=(2.4*len(crops),6),constrained_layout=True)
    for ax,im,title in zip(np.atleast_1d(axes),crops,titles):
        ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        ax.set_title(title,fontsize=8);ax.axis("off")
    fig.savefig(path,dpi=110,bbox_inches="tight");plt.close(fig)


def chart(records,key,name,ylabel,ref=None):
    x=[r["image"] for r in records]
    y=[r.get(key,np.nan) for r in records]
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot(x,y,"o-",lw=1.5,ms=3)
    if ref is not None:ax.axhline(ref,ls="--",color="gray")
    ax.set(xlabel="Chronologically sorted image index (not necessarily one image per build layer)",ylabel=ylabel)
    ax.grid(alpha=.3);fig.tight_layout();fig.savefig(OUTPUT/name,dpi=220);plt.close(fig)


def main():
    paths=discover_layers()
    print(f"Found {len(paths)} optical images")
    if len(paths)!=EXPECTED_LAYER_COUNT:
        print(f"WARNING: Expected {EXPECTED_LAYER_COUNT} images")
    # Reference is normally RGB with blue lines: read color as is.
    ref=cv2.imread(str(REFERENCE),cv2.IMREAD_COLOR)
    if ref is None:
        raise FileNotFoundError(f"Reference image: {REFERENCE}")
    geom=reference_geometry(ref)
    ref_shape=ref.shape[:2]
    del ref
    gc.collect()
    scale=float(CALIBRATED_MM_PER_PX if CALIBRATED_MM_PER_PX is not None else
                NOMINAL_WIDTH_MM/geom["width_px"])
    print(f"Reference width: {geom['width_px']:.2f} px; horizontal scale {scale:.6f} mm/px")
    if CALIBRATED_MM_PER_PX is None:
        print("WARNING: Nominal-width-derived scale; independent calibration needed.")
    print(f"Reference slope dx/dy: {geom['slope']:.5f}; mode: {TRACK_MODE}")
    prev=geom["center"]
    baseline=None
    baseline_valid=False
    records=[];crops=[];titles=[]
    for idx,path in enumerate(paths,1):
        gray=read_gray(path)
        if gray.shape[:2]!=ref_shape:
            raise ValueError(f"{path.name}: {gray.shape[:2]} != reference {ref_shape}")
        prior=prev
        strip,x_start,y_start=extract_deskewed_strip(gray,geom,prior)
        profile=make_profile(strip,geom,y_start)
        est=estimate(profile,prior-x_start,geom["width_px"])
        wc,ed=est["weighted"],est["edge"]
        center=est["center"]+x_start if np.isfinite(est["center"]) else np.nan
        if idx==1 and np.isfinite(center):
            baseline=center
            baseline_valid=est["validated"]
        # Numeric provisional shifts are available on REVIEW frames; the
        # validated series is intentionally absent until image checks pass.
        provisional=(center-baseline)*scale if np.isfinite(center) and baseline is not None else np.nan
        validated=provisional if est["validated"] and baseline_valid else np.nan
        if est["advance"]:
            prev=center
        width_px=ed["width_px"] if ed is not None else np.nan
        row=dict(image=idx,filename=path.name,status=est["status"],source=est["source"],
                 validated=int(est["validated"]),prior_center_px=prior,
                 selected_center_px=center,
                 weighted_center_px=(wc["center"]+x_start if wc else np.nan),
                 gradient_center_px=(ed["center"]+x_start if ed else np.nan),
                 methods_disagreement_px=est["agreement"],step_vs_prior_px=est["step"],
                 shift_from_image1_mm_provisional=provisional,
                 shift_from_image1_mm_validated=validated,
                 shift_from_image1_px_provisional=(center-baseline if baseline is not None and np.isfinite(center) else np.nan),
                 detected_width_px=width_px,detected_width_mm=width_px*scale if np.isfinite(width_px) else np.nan,
                 fixed_box_width_mm=NOMINAL_WIDTH_MM,fixed_box_width_px=geom["width_px"],
                 weighted_snr=(wc["snr"] if wc else np.nan),
                 weighted_edge_response_fraction=(wc["edge_fraction"] if wc else np.nan),
                 gradient_score=(ed["score"] if ed else np.nan))
        records.append(row)
        print(f"Image {idx:02d} {path.name:25s} status={est['status']:28s} "
              f"weighted={row['weighted_center_px']:.2f} edge={row['gradient_center_px']:.2f} "
              f"provisional shift={provisional:+.4f} mm")
        if np.isfinite(center):
            disp=cv2.cvtColor(display_u8(strip),cv2.COLOR_GRAY2BGR)
            xlocal=int(round(center-x_start))
            half=int(round(geom["width_px"]/2))
            cv2.rectangle(disp,(xlocal-half,geom["y0"]-y_start),
                          (xlocal+half,geom["y1"]-y_start-1),
                          (255,0,0) if est["validated"] else (0,165,255),2)
            cv2.imwrite(str(OUTPUT/f"{path.stem}_image{idx:02d}_deskewed_strip.png"),disp)
            crops.append(disp);titles.append(f"{idx}: {est['status']}")
            if len(crops)>=PANEL_SIZE:
                save_panel(crops,titles,OUTPUT/f"panel_ending_{idx:02d}.png")
                crops=[];titles=[]
            if SAVE_FULL_FRAME_OVERLAYS:
                big=cv2.cvtColor(display_u8(gray),cv2.COLOR_GRAY2BGR)
                plot_slanted_box(big,center,geom["width_px"],geom["y0"],geom["y1"],
                                 geom,f"Image {idx}: {est['status']}",
                                 (255,0,0) if est["validated"] else (0,165,255))
                cv2.imwrite(str(OUTPUT/f"{path.stem}_image{idx:02d}_full_box.png"),big)
                del big
        if PLOT_PROFILE_EVERY and (idx-1)%PLOT_PROFILE_EVERY==0:
            save_profile_plot(profile,prior-x_start,est,x_start,
                              OUTPUT/f"{path.stem}_image{idx:02d}_profile.png")
        del gray,strip,profile
        if idx%10==0:gc.collect()
    if crops:save_panel(crops,titles,OUTPUT/"panel_final.png")
    if not records:return
    with (OUTPUT/"automatic_tracking_measurements.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0].keys()))
        writer.writeheader();writer.writerows(records)
    chart(records,"shift_from_image1_mm_provisional","shift_vs_image_mm_provisional.png","Provisional lateral shift (mm)",0.)
    chart(records,"shift_from_image1_mm_validated","shift_vs_image_mm_validated.png","Validated lateral shift (mm)",0.)
    chart(records,"detected_width_mm","detected_width_vs_image_mm.png","Apparent edge-detected width (mm)",NOMINAL_WIDTH_MM)
    chart(records,"methods_disagreement_px","weighted_vs_edge_disagreement.png","Weighted vs gradient center difference (px)")
    with (OUTPUT/"automatic_tracking_summary.txt").open("w",encoding="utf-8") as f:
        f.write(f"Nominal width: {NOMINAL_WIDTH_MM} mm; nominal visible length: {NOMINAL_LENGTH_MM} mm\n")
        f.write(f"Reference width: {geom['width_px']}px; scale={scale:.7f} mm/px\n")
        f.write(f"Images: {len(records)}; validated: {sum(r['validated'] for r in records)}\n")
        f.write("Provisional shifts are NOT verified physical drift; review status and image registration.\n")
        f.write("Blue-reference scale uses nominal width unless CALIBRATED_MM_PER_PX is set.\n")
        f.write("Image order is optical filename order, not automatically known consecutive layers.\n")
    print(f"\nSaved {len(records)} records to {OUTPUT}")
    print("Review orange strip overlays and compare provisional vs validated drift plots.")


if __name__=="__main__":
    main()
