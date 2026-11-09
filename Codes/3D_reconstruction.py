"""Integrated V2 prior-layer edge tracking and smooth CAD STEP reconstruction.

Reference image: reference_#Thinkess_mm.png, containing TWO narrow blue lines along a slanted wall.
The code uses the original mean-profile edge detection logic from code 'automatic_thinwall_tracking_v2_with_mean.py' .

"""
from pathlib import Path
from scipy.ndimage import median_filter, gaussian_filter1d
import cadquery as cq
from matplotlib.collections import PolyCollection
import re
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#ROOT = Path(__file__).resolve().parent
ROOT = Path(r"C:\Users\killedar\OneDrive - University of South Carolina\GitHub\Paper-LPBF-3D-Reconstruction\Test_01\processed")
REFERENCE = Path(r"C:\Users\killedar\OneDrive - University of South Carolina\GitHub\Paper-LPBF-3D-Reconstruction\Codes\Thin_wall_drift_tracking\Edge_tracking_output_0.3mm\reference_0.3mm.png")
#OUTPUT = Path.home() / "LPBF_Tracking_3D" / "Edge_tracking_output_0.2mm"  # avoid OneDrive plot-save errors
OUTPUT = Path(r"C:\Users\killedar\OneDrive - University of South Carolina\GitHub\Paper-LPBF-3D-Reconstruction\Codes\Thin_wall_drift_tracking\Edge_tracking_output_0.3mm\LPBF_3D_Reconstruction")
OUTPUT.mkdir(parents=True, exist_ok=True)
IMAGES= Path(r"C:\Users\killedar\OneDrive - University of South Carolina\GitHub\Paper-LPBF-3D-Reconstruction\Test_01\processed")
#OUTPUT.mkdir(exist_ok=True)
#REFERENCE = ROOT / 'reference_0.4mm.png'
#OUTPUT = ROOT / 'automatic_box_tracking_outputs_0.4mm'
#OUTPUT.mkdir(exist_ok=True)
NOMINAL_WIDTH_MM = 0.3
NOMINAL_LENGTH_MM = 40.0
EXPECTED_LAYER_COUNT = 40
BLUR_KERNEL = 5
PROFILE_SIGMA = 2.0
TRIM_TOP = 15
TRIM_BOTTOM = 15
EDGE_SEARCH_PX = 16       # expand cautiously if large drift, not to next wall
WIDTH_TOLERANCE_PX = 7  # reference appears about 11-13 pixels wide
MAX_CENTER_STEP_PX = 12 # flag, not a physical limit
PANEL_SIZE = 8


def discover_layers():
    items = []
    for p in ROOT.iterdir():
        if p.is_file():
            m = re.fullmatch(r'optical_(\d+)\.png', p.name, re.I)
            if m:
                items.append((int(m.group(1)), p))
    items.sort()
    if not items:
        raise FileNotFoundError(f'No optical_<number>.png files in {ROOT}')
    return [p for _, p in items]


def read_image(path):
    original = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if original is None:
        raise FileNotFoundError(path)
    if original.ndim == 2:
        gray = original
        if gray.dtype == np.uint8:
            disp = gray
        else:
            lo, hi = np.percentile(gray, [0.5, 99.5])
            disp = np.clip((gray.astype(np.float32) - lo) * (255.0 / max(hi-lo,1)), 0,255).astype(np.uint8)
        bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR) if original.shape[2] == 4 else original
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            lo, hi = np.percentile(gray,[0.5,99.5]); disp = np.clip((gray.astype(np.float32)-lo)*255/max(hi-lo,1),0,255).astype(np.uint8)
            bgr = cv2.cvtColor(disp,cv2.COLOR_GRAY2BGR)
    return gray, bgr


def reference_geometry(bgr):
    """Fit two blue strokes from their LEFTMOST and RIGHTMOST x for each valid y.

    Unlike column projections, works when each guide slopes hundreds of pixels.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (85, 40, 40), (140, 255, 255))
    cv2.imwrite(str(OUTPUT/'reference_blue_mask.png'),blue)
    ys, xs = np.nonzero(blue)
    if len(xs) < 100:
        raise RuntimeError('Insufficient blue pixels. Check reference_blue_mask.png')
    unique_ys, inverse, counts = np.unique(ys, return_inverse=True, return_counts=True)
    # For this reference there are two strokes, each 1-2 px thick per row.
    valid_ys = unique_ys[(counts >= 2) & (counts <= 20)]
    if len(valid_ys) < 60:
        raise RuntimeError('Could not identify enough rows with two blue guides. Inspect reference_blue_mask.png')
    lefts, rights, rows = [], [], []
    for y in valid_ys:
        coords = xs[ys == y]
        if coords.max() - coords.min() < 3:
            continue
        lefts.append(float(coords.min()))
        rights.append(float(coords.max()))
        rows.append(float(y))
    rows = np.asarray(rows); lefts = np.asarray(lefts); rights = np.asarray(rights)
    if len(rows) < 60:
        raise RuntimeError('Reference lacks two distinct blue guides on enough rows.')
    # Robustly remove stray pixels and short badly-detected rows.
    separation = rights - lefts
    median_sep = np.median(separation)
    valid = np.abs(separation-median_sep) <= max(4,0.35*median_sep)
    rows, lefts, rights = rows[valid], lefts[valid], rights[valid]
    if len(rows)<60:
        raise RuntimeError('Inconsistent blue-guide separation. Check reference mask.')
    slope_left, intercept_left = np.polyfit(rows,lefts,1)
    slope_right, intercept_right = np.polyfit(rows,rights,1)
    slope = float((slope_left+slope_right)/2)
    y_mid = float((np.min(rows)+np.max(rows))/2)
    # Inner edges: rightmost blue pixel of left stroke and leftmost of right stroke.
    # Since strokes may be 1-2px, use outer-gap estimate conservatively.
    blue_separation = float(np.median(rights-lefts))
    width_px = blue_separation - 1.0  # approximately one blue stroke pixel
    if width_px <= 2:
        raise RuntimeError(f'Blue guide gap too small ({width_px:.2f} px)')
    ref_center = float(np.median((lefts+rights)/2 - slope*(rows-y_mid)))
    y0 = int(np.min(rows)); y1 = int(np.max(rows))+1
    geom = {'slope':slope, 'y_mid':y_mid, 'center':ref_center,
            'width_px':width_px, 'y0':y0,'y1':y1}
    preview = bgr.copy()
    for yy in (y0,y1-1):
        xx=int(round(ref_center+slope*(yy-y_mid)))
        cv2.circle(preview,(xx,yy),6,(0,255,255),-1)
    plot_slanted_box(preview, ref_center, width_px, y0,y1,geom,'Reference',color=(0,255,0))
    cv2.imwrite(str(OUTPUT/'reference_detected_slanted_guides.png'), preview)
    return geom


def deskew(img, geom):
    """Horizontal shear, no rotation/cropping or rescaling: x_src=x_dst+slope*(y-y_mid)."""
    h,w=img.shape[:2]
    M=np.array([[1.0,-geom['slope'], geom['slope']*geom['y_mid']], [0,1,0]], dtype=np.float32)
    return cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def plot_slanted_box(img, center, width, y0,y1,geom,label,color=(255,0,0)):
    """Draw in ORIGINAL (unsheared) image pixel locations."""
    slope=geom['slope']; mid=geom['y_mid']
    corners=[]
    for yy, sign in ((y0,-1),(y0,1),(y1,1),(y1,-1)):
        corners.append([int(round(center+sign*width/2+slope*(yy-mid))),int(yy)])
    cv2.polylines(img,[np.asarray(corners,np.int32)],True,color,2,cv2.LINE_AA)
    xtext=max(5,min(img.shape[1]-200,corners[0][0]))
    cv2.putText(img,label,(xtext,max(25,y0-14)),cv2.FONT_HERSHEY_SIMPLEX,.62,color,2)
    return img


def profile_edges(gray, geom, expected_center):
    corrected=deskew(gray,geom)
    blur=cv2.GaussianBlur(corrected,(BLUR_KERNEL,BLUR_KERNEL),0).astype(np.float32)
    y0=geom['y0']+TRIM_TOP; y1=geom['y1']-TRIM_BOTTOM
    if y0>=y1 or y0<0 or y1>blur.shape[0]:
        raise ValueError('Reference y interval is outside layer image. Check camera alignment.')
    # Analyze only near the targeted wall, not the other parallel walls.
    profile=blur[y0:y1,:].mean(axis=0)
    profile=cv2.GaussianBlur(profile[np.newaxis,:],(0,0),sigmaX=PROFILE_SIGMA).ravel()
    grad=np.gradient(profile)
    w=geom['width_px']; n=len(profile)
    left_expect=expected_center-w/2; right_expect=expected_center+w/2
    radius=EDGE_SEARCH_PX
    li=np.arange(max(1,int(left_expect-radius)),min(n-1,int(left_expect+radius+1)))
    ri=np.arange(max(1,int(right_expect-radius)),min(n-1,int(right_expect+radius+1)))
    if not len(li) or not len(ri):
        raise RuntimeError('Search window outside image.')
    # A wall can be brighter OR darker than its background. Try both polarities.
    best=None
    for polarity in (1,-1):
        lg=polarity*grad[li]; rg=-polarity*grad[ri]
        for l in li[np.argsort(lg)[-min(12,len(li)):]]:
            for r in ri[np.argsort(rg)[-min(12,len(ri)):]]:
                width=float(r-l)
                if width<=2 or abs(width-w)>WIDTH_TOLERANCE_PX:
                    continue
                c=(l+r)/2
                strength=float(polarity*(grad[l]-grad[r]))
                score=strength - 0.8*abs(width-w) - 0.65*abs(c-expected_center)
                if best is None or score>best['score']:
                    best={'left':float(l),'right':float(r),'center':float(c),
                          'width_px':width,'score':score, 'polarity':polarity}
    if best is None:
        raise RuntimeError('No suitable wall edges near previous layer; inspect frame and adjust EDGE_SEARCH_PX/WIDTH_TOLERANCE_PX.')
    return best,profile


def save_panel(crops,titles,path):
    fig,axes=plt.subplots(1,len(crops),figsize=(2.5*len(crops),8),constrained_layout=True)
    for ax,img,title in zip(np.atleast_1d(axes),crops,titles):
        ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));ax.set_title(title,fontsize=8);ax.axis('off')
    fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)


def chart(x,y,name,ylabel,reference=None):
    fig,ax=plt.subplots(figsize=(8,4.5));ax.plot(x,y,'o-',lw=1.4,ms=3)
    if reference is not None:ax.axhline(reference,color='gray',ls='--')
    ax.set(xlabel='Layer number (numerically sorted optical filenames)',ylabel=ylabel)
    ax.grid(alpha=.3);fig.tight_layout();fig.savefig(OUTPUT/name,dpi=250);plt.close(fig)



def save_mean_shift_summary(rows, total_frames, shift_key, definition):
    """Summarize shifts, excluding the first frame's artificial zero baseline.

    Includes all finite shifts and separately the automatically accepted ('OK')
    shifts, so suspicious results cannot silently influence the validated mean.
    """
    transition_count = max(total_frames - 1, 0)
    data = []
    accepted = []
    for row in rows:
        if row.get('layer', 0) <= 1:
            continue
        value = row.get(shift_key, float('nan'))
        try:
            value = float(value)
        except (ValueError, TypeError):
            continue
        if not np.isfinite(value):
            continue
        data.append(value)
        if row.get('status') == 'OK':
            accepted.append(value)

    def stats(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return float('nan'), float('nan')
        return float(np.mean(values)), float(np.mean(np.abs(values)))

    raw_signed, raw_absolute = stats(data)
    accepted_signed, accepted_absolute = stats(accepted)

    text_lines = [
        'Mean wall shift across the image sequence',
        f'Shift definition: {definition}',
        f'Images processed: {total_frames}',
        f'Possible consecutive transitions: {transition_count}',
        'First image excluded: it has no preceding image (zero is only a plotting baseline).',
        f'Finite shift measurements: {len(data)}',
        f'Automatically accepted (status OK): {len(accepted)}',
        f'Mean signed shift (mm): {raw_signed:+.8f}',
        f'Mean absolute shift (mm): {raw_absolute:.8f}',
        #f'Mean signed shift, accepted only (mm): {accepted_signed:+.8f}',
        #f'Mean absolute shift, accepted only (mm): {accepted_absolute:.8f}',
        'Signed means can cancel positive and negative shifts.',
        'Absolute means measure average shift magnitude, regardless of direction.',
        'Review status is an automatic flag, not independent physical validation.',
    ]
    (OUTPUT / 'mean_shift_summary.txt').write_text('\n'.join(text_lines) + '\n', encoding='utf-8')

    records = [
        ('all_finite', len(data), raw_signed, raw_absolute),
        ('automatically_accepted', len(accepted), accepted_signed, accepted_absolute),
    ]
    with (OUTPUT / 'mean_shift_summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['category', 'number_of_shifts', 'possible_shifts',
                         'mean_signed_shift_mm', 'mean_absolute_shift_mm', 'definition'])
        for label, count, mean_signed, mean_abs in records:
            writer.writerow([label, count, transition_count, mean_signed, mean_abs, definition])

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(2)
    width = 0.35
    a = [raw_signed, accepted_signed]
    b = [raw_absolute, accepted_absolute]
    ax.bar(x - width/2, a, width, label='Mean signed shift')
    ax.bar(x + width/2, b, width, label='Mean absolute shift')
    ax.axhline(0, color='gray', lw=0.9)
    ax.set_xticks(x, [f'All finite\n(n={len(data)})',
                      f'Automatically accepted\n(n={len(accepted)})'])
    ax.set_ylabel('Average shift (mm)')
    ax.set_title(f'Mean wall shift across {total_frames} frames\n{definition}')
    ax.legend()
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    #fig.savefig(OUTPUT / 'mean_shift_summary.png', dpi=250)
    plt.close(fig)

    print(f'\nMean-shift summary ({definition}; {total_frames} frames):')
    print(f'  All finite: n={len(data)}/{transition_count}; '
          f'signed={raw_signed:+.6f} mm; absolute={raw_absolute:.6f} mm')
    print(f'  Automatically accepted: n={len(accepted)}/{transition_count}; '
          f'signed={accepted_signed:+.6f} mm; absolute={accepted_absolute:.6f} mm')
    return text_lines



# ====================== RECONSTRUCTION SETTINGS =============================
# No external tracking CSV required. V2 tracks all frames in the same run.
LAYER_THICKNESS_MM = 0.030
# Optional independently verified physical pixel pitches. If None, the nominal
# reference width and blue-guide length determine scale (not metrology).
PIXEL_SIZE_X_MM = None
PIXEL_SIZE_Y_MM = None

# False calls your EXACT original V2 profile_edges function on whole images.
# True uses the SAME V2 gradient candidates/penalties/previous-layer prior,
# but computes the mean profile from a deskewed narrow strip to save RAM.
MEMORY_EFFICIENT_PROFILE = True

# 'rowwise' reconstructs both measured contours along the visible wall length.
# 'layer_average' extrudes the exact single V2 edge pair along that length.
RECONSTRUCTION_MODE = 'rowwise'
ROW_SAMPLE_STEP_PX = 4
ROW_EDGE_SEARCH_PX = 12
ROW_MEDIAN_WINDOW_PX = 15
ROW_SMOOTH_SIGMA_PX = 3.0
MIN_ROW_VALID_FRACTION = 0.30
ROW_MIN_WIDTH_FACTOR = 0.45
ROW_MAX_WIDTH_FACTOR = 1.60

# If local contours cannot be verified, use the V2 layer-average pair and FLAG
# the layer explicitly. Set False to stop instead of using a coarse fallback.
ALLOW_FLAGGED_ROW_FALLBACK = True
# Refuse to export an invalid 3D shape if global tracking entirely failed.
STOP_IF_TRACKING_FAILED = True
SAVE_EDGE_OVERLAYS = True
# CAD smoothing changes local sub-layer edge roughness. The mean left and
# right edge of every measured layer is re-anchored to the V2 result.
CAD_LENGTH_STATIONS = 65             # spline interpolation points along 40 mm
CAD_LENGTH_SMOOTH_SIGMA_MM = 0.15     # 0 turns off longitudinal smoothing
CAD_BUILD_SMOOTH_SIGMA_LAYERS = 0.55  # 0 turns off interlayer smoothing
CAD_MIN_WIDTH_MM = 0.003             # positive-thickness sanity check
CAD_SAVE_PREVIEW = False             # optional PNG; STEP itself is never triangulated
CAD_PREVIEW_TESSELLATION_MM = 0.12    # display tessellation only, never exported as STL
RECON_DIR = OUTPUT / 'measured_3d_STEP_reconstruction'
# ============================================================================


def read_gray_unchanged(path):
    """Avoid creating full-frame 8-bit BGR copies for 16-bit optical images."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    if im.ndim == 3:
        if im.shape[2] == 4:
            return cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def deskewed_strip(gray, geom, x_start, x_stop, y_start, y_stop):
    """Sample x_dst at original x_src=x_dst+slope*(y-y_mid), on small ROI."""
    yy=np.arange(int(y_start),int(y_stop),dtype=np.float32)
    xx=np.arange(int(x_start),int(x_stop),dtype=np.float32)
    if len(yy)<4 or len(xx)<5:
        raise RuntimeError('Small deskew window; inspect reference ROI.')
    map_x=xx[None,:]+float(geom['slope'])*(yy[:,None]-float(geom['y_mid']))
    map_y=np.broadcast_to(yy[:,None],map_x.shape).copy()
    strip=cv2.remap(gray,map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return strip, yy.astype(float), xx.astype(float)


def profile_edges_low_memory(gray,geom,expected_center):
    """V2 profile/gradient algorithm evaluated in narrow deskewed field.

    Same: 5x5 blur, length-average profile, sigma=PROFILE_SIGMA,
    left/right gradient candidates, 0.8 width and 0.65 center penalties.
    """
    w=float(geom['width_px'])
    radius=EDGE_SEARCH_PX
    padding=max(20,int(4*PROFILE_SIGMA)+8)
    x0=int(np.floor(expected_center-w/2-radius-padding))
    x1=int(np.ceil(expected_center+w/2+radius+padding))+1
    y0=int(geom['y0']+TRIM_TOP)
    y1=int(geom['y1']-TRIM_BOTTOM)
    if y0>=y1 or y0<0 or y1>gray.shape[0]:
        raise ValueError('Invalid reference y interval relative to optical image.')
    # Include two extra rows to keep 5x5 Gaussian treatment identical to
    # the full-frame implementation in the measured region.
    strip,_,xx=deskewed_strip(gray,geom,x0,x1,y0-2,y1+2)
    blurred=cv2.GaussianBlur(strip,(BLUR_KERNEL,BLUR_KERNEL),0)
    profile=blurred[2:-2].astype(np.float32).mean(axis=0)
    profile=cv2.GaussianBlur(profile[np.newaxis,:],(0,0),sigmaX=PROFILE_SIGMA).ravel()
    grad=np.gradient(profile)
    le=expected_center-w/2
    re=expected_center+w/2
    li=np.arange(max(1,int(le-radius)),min(gray.shape[1]-1,int(le+radius+1)))
    ri=np.arange(max(1,int(re-radius)),min(gray.shape[1]-1,int(re+radius+1)))
    li=li[(li>=x0+2)&(li<x1-2)]
    ri=ri[(ri>=x0+2)&(ri<x1-2)]
    if not len(li) or not len(ri):
        raise RuntimeError('V2 edge windows outside local deskew strip.')
    best=None
    for polarity in (1,-1):
        lg=polarity*grad[li-x0]
        rg=-polarity*grad[ri-x0]
        for l in li[np.argsort(lg)[-min(12,len(li)):]]:
            for r in ri[np.argsort(rg)[-min(12,len(ri)):]]:
                width=float(r-l)
                if width<=2 or abs(width-w)>WIDTH_TOLERANCE_PX:
                    continue
                c=(l+r)/2.0
                strength=float(polarity*(grad[l-x0]-grad[r-x0]))
                score=strength-0.8*abs(width-w)-0.65*abs(c-expected_center)
                if best is None or score>best['score']:
                    best={'left':float(l),'right':float(r),'center':float(c),
                          'width_px':width,'score':score,'polarity':polarity}
    if best is None:
        raise RuntimeError('No V2 edge pair near previous layer; inspect image.')
    return best,profile


def rowwise_contours(gray,geom,measurement,sample_ys):
    """Refine contours while preserving the EXACT V2 layer mean edge pair.

    Candidate neighborhoods come from V2-measured left/right, never from
    an independently re-estimated center or a forced nominal thickness.
    """
    left0=float(measurement['left']);right0=float(measurement['right'])
    y0=int(geom['y0']);y1=int(geom['y1'])
    radius=max(ROW_EDGE_SEARCH_PX,int(round(0.20*(right0-left0))))
    padding=radius+20
    x0=int(np.floor(left0-padding));x1=int(np.ceil(right0+padding))+1
    strip,ys,xs=deskewed_strip(gray,geom,x0,x1,y0,y1)
    smooth=cv2.GaussianBlur(strip,(5,9),0).astype(np.float32)
    gradient=np.gradient(smooth,axis=1)
    polarity=int(measurement['polarity'])
    li=np.flatnonzero(np.abs(xs-left0)<=radius)
    ri=np.flatnonzero(np.abs(xs-right0)<=radius)
    if len(li)<3 or len(ri)<3:
        raise RuntimeError('Insufficient row-wise candidate pixels.')
    # Follow the same edge-gradient polarity as V2; do not re-estimate it.
    # Low-confidence rows will be interpolated from detected neighbors.
    strength_scale=max(1e-5,float(np.percentile(np.abs(gradient),95)))
    penalty_left=0.25*strength_scale*(np.abs(xs[li]-left0)/radius)
    penalty_right=0.25*strength_scale*(np.abs(xs[ri]-right0)/radius)
    ls=polarity*gradient[:,li]-penalty_left[None,:]
    rs=-polarity*gradient[:,ri]-penalty_right[None,:]
    il=li[np.argmax(ls,axis=1)]
    ir=ri[np.argmax(rs,axis=1)]
    left=xs[il].copy();right=xs[ir].copy()
    width=right-left
    coarse_width=right0-left0
    row_strength=np.minimum(polarity*gradient[np.arange(len(ys)),il],
                            -polarity*gradient[np.arange(len(ys)),ir])
    valid=(row_strength>0.02*strength_scale)&(width>ROW_MIN_WIDTH_FACTOR*coarse_width)&(width<ROW_MAX_WIDTH_FACTOR*coarse_width)
    valid_fraction=float(valid.mean())
    if valid_fraction < MIN_ROW_VALID_FRACTION or np.count_nonzero(valid)<5:
        raise RuntimeError(f'Insufficient credible rowwise edge detections: {valid_fraction:.1%}.')
    left=np.interp(ys,ys[valid],left[valid]);right=np.interp(ys,ys[valid],right[valid])
    window=ROW_MEDIAN_WINDOW_PX|1
    left=gaussian_filter1d(median_filter(left,size=window),ROW_SMOOTH_SIGMA_PX)
    right=gaussian_filter1d(median_filter(right,size=window),ROW_SMOOTH_SIGMA_PX)
    l=np.interp(sample_ys,ys,left);r=np.interp(sample_ys,ys,right)
    # Critical consistency constraint: if V2 reports edges L_i and R_i,
    # row-wise refined edges average exactly to those same V2 edges.
    l+=left0-float(l.mean())
    r+=right0-float(r.mean())
    if np.any(r-l<=0.2):
        raise RuntimeError('Row-wise contours crossed after V2 mean anchoring.')
    diagnostics={'valid_row_fraction':valid_fraction,
                 'mean_width_px':float(np.mean(r-l)),
                 'std_width_px':float(np.std(r-l)),
                 'rowwise_status':'ROW_REFINED'}
    return l,r,diagnostics,strip,xs


def save_contour_overlay(strip, sample_ys, left, right, xs, geom, center, path):
    """Compact optical crop showing measured edges (red/blue) and V2 center."""
    small=strip[::2]
    # Sample a small subset for percentile estimation, avoiding huge arrays.
    lo,hi=np.percentile(small[::5,::3],[1,99])
    scale=255.0/max(float(hi-lo),1.0)
    display=np.clip((small.astype(np.float32)-lo)*scale,0,255).astype(np.uint8)
    rgb=cv2.cvtColor(display,cv2.COLOR_GRAY2BGR)
    for y,l,r in zip(sample_ys,left,right):
        yy=int(round((y-geom['y0'])/2))
        if not 0<=yy<rgb.shape[0]:continue
        xl=int(round(l-xs[0]));xr=int(round(r-xs[0]))
        if 0<=xl<rgb.shape[1]:cv2.circle(rgb,(xl,yy),1,(0,0,255),-1)
        if 0<=xr<rgb.shape[1]:cv2.circle(rgb,(xr,yy),1,(255,0,0),-1)
    cx=int(round(center-xs[0]))
    if 0<=cx<rgb.shape[1]: cv2.line(rgb,(cx,0),(cx,rgb.shape[0]-1),(0,255,0),1)
    if not cv2.imwrite(str(path),rgb):
        raise OSError(f'Could not save overlay: {path}')


def _smooth_for_cad(left_mm, right_mm, y_mm):
    """Prepare continuous measured curves, retaining V2 mean edges per layer.

    This is *not* a nominal-width correction. Width and drift vary by layer.
    Smoothing can suppress real small-scale variation; tune it with overlays.
    """
    left = np.asarray(left_mm, dtype=np.float64).copy()
    right = np.asarray(right_mm, dtype=np.float64).copy()
    y = np.asarray(y_mm, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] < 4:
        raise ValueError('Expected 2D left/right contours [layers, y samples].')
    if len(y) != left.shape[1] or np.any(np.diff(y) <= 0):
        raise ValueError('Wall-length coordinates must increase strictly.')
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise RuntimeError('Contour data contain NaN/Inf; inspect tracking output.')
    if np.any(right <= left):
        raise RuntimeError('Detected left/right edges cross before CAD processing.')
    mean_left = left.mean(axis=1)
    mean_right = right.mean(axis=1)
    dy = float(np.median(np.diff(y)))
    if CAD_LENGTH_SMOOTH_SIGMA_MM > 0:
        sigma_samples = CAD_LENGTH_SMOOTH_SIGMA_MM / dy
        left = gaussian_filter1d(left, sigma_samples, axis=1, mode='nearest')
        right = gaussian_filter1d(right, sigma_samples, axis=1, mode='nearest')
    if CAD_BUILD_SMOOTH_SIGMA_LAYERS > 0:
        left = gaussian_filter1d(left, CAD_BUILD_SMOOTH_SIGMA_LAYERS,
                                 axis=0, mode='nearest')
        right = gaussian_filter1d(right, CAD_BUILD_SMOOTH_SIGMA_LAYERS,
                                  axis=0, mode='nearest')
    # Retain each original detected layer's mean edges/width/center EXACTLY.
    left += (mean_left - left.mean(axis=1))[:, None]
    right += (mean_right - right.mean(axis=1))[:, None]
    target_y = np.linspace(float(y[0]), float(y[-1]), int(CAD_LENGTH_STATIONS))
    left_fit = np.stack([np.interp(target_y, y, row) for row in left])
    right_fit = np.stack([np.interp(target_y, y, row) for row in right])
    # Account for sampling when re-anchoring CAD station curves too.
    left_fit += (mean_left - left_fit.mean(axis=1))[:, None]
    right_fit += (mean_right - right_fit.mean(axis=1))[:, None]
    if np.min(right_fit - left_fit) <= CAD_MIN_WIDTH_MM:
        raise RuntimeError('CAD smoothing caused a non-positive or too-thin region. '
                           'Reduce CAD smoothing and inspect source overlays.')
    return left_fit, right_fit, target_y


def _section_wire(left_row, right_row, y_mm, z_mm):
    """Closed planar wire: smooth left edge + two straight ends + smooth right."""
    left_points = [cq.Vector(float(x), float(y), float(z_mm))
                   for x, y in zip(left_row, y_mm)]
    right_points = [cq.Vector(float(x), float(y), float(z_mm))
                    for x, y in zip(right_row, y_mm)]
    edges = [
        cq.Edge.makeSpline(left_points),
        cq.Edge.makeLine(left_points[-1], right_points[-1]),
        cq.Edge.makeSpline(right_points[::-1]),
        cq.Edge.makeLine(right_points[0], left_points[0]),
    ]
    wire = cq.Wire.assembleEdges(edges)
    if not wire.IsClosed():
        raise RuntimeError(f'CAD wire at build height {z_mm:.4f} mm is open.')
    return wire


def loft_smooth_cad_step(left_mm, right_mm, y_mm, layer_thickness_mm):
    """Build an OCC solid B-rep by lofting spline wires; NO STL/mesh is made.

    Each measured layer is a section at its slab midpoint. Extra end sections
    close the solid at z=0 and z=n*thickness, ensuring correct build height.
    Non-ruled ThruSections yields smooth BSpline faces, not triangular facets.
    """
    l, r, yy = _smooth_for_cad(left_mm, right_mm, y_mm)
    n = len(l)
    if n < 2:
        raise RuntimeError('At least two measured layers required for smooth loft.')
    h = float(layer_thickness_mm)
    zs = [0.0] + [(i + 0.5) * h for i in range(n)] + [n * h]
    ls = [l[0]] + list(l) + [l[-1]]
    rs = [r[0]] + list(r) + [r[-1]]
    wires = [_section_wire(L, R, yy, z) for L, R, z in zip(ls, rs, zs)]
    print(f'CAD: lofting {len(wires)} smooth closed planar sections...')
    solid = cq.Solid.makeLoft(wires, ruled=False)
    if not solid.isValid() or len(solid.Solids()) != 1 or solid.Volume() <= 0:
        raise RuntimeError('STEP B-rep is not a valid single solid. '
                           'Try reducing CAD smoothing or station count.')
    bounds = solid.BoundingBox()
    if abs(bounds.zlen - n * h) > max(1e-3, 0.02*n*h):
        raise RuntimeError(f'Unexpected CAD height {bounds.zlen} mm; '
                           f'expected {n*h} mm.')
    return solid, (l, r, yy)


def save_gray_cad_preview(solid, path):
    """Save a shaded preview; tessellation is for rendering ONLY, not a mesh file."""
    nodes, facets = solid.tessellate(float(CAD_PREVIEW_TESSELLATION_MM), 0.25)
    if not facets:
        raise RuntimeError('CAD solid preview has no renderable facets.')
    pts = np.array([[p.x, p.y, p.z] for p in nodes], dtype=float)
    tri = np.asarray(facets, dtype=int)
    points = pts[tri]
    ext = solid.BoundingBox()
    # Visual exaggeration only; STEP coordinates are NOT scaled.
    scale = np.array([max(1., 7. / max(ext.xlen, 0.10)),
                      1., max(1., 5. / max(ext.zlen, 0.03))])
    position = (points - pts.mean(axis=0)) * scale
    az, el = np.deg2rad(-62), np.deg2rad(27)
    u = np.array([np.cos(az), -np.sin(az), 0])
    v = np.array([-np.sin(el)*np.sin(az), -np.sin(el)*np.cos(az), np.cos(el)])
    camera = np.array([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)])
    projected = np.stack((position@u, position@v), axis=-1)
    order = np.argsort((position@camera).mean(axis=1))
    # Smooth gray shading from local normals; NEVER draw triangle edges.
    na = np.cross(points[:,1]-points[:,0], points[:,2]-points[:,0])
    norms = np.maximum(np.linalg.norm(na, axis=1, keepdims=True), 1e-15)
    cos = np.abs((na/norms) @ np.array([0.2, -0.5, 0.84]))
    shade = np.clip(0.44 + 0.37*cos, 0.0, 1.0)
    colors = np.stack((shade, shade, shade), axis=1)
    fig, ax = plt.subplots(figsize=(13, 7))
    patches = PolyCollection(projected[order], facecolors=colors[order],
                             edgecolors='none', linewidths=0,
                             antialiaseds=False)
    ax.add_collection(patches)
    xy = projected.reshape(-1,2)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    pad = .06 * max(float(np.max(hi-lo)), 1.)
    ax.set_xlim(lo[0]-pad, hi[0]+pad)
    ax.set_ylim(lo[1]-pad, hi[1]+pad)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Smooth measured-edge LPBF wall (CAD STEP B-rep preview)\n'
                 'Thickness/build-height visually exaggerated', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_reconstruction_tables(out_dir,paths,tracking,contour_ys,left_px,right_px,mm_per_px,y_mm,diagnostics):
    with (out_dir/'reconstruction_edge_profiles.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f)
        writer.writerow(['layer','filename','sample_y_px','length_y_mm','left_deskewed_px',
                         'right_deskewed_px','center_deskewed_px','width_mm'])
        for k,p in enumerate(paths):
            for j,y in enumerate(contour_ys):
                l=float(left_px[k,j]);r=float(right_px[k,j])
                writer.writerow([k+1,p.name,float(y),float(y_mm[j]),l,r,(l+r)/2,
                                 (r-l)*mm_per_px])
    with (out_dir/'reconstruction_per_layer_summary.csv').open('w',newline='',encoding='utf8') as f:
        fields=['layer','filename','status','center_deskewed_px',
                'shift_from_previous_layer_mm','coarse_width_mm','refined_average_width_mm',
                'refined_width_std_mm','valid_row_fraction','rowwise_status']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for k,(row,diag) in enumerate(zip(tracking,diagnostics)):
            writer.writerow({'layer':k+1,'filename':paths[k].name,
                  'status':row['status'],'center_deskewed_px':row['center_deskewed_px'],
                  'shift_from_previous_layer_mm':row.get('shift_from_previous_layer_mm',float('nan')),
                  'coarse_width_mm':row['detected_width_mm'],
                  'refined_average_width_mm':np.mean(right_px[k]-left_px[k])*mm_per_px,
                  'refined_width_std_mm':np.std(right_px[k]-left_px[k])*mm_per_px,
                  'valid_row_fraction':diag.get('valid_row_fraction',0),
                  'rowwise_status':diag['rowwise_status']})


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    RECON_DIR.mkdir(parents=True,exist_ok=True)
    overlay_dir=RECON_DIR/'edge_overlays'
    if SAVE_EDGE_OVERLAYS: overlay_dir.mkdir(parents=True,exist_ok=True)
    if not REFERENCE.is_file():
        raise FileNotFoundError(f'Reference PNG not found: {REFERENCE}\n'
                                'Check 1mm versus 1.0mm folder and image name.')
    layers=discover_layers()
    if len(layers)!=EXPECTED_LAYER_COUNT:
        raise RuntimeError(f'Expected {EXPECTED_LAYER_COUNT} layers but found {len(layers)}. '
                           'Do not silently omit building layers in the 3D model.')
    _,ref_bgr=read_image(REFERENCE)
    reference_shape=ref_bgr.shape[:2]
    geom=reference_geometry(ref_bgr)
    del ref_bgr
    mm_per_px=(NOMINAL_WIDTH_MM/geom['width_px'] if PIXEL_SIZE_X_MM is None
               else float(PIXEL_SIZE_X_MM))
    mm_y=(NOMINAL_LENGTH_MM/max(1,(geom['y1']-geom['y0']-1)) if PIXEL_SIZE_Y_MM is None
          else float(PIXEL_SIZE_Y_MM))
    y0,y1=int(geom['y0']),int(geom['y1'])
    sample_y=np.arange(y0,y1,max(1,int(ROW_SAMPLE_STEP_PX)),dtype=float)
    if len(sample_y)==0 or sample_y[-1]!=y1-1:
        sample_y=np.r_[sample_y,float(y1-1)]
    y_mm=(sample_y-y0)*mm_y
    prev=float(geom['center']);previous_index=None
    rows=[];diagnostics=[];left_profiles=[];right_profiles=[]
    print(f'Found {len(layers)} images: {layers[0].name} to {layers[-1].name}')
    print(f'V2 reference width={geom["width_px"]:.2f} px; xscale={mm_per_px:.8f} mm/px')
    print(f'V2 slope={geom["slope"]:.6f}; 3D wall length={y_mm[-1]:.3f} mm; '
          f'build height={len(layers)*LAYER_THICKNESS_MM:.3f} mm')
    for i,p in enumerate(layers,1):
        gray=read_gray_unchanged(p)
        if gray.shape[:2] != reference_shape:
            raise ValueError(f"{p.name}: image shape {gray.shape[:2]} != reference {reference_shape}")
        try:
            if MEMORY_EFFICIENT_PROFILE:
                measurement,_=profile_edges_low_memory(gray,geom,prev)
            else:
                measurement,_=profile_edges(gray,geom,prev)   # exact V2 function
        except RuntimeError as e:
            rows.append({'layer':i,'filename':p.name,'status':'FAILED'})
            print(f'Layer {i:02d} tracking FAILED: {e}')
            if STOP_IF_TRACKING_FAILED:
                raise RuntimeError(f'Reconstruction aborted: V2 tracking failed at layer {i}.') from e
            continue
        c=float(measurement['center'])
        if i==1:
            step_px=0.0;status='FIRST_FRAME'
        elif previous_index==i-1:
            step_px=c-prev
            status='REVIEW large step' if abs(step_px)>MAX_CENTER_STEP_PX else 'OK'
        else:
            step_px=float('nan');status='REVIEW missing immediately previous detection'
        record={'layer':i,'filename':p.name,'status':status,
                'center_deskewed_px':c,'shift_from_previous_layer_px':step_px,
                'shift_from_previous_layer_mm':step_px*mm_per_px,
                'step_from_previous_px':step_px,
                'detected_width_px':measurement['width_px'],
                'detected_width_mm':measurement['width_px']*mm_per_px,
                'fixed_box_width_px':geom['width_px'],
                'fixed_box_width_mm':NOMINAL_WIDTH_MM,
                'blue_guide_y_span_px':y1-y0,
                'edge_left_deskewed_px':measurement['left'],
                'edge_right_deskewed_px':measurement['right'],
                'profile_score':measurement['score']}
        if RECONSTRUCTION_MODE=='layer_average':
            l=np.full(sample_y.shape,float(measurement['left']))
            r=np.full(sample_y.shape,float(measurement['right']))
            diag={'valid_row_fraction':float('nan'),'rowwise_status':'LAYER_AVERAGE_ONLY'}
            strip=None;xs=None
        elif RECONSTRUCTION_MODE=='rowwise':
            try:
                l,r,diag,strip,xs=rowwise_contours(gray,geom,measurement,sample_y)
            except RuntimeError as e:
                if not ALLOW_FLAGGED_ROW_FALLBACK:
                    raise RuntimeError(f'Layer {i} rowwise edge extraction failed: {e}') from e
                l=np.full(sample_y.shape,float(measurement['left']))
                r=np.full(sample_y.shape,float(measurement['right']))
                diag={'valid_row_fraction':0.0,'rowwise_status':'FALLBACK_COARSE_EDGES'}
                print(f'  REVIEW layer {i}: fallback to V2 average edges: {e}')
                strip=None;xs=None
        else:
            raise ValueError('RECONSTRUCTION_MODE must be rowwise or layer_average')
        if SAVE_EDGE_OVERLAYS:
            if strip is None:
                pad=max(24,ROW_EDGE_SEARCH_PX+12)
                strip,_,xs=deskewed_strip(gray,geom,
                       int(np.floor(measurement['left']-pad)),
                       int(np.ceil(measurement['right']+pad))+1,y0,y1)
            save_contour_overlay(strip,sample_y,l,r,xs,geom,c,
                                 overlay_dir/f'{p.stem}_edges.png')
        rows.append(record);left_profiles.append(l);right_profiles.append(r);diagnostics.append(diag)
        prev=c;previous_index=i
        print(f'Layer {i:02d}: V2 center={c:.2f} px | width={record["detected_width_mm"]:.4f} mm | '
              f'incremental shift={record["shift_from_previous_layer_mm"]:+.5f} mm | '
              f'{diag["rowwise_status"]} | {status}')
        del gray
    if len(left_profiles)!=len(layers):
        raise RuntimeError('Missing layer(s): cannot make a measured 40-layer solid safely.')
    # Save exact V2 statistics/graphs and their tracking table.
    with (OUTPUT/'automatic_tracking_measurements.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    x_index=[r['layer'] for r in rows]
    chart(x_index,[r['shift_from_previous_layer_mm'] for r in rows],
          'shift_vs_layer_mm.png','Incremental shift: current minus previous layer (mm)',0)
    chart(x_index,[r['shift_from_previous_layer_px'] for r in rows],
          'shift_vs_layer_px.png','Incremental shift: current minus previous layer (pixels)',0)
    chart(x_index,[r['detected_width_mm'] for r in rows],
          'detected_width_vs_layer_mm.png','Detected wall width (mm)',NOMINAL_WIDTH_MM)
    save_mean_shift_summary(rows,len(layers),'shift_from_previous_layer_mm',
                            'Incremental displacement from the immediately preceding frame')
    left_px=np.stack(left_profiles,axis=0)
    right_px=np.stack(right_profiles,axis=0)
    # The blue guide center is the x=0 visualization origin only. It is not
    # used as a second drift reference; tracked centers come from V2 layer steps.
    left_mm=(left_px-float(geom['center']))*mm_per_px
    right_mm=(right_px-float(geom['center']))*mm_per_px
    solid, (cad_l, cad_r, cad_y) = loft_smooth_cad_step(
        left_mm, right_mm, y_mm, LAYER_THICKNESS_MM)
    step_file = RECON_DIR / f'lpbf_wall_measured_edges_{len(layers)}layers_smooth.step'
    cq.exporters.export(solid, str(step_file), exportType='STEP', unit='MM')
    print(f'Wrote genuine CAD STEP solid: {step_file}')
    np.savez_compressed(RECON_DIR/'measured_contours.npz',
                        sample_y_px=sample_y, y_mm=y_mm,
                        left_deskewed_px=left_px, right_deskewed_px=right_px,
                        left_mm=left_mm, right_mm=right_mm,
                        cad_left_mm=cad_l, cad_right_mm=cad_r, cad_y_mm=cad_y,
                        layer_thickness_mm=LAYER_THICKNESS_MM)
    save_reconstruction_tables(RECON_DIR, layers, rows, sample_y,
                               left_px, right_px, mm_per_px, y_mm, diagnostics)
    if CAD_SAVE_PREVIEW:
        try:
            save_gray_cad_preview(solid,
                RECON_DIR/f'lpbf_wall_measured_edges_{len(layers)}layers_smooth_gray.png')
        except Exception as exc:
            print(f'Preview was skipped (STEP file is intact): {exc}')
    bounds=solid.BoundingBox()
    message=(f'Exported CAD B-rep solid: valid={solid.isValid()}, '
             f'solids={len(solid.Solids())}, faces={len(solid.Faces())}, '
             f'volume={solid.Volume():.6f} mm^3, '
             f'height={bounds.zlen:.4f} mm, length={bounds.ylen:.4f} mm, '
             f'width envelope={bounds.xlen:.4f} mm')
    print('\n'+message)
    print('Only the measured-width STEP solid was generated (NO STL or PLY).')
    print('Inspect edge_overlays and rowwise_status before using the STEP solid.')
    print('3D outputs:',RECON_DIR)
    (RECON_DIR/'reconstruction_summary.txt').write_text(
        message+'\n'+'Nominal width used only as approximate x calibration from blue guides.\n'
        +'V2 current-vs-previous-layer tracking and width/gradient penalties retained.\n'
        +'The layer-averaged measured edges are identical to global V2 edges.\n'
        +'Smooth spline loft exported as one CAD STEP B-rep (no STL).\n'
        +'No stationary-background registration: camera motion can imitate wall drift.\n',
        encoding='utf8')

if __name__=='__main__':
    main()
