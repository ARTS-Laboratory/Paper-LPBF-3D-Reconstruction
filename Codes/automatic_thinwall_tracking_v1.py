"""Track one LPBF wall through numerically ordered optical_*.png layers.

Reference image: reference_#Thinkess_mm.png, containing TWO narrow blue lines along a slanted wall.
The script fits those lines as x = a*y + b, horizontally straightens every
image into a common coordinate system, then applies the original mean-profile
edge detection with a previous-layer position prior.

Install: pip install numpy opencv-python matplotlib
Run: python automatic_box_tracking_slanted_reference.py

Important: This assumes the reference and layer images have the same pixel
coordinate system, image dimensions, and camera pose. Outputs are in deskewed
image coordinates. Verify the selected wall in layer 1 and the final layer.
"""
from pathlib import Path
import re
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#ROOT = Path(__file__).resolve().parent
ROOT = Path(r"C:\Users\killedar\OneDrive - University of South Carolina\GitHub\Paper-LPBF-3D-Reconstruction\Test_01\processed")
REFERENCE = Path(r"C:\Users\killedar\OneDrive - University of South Carolina\GitHub\Paper-LPBF-3D-Reconstruction\Codes\Thin_wall_drift_tracking\Edge_tracking_output_0.8mm\reference_0.8mm.png")
OUTPUT = Path(r"C:\Users\killedar\OneDrive - University of South Carolina\GitHub\Paper-LPBF-3D-Reconstruction\Codes\Thin_wall_drift_tracking\Edge_tracking_output_0.8mm")
OUTPUT.mkdir(exist_ok=True)
IMAGES= Path(r"C:\Users\killedar\OneDrive - University of South Carolina\GitHub\Paper-LPBF-3D-Reconstruction\Test_01\processed")
#OUTPUT.mkdir(exist_ok=True)
#REFERENCE = ROOT / 'reference_0.4mm.png'
#OUTPUT = ROOT / 'automatic_box_tracking_outputs_0.4mm'
#OUTPUT.mkdir(exist_ok=True)
NOMINAL_WIDTH_MM = 0.8
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


def main():
    layers=discover_layers()
    print(f'Found {len(layers)} layers: {layers[0].name} through {layers[-1].name}')
    if len(layers)!=EXPECTED_LAYER_COUNT:print('WARNING: Expected 40 images; processing all found images.')
    _,ref_bgr=read_image(REFERENCE)
    geom=reference_geometry(ref_bgr)
    print('IMAGES:',geom)
    cv2.imwrite(str(OUTPUT/'reference_deskewed.png'),deskew(ref_bgr,geom))
    mm_per_px=NOMINAL_WIDTH_MM/geom['width_px']
    print(f'Horizontal scale: {mm_per_px:.6f} mm/pixel (0.2 mm nominal width / {geom["width_px"]:.2f} pixels)')
    print('NOTE: 40 mm is the stated full wall length, NOT an independently validated pixel calibration.')
    y0,y1=geom['y0'],geom['y1'];prev=geom['center'];first=None;rows=[];crops=[];titles=[]
    for i,p in enumerate(layers,1):
        gray,bgr=read_image(p)
        if gray.shape[:2]!=ref_bgr.shape[:2]:
            raise ValueError(f'{p.name}: shape {gray.shape[:2]} differs from reference {ref_bgr.shape[:2]}')
        try:
            measurement,profile=profile_edges(gray,geom,prev)
        except RuntimeError as e:
            print(f'Layer {i}, {p.name}: TRACKING FAILED: {e}')
            rows.append({'layer':i,'filename':p.name,'status':'FAILED'})
            continue
        c=measurement['center'];measured_w=measurement['width_px']
        if first is None:first=c
        step=c-prev
        warning='REVIEW large step' if abs(step)>MAX_CENTER_STEP_PX else 'OK'
        fixed_width=geom['width_px']
        # Original image overlay with slanted (not vertical) reference-sized box.
        original=plot_slanted_box(bgr.copy(),c,fixed_width,y0,y1,geom,f'Layer {i} ({p.stem})')
        cv2.imwrite(str(OUTPUT/f'{p.stem}_layer{i:02d}_boxed_full.png'),original)
        # Cropping from deskewed image for easy visual comparison.
        straight=deskew(bgr,geom)
        xa=max(0,int(c-fixed_width/2-35));xb=min(straight.shape[1],int(c+fixed_width/2+36))
        ya=max(0,y0-20);yb=min(straight.shape[0],y1+21)
        straight_box=straight.copy()
        cv2.rectangle(straight_box,(int(c-fixed_width/2),y0),(int(c+fixed_width/2),y1),(255,0,0),2)
        crop=straight_box[ya:yb,xa:xb]
        cv2.imwrite(str(OUTPUT/f'{p.stem}_layer{i:02d}_boxed_crop.png'),crop)
        crops.append(crop);titles.append(f'Layer {i}\n{p.stem}')
        if len(crops)==PANEL_SIZE:
            save_panel(crops,titles,OUTPUT/f'boxed_panel_layers_{i-PANEL_SIZE+1:02d}_to_{i:02d}.png')
            crops=[];titles=[]
        rows.append({'layer':i,'filename':p.name,'status':warning,
                     'center_deskewed_px':c,'shift_from_layer1_px':c-first,
                     'shift_from_layer1_mm':(c-first)*mm_per_px,
                     'step_from_previous_px':step,
                     'detected_width_px':measured_w,'detected_width_mm':measured_w*mm_per_px,
                     'fixed_box_width_px':fixed_width,'fixed_box_width_mm':NOMINAL_WIDTH_MM,
                     'blue_guide_y_span_px':y1-y0,'edge_left_deskewed_px':measurement['left'],
                     'edge_right_deskewed_px':measurement['right'],'profile_score':measurement['score']})
        prev=c
        print(f'Layer {i:02d}: {p.name}: shift={(c-first)*mm_per_px:+.4f} mm; '
              f'detected width={measured_w*mm_per_px:.4f} mm; {warning}')
    if crops:
        i=len(layers);save_panel(crops,titles,OUTPUT/f'boxed_panel_layers_{i-len(crops)+1:02d}_to_{i:02d}.png')
    if rows:
        keys=list(dict.fromkeys(k for row in rows for k in row.keys()))
        with (OUTPUT/'automatic_tracking_measurements.csv').open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    good=[r for r in rows if 'shift_from_layer1_mm' in r]
    if good:
        ix=[r['layer'] for r in good]
        chart(ix,[r['shift_from_layer1_mm'] for r in good],'shift_vs_layer_mm.png','Shift from first successfully tracked layer (mm)',0)
        chart(ix,[r['shift_from_layer1_px'] for r in good],'shift_vs_layer_px.png','Shift from first successfully tracked layer (pixels)',0)
        chart(ix,[r['detected_width_mm'] for r in good],'detected_width_vs_layer_mm.png','Detected wall width (mm)',NOMINAL_WIDTH_MM)
    with (OUTPUT/'automatic_tracking_summary.txt').open('w',encoding='utf8') as f:
        f.write(f'Nominal wall width {NOMINAL_WIDTH_MM} mm; nominal wall length {NOMINAL_LENGTH_MM} mm\n')
        f.write(f'Blue line slope dx/dy: {geom["slope"]:.6f}; gap: {geom["width_px"]:.3f} px\n')
        f.write(f'Width scale: {mm_per_px:.8f} mm/px\n')
        f.write('Length 40mm not independently calibrated. Camera registration must be verified.\n')
        f.write('Tracking uses straightened (deskewed) image coordinates.\n')
        f.write(f'Frames: {len(layers)}; tracked: {len(good)}\n')
        f.write('Filename numbering is sorted; missing captures may make layers nonconsecutive.\n')
    print('Saved results to:',OUTPUT)


if __name__=='__main__':main()
