"""LPBF wall segmentation ONLY: audit CVAT masks, train U-Net, predict masks from BIN or PNG.

Dependencies: pip install numpy pillow opencv-python torch matplotlib
Examples:
  python lpbf_unet_segmentation.py inspect --annotations segmentation_mask_1.1
  python lpbf_unet_segmentation.py train --annotations segmentation_mask_1.1 --source auto --epochs 30 --val-image optical_164473
  python lpbf_unet_segmentation.py predict --source png --exclude-annotated
  python lpbf_unet_segmentation.py predict --source auto --raw raw

CVAT Segmentation Mask 1.1: class colors from labelmap.txt; object colors identify
instances *within each frame*, not necessarily across frames. Training learns a binary
wall-vs-background mask. Only three annotated frames: validation is preliminary and does not establish generalization.
"""
import argparse
import csv
import io
import math
import random
import re
import zipfile
from contextlib import contextmanager
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

WIDTH, HEIGHT = 5472, 3648
MODEL_SIZE = 512


class DirectoryArchive:
    """Read extracted CVAT Segmentation Mask 1.1 folder like a ZIP archive."""
    def __init__(self, root):
        self.root = Path(root)
        if not (self.root / 'labelmap.txt').is_file():
            candidates = list(self.root.rglob('labelmap.txt'))
            if len(candidates) != 1:
                raise FileNotFoundError(f'Expected one labelmap.txt below {root}; found {len(candidates)}')
            self.root = candidates[0].parent
    def namelist(self):
        return [str(p.relative_to(self.root)).replace('\\', '/') for p in self.root.rglob('*') if p.is_file()]
    def read(self, name):
        return (self.root / name).read_bytes()
    def close(self):
        pass

@contextmanager
def open_annotations(path):
    path = Path(path)
    z = DirectoryArchive(path) if path.is_dir() else zipfile.ZipFile(path)
    try:
        yield z
    finally:
        z.close()


def rgb_code(im):
    im = np.asarray(im, dtype=np.uint32)
    return (im[..., 0] << 16) | (im[..., 1] << 8) | im[..., 2]


def unpack_color(code):
    return (int(code >> 16) & 255, int(code >> 8) & 255, int(code) & 255)


def png_data(z, name):
    with Image.open(io.BytesIO(z.read(name))) as im:
        return np.array(im.convert('RGB'))


def get_manifest(z):
    names = set(z.namelist())
    class_names = sorted(n for n in names if n.startswith('SegmentationClass/') and n.endswith('.png'))
    if not class_names:
        raise ValueError('ZIP has no SegmentationClass/*.png; export as CVAT Segmentation Mask 1.1')
    labels = {}
    for line in z.read('labelmap.txt').decode('utf-8-sig').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        name, color, *_ = line.split(':')
        labels[name] = tuple(map(int, color.split(',')))
    wall_codes = {tuple(rgb) for name, rgb in labels.items() if 'wall' in name.lower()}
    if not wall_codes:
        raise ValueError('No wall label in labelmap.txt')
    return class_names, labels, {r << 16 | g << 8 | b for r, g, b in wall_codes}


def wall_mask(z, image_name, codes):
    arr = rgb_code(png_data(z, 'SegmentationClass/' + image_name))
    return np.isin(arr, list(codes)).astype(np.uint8)


def image_stats(z, image_name, codes):
    class_rgb = png_data(z, 'SegmentationClass/' + image_name)
    object_rgb = png_data(z, 'SegmentationObject/' + image_name)
    cc = rgb_code(class_rgb)
    obj = rgb_code(object_rgb)
    u, n = np.unique(obj, return_counts=True)
    instances = [(int(key), int(px)) for key, px in zip(u, n) if key != 0]
    total_wall = int(np.isin(cc, list(codes)).sum())
    return {'image': image_name, 'height': class_rgb.shape[0], 'width': class_rgb.shape[1],
            'instances': len(instances), 'wall_pixels': total_wall,
            'wall_area_pct': 100 * total_wall / cc.size}, instances, obj, class_rgb


def inspect(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rows, objects = [], []
    with open_annotations(args.annotations) as z:
        items, labels, codes = get_manifest(z)
        for name in items:
            fname = Path(name).name
            row, inst, ids, rgb = image_stats(z, fname, codes)
            rows.append(row)
            for idx, (col, area) in enumerate(sorted(inst), 1):
                objects.append({'image': fname, 'object_color_id': col,
                                'rgb': str(unpack_color(col)), 'area_pixels': area})
            # Preview downscaled, retaining original PNG mask pixels in the ZIP.
            mask = np.isin(rgb_code(rgb), list(codes))
            view = np.zeros_like(rgb)
            view[mask] = (80, 225, 100)
            view = cv2.resize(view, (1368, 912), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out / f'{Path(fname).stem}_mask_preview.png'), cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
            print(f'{fname}: {row["instances"]} instances, {row["wall_area_pct"]:.1f}% wall pixels')
    write_csv(out / 'annotation_summary.csv', rows)
    write_csv(out / 'annotation_instances.csv', objects)
    print('Label names:', ', '.join(labels))
    print('Saved annotation summaries:', out)


def write_csv(path, rows):
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)


def read_raw(path):
    # Original LabVIEW files use right-aligned uint16 samples with 0,4,or 8-byte prefix.
    expected = WIDTH * HEIGHT * 2
    sz = path.stat().st_size
    offset = sz - expected
    if offset not in (0, 4, 8):
        raise ValueError(f'{path}: size {sz}; expected {expected} (+ optional 0/4/8 byte header)')
    a = np.memmap(path, mode='r', dtype='<u2', offset=offset, shape=(HEIGHT, WIDTH))
    if int(a.max()) > 4095:
        raise ValueError(f'{path}: max={int(a.max())} >4095. Check 12-bit alignment before training')
    return a


def read_png_12bit(path):
    """Read a 16-bit grayscale PNG from processOptical.py (12-bit shifted left 4).

    OpenCV preserves the 16-bit samples. Dividing by 16 restores the original
    sensor codes; PNG is lossless. 8-bit previews are explicitly rejected.
    """
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None or a.ndim != 2:
        raise ValueError(f"Could not read 16-bit grayscale PNG: {path}")
    if a.shape != (HEIGHT, WIDTH):
        raise ValueError(f"{path}: PNG dimensions {a.shape} do not match {HEIGHT}x{WIDTH}")
    if a.dtype != np.uint16:
        raise ValueError(f"{path}: PNG dtype {a.dtype}, expected uint16. "
                         "An 8-bit preview cannot restore original 12-bit values.")
    # The supplied conversion script uses image << 4; reject other formats.
    if np.any((a & 15) != 0):
        raise ValueError(f"{path}: pixels are not multiples of 16. "
                         "This PNG does not match processOptical.py's <<4 conversion.")
    return a >> 4


def find_source(stem, rawdir, pngdir, preference):
    if preference in ('bin', 'auto'):
        for suffix in ('.bin', '.raw', '.dat'):
            p = rawdir / (stem + suffix)
            if p.exists(): return p
    if preference in ('png', 'auto'):
        # Support the original processOptical.py output folder.
        for parent in (pngdir, pngdir / "optical images", rawdir):
            p = parent / (stem + '.png')
            if p.exists(): return p
    raise FileNotFoundError(f"No {preference} image found for {stem}. "
                            f"Looked under {rawdir} and {pngdir}")


def read_image(path):
    return read_png_12bit(path) if path.suffix.lower() == '.png' else read_raw(path)


def torch_setup():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
    except ImportError as e:
        raise RuntimeError('Install PyTorch first: pip install torch') from e
    return torch, nn, Dataset, DataLoader


def create_model():
    torch, nn, *_ = torch_setup()
    class DoubleConv(nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(True),
                                     nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(True))
        def forward(self, x): return self.net(x)
    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1=DoubleConv(1,24);self.e2=DoubleConv(24,48);self.e3=DoubleConv(48,96)
            self.pool=nn.MaxPool2d(2);self.bot=DoubleConv(96,192)
            self.d3=DoubleConv(192+96,96);self.d2=DoubleConv(96+48,48);self.d1=DoubleConv(48+24,24)
            self.out=nn.Conv2d(24,1,1)
        def forward(self,x):
            a=self.e1(x);b=self.e2(self.pool(a));c=self.e3(self.pool(b));d=self.bot(self.pool(c))
            d=self.d3(torch.cat([nn.functional.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
            d=self.d2(torch.cat([nn.functional.interpolate(d,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
            d=self.d1(torch.cat([nn.functional.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
            return self.out(d)
    return UNet()


def load_examples(annotation_zip, rawdir, pngdir, source):
    samples=[]
    with open_annotations(annotation_zip) as z:
        names, labels, codes = get_manifest(z)
        for n in names:
            fname=Path(n).name
            p=find_source(Path(fname).stem, rawdir, pngdir, source)
            print(f"Training/validation source: {p}")
            a=read_image(p)
            mask=wall_mask(z, fname, codes)
            if mask.shape != a.shape: raise ValueError(f'Shape mismatch: {fname}')
            samples.append((fname, a, mask))
    return samples


def patch_data(samples, patch, count, seed, training=True):
    # Precomputes foreground pixel coordinates for balanced sampling.
    torch, nn, Dataset, DataLoader=torch_setup()
    class Patches(Dataset):
        def __init__(self):
            self.samples=samples
            self.rng_seed=seed
            self.fg=[]
            for _,_,m in samples:
                yy, xx=np.where(m[::4,::4] != 0)
                self.fg.append((yy,xx))
        def __len__(self):return count
        def __getitem__(self,index):
            rng=np.random.default_rng(self.rng_seed + index * 137)
            k=int(rng.integers(len(self.samples)))
            name, raw, mask=self.samples[k]
            h,w=mask.shape
            if rng.random()<0.75 and len(self.fg[k][0]):
                ys,xs=self.fg[k]
                p=int(rng.integers(len(ys)))
                cy=int(ys[p]*4);cx=int(xs[p]*4)
            else:
                cy=int(rng.integers(h));cx=int(rng.integers(w))
            y=int(np.clip(cy - rng.integers(patch),0,h-patch))
            x=int(np.clip(cx - rng.integers(patch),0,w-patch))
            # Avoid boosting each image to a private range; preserve original 12-bit radiometry.
            a=np.array(raw[y:y+patch,x:x+patch],dtype=np.float32)/4095.0
            b=np.array(mask[y:y+patch,x:x+patch],dtype=np.float32)
            if training:
                if rng.random()<0.5: a=np.flip(a,0);b=np.flip(b,0)
                if rng.random()<0.5: a=np.flip(a,1);b=np.flip(b,1)
                if rng.random()<0.5: a=np.rot90(a,2);b=np.rot90(b,2)
                # mild illumination gain simulates variation but does not synthesize bit depth
                if rng.random()<0.3: a=np.clip(a*float(rng.uniform(.9,1.1)),0,1)
            return torch.from_numpy(a.copy()[None]),torch.from_numpy(b.copy()[None])
    return Patches()


def soft_dice_loss(logits, target):
    torch, *_=torch_setup()
    prob=torch.sigmoid(logits)
    return 1-(2*(prob*target).sum((1,2,3))+1)/((prob+target).sum((1,2,3))+1)


def train(args):
    torch, nn, Dataset, DataLoader = torch_setup()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    samples=load_examples(args.annotations,args.raw,args.png_dir,args.source)
    if len(samples)<3: raise ValueError('At least 3 annotated images recommended for this holdout example')
    val_name=args.val_image if args.val_image.endswith('.png') else args.val_image+'.png'
    val=[s for s in samples if s[0]==val_name]
    train_samples=[s for s in samples if s[0]!=val_name]
    if len(val)!=1:raise ValueError(f'Validation image {val_name} not in annotations: {[s[0] for s in samples]}')
    print('Training frames:',[s[0] for s in train_samples], 'Validation frame:',val_name)
    tr=DataLoader(patch_data(train_samples,args.patch,args.steps_per_epoch*args.batch,args.seed),batch_size=args.batch,shuffle=True,num_workers=0)
    va=DataLoader(patch_data(val,args.patch,args.val_patches,args.seed+999,False),batch_size=args.batch,num_workers=0)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=create_model().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    bce=nn.BCEWithLogitsLoss()
    args.output.mkdir(parents=True,exist_ok=True)
    best=-1
    history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); tl=[]
        for x,y in tr:
            x=x.to(device);y=y.to(device)
            opt.zero_grad(set_to_none=True)
            pred=model(x)
            loss=bce(pred,y)+soft_dice_loss(pred,y).mean()
            loss.backward();opt.step();tl.append(float(loss.item()))
        model.eval(); scores=[]
        with torch.no_grad():
            for x,y in va:
                x=x.to(device);y=y.to(device)
                pred=(torch.sigmoid(model(x))>=0.5).float()
                scores.extend(((2*(pred*y).sum((1,2,3))+1)/((pred+y).sum((1,2,3))+1)).cpu().tolist())
        score=float(np.mean(scores))
        history.append({'epoch':epoch,'train_loss':float(np.mean(tl)),'val_patch_dice':score})
        print(f'Epoch {epoch:03d} loss={np.mean(tl):.4f} val_patch_dice={score:.4f}')
        if score>best:
            best=score
            torch.save({'model':model.state_dict(),'patch':args.patch,'threshold':0.5,
                        'val_image':val_name,'format':'raw right-aligned 12bit / 4095'},args.output/'best_model.pt')
    write_csv(args.output/'training_history.csv',history)
    print('Model saved to:',args.output/'best_model.pt')


def load_checkpoint(path,device):
    torch, *_=torch_setup()
    # Locally-generated checkpoint only. Do not load untrusted model weights.
    check=torch.load(path,map_location='cpu',weights_only=True)
    model=create_model()
    model.load_state_dict(check['model']);model.to(device).eval()
    return model,check


def image_patches(img, model, device, patch=512, overlap=96, batch=4):
    torch,*_=torch_setup()
    h,w=img.shape
    stride=patch-overlap
    def positions(n):
        arr=list(range(0,max(n-patch,0)+1,stride))
        if not arr or arr[-1]!=n-patch:arr.append(n-patch)
        return arr
    ys=positions(h); xs=positions(w)
    # inference accumulates probabilities; normalize only by fixed 12-bit scale
    total=np.zeros((h,w),np.float32); weights=np.zeros((h,w),np.float32)
    tile_pos=[(y,x) for y in ys for x in xs]
    weight=np.ones((patch,patch),np.float32)
    for start in range(0,len(tile_pos),batch):
        group=tile_pos[start:start+batch]
        tiles=np.stack([img[y:y+patch,x:x+patch].astype(np.float32)/4095.0 for y,x in group])[:,None]
        with torch.no_grad():
            inp=torch.from_numpy(tiles).to(device)
            probs=torch.sigmoid(model(inp)).cpu().numpy()[:,0]
        for (y,x),p in zip(group,probs):
            total[y:y+patch,x:x+patch]+=p*weight
            weights[y:y+patch,x:x+patch]+=weight
    return total/np.maximum(weights,1e-6)


def process_mask(pred, source_name, outdir, threshold, min_area):
    binary=(pred>=threshold).astype(np.uint8)
    outdir.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(outdir/f'{source_name}_mask.png'),binary*255)
    n, cc, stats, cent=cv2.connectedComponentsWithStats(binary,connectivity=8)
    rows=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area<min_area:continue
        rows.append({'filename':source_name,'component_id':i,'area_px':int(area),
                     'x':int(x),'y':int(y),'width_px':int(w),'height_px':int(h),
                     'center_x':float(cent[i,0]),'center_y':float(cent[i,1])})
    return rows


def predict(args):
    torch,*_=torch_setup()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,ckpt=load_checkpoint(args.model,device)
    # Build a list of unique basenames; prefer BIN in auto mode.
    allstems = set()
    if args.source in ('bin', 'auto'):
        for ext in ('*.bin', '*.raw', '*.dat'):
            allstems.update(p.stem for p in args.raw.glob(ext))
    if args.source in ('png', 'auto'):
        allstems.update(p.stem for p in args.png_dir.glob('optical_*.png'))
        allstems.update(p.stem for p in (args.png_dir / 'optical images').glob('optical_*.png'))
        allstems.update(p.stem for p in args.raw.glob('optical_*.png'))
    stems = sorted(allstems)
    if args.exclude_annotated:
        with open_annotations(args.annotations) as z:
            annotated, _, _ = get_manifest(z)
        annotated_stems = {Path(name).stem for name in annotated}
        stems = [stem for stem in stems if stem not in annotated_stems]
        print(f'Excluding {len(annotated_stems)} annotated image names from prediction.')
    if args.limit: stems=stems[:args.limit]
    if not stems:
        raise ValueError(f'No input images found in {args.raw} or {args.png_dir}')
    allrows=[]
    for ix,stem in enumerate(stems,1):
        p=find_source(stem,args.raw,args.png_dir,args.source)
        a=read_image(p)
        prob=image_patches(a,model,device,patch=args.patch,overlap=args.overlap,batch=args.batch)
        rows=process_mask(prob,p.stem,args.output,args.threshold,args.min_area)
        allrows.extend(rows)
        print(f'[{ix}/{len(stems)}] {p.name}: {len(rows)} provisional connected components')
    write_csv(args.output/'predicted_components.csv',allrows)
    print('Connected component IDs are provisional and NOT physical wall IDs.')


def build_parser():
    root=Path(__file__).resolve().parent
    default_annotations = root / 'segmentation_mask_1.1'
    if not default_annotations.exists():
        default_annotations = root / 'segmentation_mask_1.1.zip'
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('inspect',help='Audit CVAT masks; works without original binary images')
    a.add_argument('--annotations',type=Path,default=default_annotations)
    a.add_argument('--output',type=Path,default=root/'processed'/'annotation_audit')
    a.set_defaults(func=inspect)
    a=sub.add_parser('train',help='Train binary U-Net from matched BIN/PNG and CVAT masks')
    a.add_argument('--annotations',type=Path,default=default_annotations)
    a.add_argument('--raw',type=Path,default=root/'raw')
    a.add_argument('--png-dir',type=Path,default=root/'processed')
    a.add_argument('--source',choices=['auto','bin','png'],default='auto',help='auto prefers BIN; PNG must be lossless 16-bit from processOptical.py')
    a.add_argument('--output',type=Path,default=root/'processed'/'unet')
    a.add_argument('--val-image',default='optical_164473')
    a.add_argument('--patch',type=int,default=512)
    a.add_argument('--epochs',type=int,default=30)
    a.add_argument('--batch',type=int,default=4)
    a.add_argument('--steps-per-epoch',type=int,default=100)
    a.add_argument('--val-patches',type=int,default=40)
    a.add_argument('--lr',type=float,default=1e-3)
    a.add_argument('--seed',type=int,default=42)
    a.set_defaults(func=train)
    for key,func in [('predict',predict)]:
        a=sub.add_parser(key)
        a.add_argument('--raw',type=Path,default=root/'raw')
        a.add_argument('--png-dir',type=Path,default=root/'processed')
        a.add_argument('--source',choices=['auto','bin','png'],default='auto')
        a.add_argument('--model',type=Path,default=root/'processed'/'unet'/'best_model.pt')
        a.add_argument('--output',type=Path,default=root/'processed'/'predicted_masks')
        a.add_argument('--limit',type=int,default=0)
        a.add_argument('--exclude-annotated',action='store_true',help='Predict only images absent from CVAT ZIP (e.g. the other 37)')
        a.add_argument('--annotations',type=Path,default=default_annotations)
        a.add_argument('--patch',type=int,default=512)
        a.add_argument('--overlap',type=int,default=96)
        a.add_argument('--batch',type=int,default=4)
        a.add_argument('--threshold',type=float,default=0.5)
        a.add_argument('--min-area',type=int,default=500)
        a.set_defaults(func=func)
    return parser

if __name__=='__main__':
    args=build_parser().parse_args()
    if hasattr(args,'overlap') and not 0 <= args.overlap < args.patch:raise ValueError('overlap must be >=0 and < patch')
    if hasattr(args,'patch') and args.patch %8:raise ValueError('patch size must be a multiple of 8')
    args.func(args)
