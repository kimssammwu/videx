"""Reproducible, non-destructive experiments on the existing session_02 JPEGs.

Run from the repository root with the stereo venv. All coordinates are in the
240x320 rotated image space; PNG enlargement is solely for visual inspection.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from scipy.optimize import least_squares

import cv2
import numpy as np

from .calibration import BoardSettings, QualitySettings, Observation, fit_calibration
from .dataset import load_pair
from .diagnose import grid_metrics, inspect_detection, numbered_corners
from .orientation import OrientationSettings, prepare_pair

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'app/stereo/data/session_02'
REFERENCE = ROOT / 'app/stereo/results/calibration_02.json'
BOARD = BoardSettings(9, 6, 22.0, 'mm')
ORIENTATION = OrientationSettings('cw90', 'ccw90')
SIZE = (240, 320)
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 150, 1e-9)
cv2.setNumThreads(2)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(DATA.rglob('*')) if p.is_file()} | {
                str(REFERENCE): hashlib.sha256(REFERENCE.read_bytes()).hexdigest()}


def load_images():
    result = []
    for n, directory in enumerate(sorted(DATA.glob('pair_*')), 1):
        meta, paths, raw = load_pair(directory / 'pair.json')
        images = prepare_pair(*raw, ORIENTATION)
        result.append((n, meta, images))
    return result


def png(path, img):
    if not cv2.imwrite(str(path), img):
        raise OSError(path)


def audit(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'corners').mkdir()
    write_json(output / 'source_hashes_before.json', hashes())
    arrays, records, contacts, originals = {}, [], [], []
    for n, meta, images in load_images():
        record = {'n': n, 'pair_id': meta['pair_id'], 'metadata': meta}
        panels, raw_panels = [], []
        for side, img in zip(('left', 'right'), images):
            original, info, alternatives = inspect_detection(img, BOARD)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            candidates = {'original': original}
            candidates.update(alternatives)
            found, corners = cv2.findChessboardCorners(gray, (9, 6), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if found:
                candidates['classic_initial'] = corners.copy()
                for w in (2, 3, 5, 11):
                    candidates[f'classic_window_{w}'] = cv2.cornerSubPix(gray, corners.copy(), (w,w), (-1,-1), (3,40,1e-4))
            found, sb = cv2.findChessboardCornersSB(gray, (9,6), cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            if found:
                candidates['sb_accurate'] = sb
            big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            found, sb = cv2.findChessboardCornersSB(big, (9,6), cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            if found:
                candidates['sb_up2'] = (sb + .5) / 2 - .5
            detail = {'original_info': info, 'candidates': {}, 'laplacian_variance_full': float(cv2.Laplacian(gray,cv2.CV_64F).var())}
            for name, points in candidates.items():
                detail['candidates'][name] = grid_metrics(points, BOARD)
                if points is not None:
                    arrays[f'{n:02}_{side}_{name}'] = points.astype(np.float32)
            record[side] = detail
            panel = numbered_corners(img, original, BOARD, f'{n:02} {side.upper()} ORIGINAL {info["detector"]}', 3)
            png(output/'corners'/f'{n:02}_{side}_original.png', panel)
            panels.append(panel)
            rawpanel = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            rawpanel = cv2.copyMakeBorder(rawpanel,32,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(rawpanel,f'{n:02} {side.upper()} {meta["pair_id"][9:15]}', (8,22),0,.55,(255,255,255),1)
            raw_panels.append(rawpanel)
        records.append(record)
        contacts.append(np.hstack(panels))
        originals.append(np.hstack(raw_panels))
        print('AUDIT', n, [(s, record[s]['original_info']['detector'], {k: round(v['homography_rms_px'],3) for k,v in record[s]['candidates'].items() if v is not None}) for s in ('left','right')],flush=True)
    for start in range(0,24,6):
        # Two columns, three rows: all 48 original JPEGs and original corner grids.
        for name, items in [('original_images',originals), ('original_numbered',contacts)]:
            sheet = np.vstack([np.hstack(items[i:i+2]) for i in range(start,start+6,2)])
            png(output/f'{name}_{start+1:02}_{start+6:02}.png', sheet)
    np.savez_compressed(output/'detections.npz', **arrays)
    write_json(output/'audit.json',records)
    write_json(output/'source_preservation.json',{'preserved':hashes()==json.loads((output/'source_hashes_before.json').read_text()),'after':hashes()})


def canonical(points):
    """These inspected poses have the board's long-axis direction toward +x.

    Only the observed 180-degree ambiguity is resolved, with both grid axes
    reversed together. This is not a general solution for arbitrary board poses.
    """
    points = points.copy()
    if np.mean(np.diff(points.reshape(6,9,2), axis=1)[:,:,0]) < 0:
        return points[::-1].copy(), True
    return points, False


def observations(output, mode='accurate', reorder=True, numbers=None):
    arrays = np.load(output/'detections.npz')
    audit = json.loads((output/'audit.json').read_text(encoding='utf-8'))
    obs, provenance = [], []
    for r in audit:
        n = r['n']
        if numbers is not None and n not in numbers:
            continue
        points, info = [], {'n':n,'pair_id':r['pair_id']}
        for side in ('left','right'):
            names = {'original':['original'], 'accurate':['sb_accurate','original','classic_window_3'],
                     'up2':['sb_up2','sb_accurate','original','classic_window_3'],
                     'small_window':['classic_window_3','sb_accurate','original']}[mode]
            selected = next((name for name in names if f'{n:02}_{side}_{name}' in arrays), None)
            if selected is None:
                break
            p = arrays[f'{n:02}_{side}_{selected}']
            flipped = False
            if reorder:
                p, flipped = canonical(p)
            points.append(p)
            info[side] = {'detector':selected,'reversed180':flipped,'grid':grid_metrics(p,BOARD)}
        if len(points)==2:
            obs.append(Observation(r['pair_id'],*points,SIZE))
            provenance.append(info)
    return obs, provenance


MODELS = {
    'full5':0,
    'k1k2t':cv2.CALIB_FIX_K3,
    'k1k2':cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
    'k1':cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
    'zero':cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
    'square_k1k2':cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
}


def rms(array):
    return float(np.sqrt(np.mean(np.asarray(array)**2)))


def rectification(kl,dl,kr,dr,R,T,alpha=0):
    return cv2.stereoRectify(kl,dl,kr,dr,SIZE,R,T,flags=cv2.CALIB_ZERO_DISPARITY,alpha=alpha)


def evaluate(obs, kl,dl,kr,dr,R,T,rect):
    r1,r2,p1,p2,q,roi1,roi2 = rect
    obj = BOARD.object_points()
    per_pair, vertical, native_vertical, mono, joint, transfer = [], [], [], [[],[]], [], []
    for o in obs:
        poses, mono_rms = [], []
        for s,(points,k,d) in enumerate(((o.left,kl,dl),(o.right,kr,dr))):
            ok,rv,tv = cv2.solvePnP(obj,points,k,d,flags=cv2.SOLVEPNP_ITERATIVE)
            projected = cv2.projectPoints(obj,rv,tv,k,d)[0]
            diff = points.reshape(-1,2)-projected.reshape(-1,2)
            mono[s].extend(np.sum(diff**2,axis=1).tolist())
            mono_rms.append(float(np.sqrt(np.mean(np.sum(diff**2,axis=1)))))
            poses.append((rv,tv))
        lr = cv2.undistortPoints(o.left,kl,dl,R=r1,P=p1).reshape(-1,2)
        rr = cv2.undistortPoints(o.right,kr,dr,R=r2,P=p2).reshape(-1,2)
        delta = lr[:,1]-rr[:,1]
        # A fixed virtual focal length prevents zoom/crop from gaming vertical RMS.
        lnative = cv2.undistortPoints(o.left,kl,dl,R=r1).reshape(-1,2)
        rnative = cv2.undistortPoints(o.right,kr,dr,R=r2).reshape(-1,2)
        dy240 = 240*(lnative[:,1]-rnative[:,1])
        vertical.extend(delta.tolist())
        native_vertical.extend(dy240.tolist())
        def residual(x):
            rb=cv2.Rodrigues(x[:3])[0]
            tb=x[3:].reshape(3,1)
            rvright=cv2.Rodrigues(R@rb)[0]
            tvright=R@tb+T.reshape(3,1)
            lpred=cv2.projectPoints(obj,x[:3],tb,kl,dl)[0].reshape(-1,2)
            rpred=cv2.projectPoints(obj,rvright,tvright,kr,dr)[0].reshape(-1,2)
            return np.concatenate([(lpred-o.left.reshape(-1,2)).ravel(),(rpred-o.right.reshape(-1,2)).ravel()])
        x0=np.r_[poses[0][0].ravel(),poses[0][1].ravel()]
        initial=residual(x0)
        # Each test board pose is fitted; global camera parameters remain frozen.
        opt=least_squares(residual,x0,method='lm',max_nfev=120,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        jerr=opt.fun.reshape(-1,2)
        joint.extend(np.sum(jerr**2,axis=1).tolist())
        trans=initial[108:].reshape(-1,2)
        transfer.extend(np.sum(trans**2,axis=1).tolist())
        inside=lambda a: np.mean((a[:,0]>=0)&(a[:,0]<SIZE[0])&(a[:,1]>=0)&(a[:,1]<SIZE[1]))
        # Triangulated board dimensions are an independent geometric shape check,
        # though the same known square length sets calibration's metric scale.
        lu=cv2.undistortPoints(o.left,kl,dl).reshape(-1,2)
        ru=cv2.undistortPoints(o.right,kr,dr).reshape(-1,2)
        homogeneous=cv2.triangulatePoints(np.c_[np.eye(3),np.zeros(3)],np.c_[R,T.reshape(3)],lu.T,ru.T)
        xyz=(homogeneous[:3]/homogeneous[3]).T
        grid=xyz.reshape(6,9,3)
        edges=np.r_[np.linalg.norm(np.diff(grid,axis=1),axis=2).ravel(),np.linalg.norm(np.diff(grid,axis=0),axis=2).ravel()]
        per_pair.append({'pair_id':o.pair_id,'left_rms_px':mono_rms[0],'right_rms_px':mono_rms[1],
                         'stereo_joint_pose_rms_px':float(np.sqrt(np.mean(np.sum(jerr**2,axis=1)))),
                         'right_transfer_from_left_pose_rms_px':float(np.sqrt(np.mean(np.sum(trans**2,axis=1)))),
                         'vertical_rms_px':rms(delta),'vertical_rms_at_f240_px':rms(dy240),
                         'vertical_max_px':float(np.max(np.abs(delta))),
                         'rectified_corner_fraction_left':float(inside(lr)),
                         'rectified_corner_fraction_right':float(inside(rr)),
                         'median_disparity_px':float(np.median(lr[:,0]-rr[:,0])),
                         'triangulated_depth_median_mm':float(np.median(xyz[:,2])),
                         'positive_depth_fraction':float(np.mean((xyz[:,2]>0)&((R@xyz.T+T.reshape(3,1))[2]>0))),
                         'triangulated_edge_median_mm':float(np.median(edges)),
                         'triangulated_edge_rms_error_mm':rms(edges-22)})
    return {'pairs':len(obs),'left_rms_px':float(np.sqrt(np.mean(mono[0]))),
            'right_rms_px':float(np.sqrt(np.mean(mono[1]))),'stereo_joint_pose_rms_px':float(np.sqrt(np.mean(joint))),
            'right_transfer_rms_px':float(np.sqrt(np.mean(transfer))),
            'vertical_rms_px':rms(vertical),'vertical_rms_at_f240_px':rms(native_vertical),
            'vertical_mean_abs_px':float(np.mean(np.abs(vertical))),
            'vertical_p95_px':float(np.percentile(np.abs(vertical),95)),
            'vertical_max_px':float(np.max(np.abs(vertical))), 'per_pair':per_pair}


def coverage(kl,dl,kr,dr,R,T):
    results=[]
    for alpha in (0,.5,1):
        rect=rectification(kl,dl,kr,dr,R,T,alpha)
        masks=[]
        source_retained=[]
        for k,d,r,p in ((kl,dl,rect[0],rect[2]),(kr,dr,rect[1],rect[3])):
            mx,my=cv2.initUndistortRectifyMap(k,d,r,p,SIZE,cv2.CV_32FC1)
            masks.append((mx>=0)&(mx<SIZE[0]-1)&(my>=0)&(my<SIZE[1]-1))
            ys,xs=np.mgrid[:SIZE[1],:SIZE[0]]
            rawpoints=np.stack([xs,ys],axis=-1).astype(np.float32).reshape(-1,1,2)
            mapped=cv2.undistortPoints(rawpoints,k,d,R=r,P=p).reshape(-1,2)
            source_retained.append(float(np.mean((mapped[:,0]>=0)&(mapped[:,0]<SIZE[0])&(mapped[:,1]>=0)&(mapped[:,1]<SIZE[1]))))
        results.append({'alpha':alpha,'roi_left':list(rect[5]),'roi_right':list(rect[6]),
                        'rectified_focal_px':float(rect[2][0,0]),
                        'valid_left_fraction':float(np.mean(masks[0])),
                        'valid_right_fraction':float(np.mean(masks[1])),
                        'valid_intersection_fraction':float(np.mean(masks[0]&masks[1])),
                        'source_pixels_retained_left_fraction':source_retained[0],
                        'source_pixels_retained_right_fraction':source_retained[1]})
    return results


def fit(obs,model='full5',joint=False, test=None):
    objects=[BOARD.object_points() for _ in obs]
    left=[o.left for o in obs]; right=[o.right for o in obs]
    flags=MODELS[model]
    K0=np.array([[240.,0,119.5],[0,240.,159.5],[0,0,1]])
    rms_l,kl,dl,*_=cv2.calibrateCamera(objects,left,SIZE,K0.copy(),None,flags=flags,criteria=CRITERIA)
    rms_r,kr,dr,*_=cv2.calibrateCamera(objects,right,SIZE,K0.copy(),None,flags=flags,criteria=CRITERIA)
    stereo_flags=(cv2.CALIB_USE_INTRINSIC_GUESS|flags) if joint else cv2.CALIB_FIX_INTRINSIC
    stereo_rms,kl,dl,kr,dr,R,T,E,F=cv2.stereoCalibrate(objects,left,right,kl,dl,kr,dr,SIZE,flags=stereo_flags,criteria=CRITERIA)
    rect=rectification(kl,dl,kr,dr,R,T)
    train=evaluate(obs,kl,dl,kr,dr,R,T,rect)
    test_metrics=evaluate(test,kl,dl,kr,dr,R,T,rect) if test else None
    baseline=float(np.linalg.norm(T))
    warnings=[]
    for label,value,limit in [('LEFT RMS',train['left_rms_px'],1),('RIGHT RMS',train['right_rms_px'],1),
                              ('Stereo RMS',stereo_rms,1.5),('Vertical RMS',train['vertical_rms_px'],1)]:
        if value>limit: warnings.append(f'{label} exceeds {limit}px')
    if abs(baseline-95)/95 > .2: warnings.append('Baseline differs from 95mm by more than 20%')
    if len(obs)<10: warnings.append('Fewer than 10 training pairs')
    matrices=dict(K_left=kl,D_left=dl,K_right=kr,D_right=dr,R=R,T=T,E=E,F=F,
                  R1=rect[0],R2=rect[1],P1=rect[2],P2=rect[3],Q=rect[4])
    return {'schema_version':2,'opencv_version':cv2.__version__,'orientation':ORIENTATION.to_dict(),
            'input_space':'rotated_raw_pixels','image_size':list(SIZE),'board':{'columns':9,'rows':6,'square_size':22.,'unit':'mm'},
            'length_unit':'mm','quality_settings':{'min_pairs':10,'max_reprojection_px':1.,'max_stereo_rms_px':1.5,
            'max_vertical_rms_px':1.,'expected_baseline_cm':9.5,'baseline_relative_tolerance':.2},
            'matrices':{k:v.tolist() for k,v in matrices.items()},'roi_left':list(rect[5]),'roi_right':list(rect[6]),
            'experiment_settings':{'model':model,'mono_flags':int(flags),'stereo_flags':int(stereo_flags),
                                   'joint_intrinsic_optimization':joint,'baseline_constraint':None,'rectification_alpha':0,
                                   'termination_criteria':list(CRITERIA)},
            'train_pair_ids':[o.pair_id for o in obs], 'test_pair_ids':[o.pair_id for o in test] if test else [],
            'report':{'used_pairs':len(obs),'left_reprojection_rms_px':train['left_rms_px'],'right_reprojection_rms_px':train['right_rms_px'],
                      'mono_fit_left_rms_px':rms_l,'mono_fit_right_rms_px':rms_r,'stereo_rms_px':stereo_rms,
                      'vertical_rms_px':train['vertical_rms_px'],'baseline':baseline,'baseline_unit':'mm','baseline_cm':baseline/10,
                      'baseline_relative_difference':abs(baseline-95)/95,
                      'relative_rotation_degrees':float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))),
                      'warnings':warnings,'numeric_checks_passed':not warnings,
                      'validation_status':'physical_distance_accuracy_unverified',
                      'synchronization_verified':False,'physical_camera_mapping_verified':False},
            'training_evaluation':train,'held_out_evaluation':test_metrics,'rectification_coverage':coverage(kl,dl,kr,dr,R,T)}


def experiment(output,name,obs,model='full5',joint=False,test=None):
    started=time.monotonic()
    result=fit(obs,model,joint,test)
    result['experiment_name']=name
    write_json(output/'experiments'/f'{name}.json',result)
    r=result['report']
    print('FIT',name,'n',len(obs),'mono',round(r['left_reprojection_rms_px'],4),round(r['right_reprojection_rms_px'],4),
          'stereo',round(r['stereo_rms_px'],4),'vertical',round(r['vertical_rms_px'],4),'B',round(r['baseline'],3),
          'test',None if not test else round(result['held_out_evaluation']['vertical_rms_px'],4),
          'seconds',round(time.monotonic()-started,1),flush=True)
    return result


def suite(output):
    (output/'experiments').mkdir(exist_ok=True)
    original,prov=observations(output,'original',False)
    clean_numbers=[p['n'] for p in prov if max(p[s]['grid']['homography_rms_px'] for s in ('left','right')) < .5]
    experiment(output,'01_original_reproduced',original)
    reversed_original,_=observations(output,'original',True)
    experiment(output,'02_original_order_corrected',reversed_original)
    original_clean,_=observations(output,'original',False,clean_numbers)
    experiment(output,'03_original_clean14',original_clean)
    corrected_original_clean,_=observations(output,'original',True,clean_numbers)
    experiment(output,'04_original_corrected_clean14',corrected_original_clean)
    accurate,provenance=observations(output,'accurate',True)
    write_json(output/'corrected_provenance.json',provenance)
    contacts=[]
    for n,meta,images in load_images():
        ob=next((o for o in accurate if o.pair_id==meta['pair_id']),None)
        panels=[numbered_corners(img,None if ob is None else getattr(ob,s),BOARD,f'{n:02} {s.upper()} CORRECTED',3)
                for s,img in zip(('left','right'),images)]
        png(output/'corners'/f'{n:02}_corrected_pair.png',np.hstack(panels))
        contacts.append(np.hstack(panels))
    for start in range(0,24,6):
        png(output/f'corrected_numbered_{start+1:02}_{start+6:02}.png',
            np.vstack([np.hstack(contacts[i:i+2]) for i in range(start,start+6,2)]))
    accurate_no_order,_=observations(output,'accurate',False)
    experiment(output,'05_accurate_no_order_correction',accurate_no_order)
    experiment(output,'06_accurate_corrected20',accurate)
    for mode in ('up2','small_window'):
        obs,_=observations(output,mode,True)
        experiment(output,f'07_{mode}_corrected',obs,'k1k2t',True)
    for model in MODELS:
        for joint in (False,True):
            experiment(output,f'08_accurate_{model}_{"joint" if joint else "fixed"}',accurate,model,joint)
    write_json(output/'source_preservation.json',{'preserved':hashes()==json.loads((output/'source_hashes_before.json').read_text()),'after':hashes()})


def recover(output):
    """Contrast/scale/crop recovery attempts, without synthesizing any corners."""
    recovered, log = {}, []
    for n,meta,images in load_images():
        if n not in (1,6,7,24):
            continue
        for side,img in zip(('left','right'),images):
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            blur=cv2.GaussianBlur(gray,(0,0),.7)
            variants={'gray':gray,'clahe':cv2.createCLAHE(2.,(4,4)).apply(gray),
                      'blur07':blur,'unsharp':cv2.addWeighted(gray,1.7,blur,-.7,0),
                      'gamma05':np.uint8(np.sqrt(gray/255)*255),
                      'gamma15':np.uint8((gray/255)**1.5*255)}
            # Visible board ROIs are deliberately broad. The cropped search changes
            # only detector context, not source JPEGs or calibrated pixel locations.
            # Crop boundaries cannot recover a physically missing last column.
            y0,y1=(45,155) if n==1 else ((175,295) if n==6 else ((90,220) if n==7 else (85,185)))
            variants['crop']=gray[y0:y1,:].copy()
            for name,variant in variants.items():
                for scale in (1,2,3):
                    large=cv2.resize(variant,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
                    found,p=cv2.findChessboardCornersSB(large,(9,6),cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY)
                    metrics=None
                    if found:
                        p=((p+.5)/scale-.5).reshape(-1,1,2)
                        if name=='crop': p[:,:,1]+=y0
                        p,_=canonical(p)
                        metrics=grid_metrics(p,BOARD)
                        recovered[f'{n:02}_{side}_{name}_s{scale}']=p
                        png(output/'corners'/f'{n:02}_{side}_recovery_{name}_s{scale}.png',
                            numbered_corners(img,p,BOARD,f'{n:02} {side} recovery {name} x{scale}',3))
                    log.append({'n':n,'side':side,'variant':name,'scale':scale,'found':bool(found),'grid':metrics})
            print('RECOVERY',n,side,[(x['variant'],x['scale'],round(x['grid']['homography_rms_px'],3))
                  for x in log if x['n']==n and x['side']==side and x['found']],flush=True)
    np.savez_compressed(output/'recovery_detections.npz',**recovered)
    write_json(output/'recovery_attempts.json',log)


def distortion_diagnostics(k,d):
    d=np.asarray(d).ravel()
    # Sensor-wide radius in the current calibration's normalized image space.
    radius=max(np.linalg.norm([(u-k[0,2])/k[0,0],(v-k[1,2])/k[1,1]])
               for u,v in ((0,0),(239,0),(0,319),(239,319)))
    radii=np.linspace(0,radius,300)
    k1,k2,p1,p2,k3=d[:5]
    scale=1+k1*radii**2+k2*radii**4+k3*radii**6
    derivative=1+3*k1*radii**2+5*k2*radii**4+7*k3*radii**6
    return {'sensor_radius_max':float(radius),'radial_scale_min':float(np.min(scale)),
            'radial_scale_max':float(np.max(scale)),'radial_derivative_min':float(np.min(derivative)),
            'radial_monotonic_on_sensor':bool(np.all(derivative>0))}


def cross_validation(output,clean=False):
    obs,prov=observations(output,'accurate',True)
    if clean:
        obs=[o for o,p in zip(obs,prov) if p['n'] not in (11,22)]
    experiments=output/'experiments'
    records=[]
    schemes={'shuffled_seed95':np.random.default_rng(95).permutation(len(obs)),
             'chronological_blocks':np.arange(len(obs))}
    for scheme,indices in schemes.items():
        for model in (('k1k2','k1') if clean else ('full5','k1k2t','k1k2','k1','zero')):
            for joint in ((False,) if clean else (False,True)):
                fits=[]
                for fold, test_indices in enumerate(np.array_split(indices,5),1):
                    train=[o for i,o in enumerate(obs) if i not in test_indices]
                    test=[o for i,o in enumerate(obs) if i in test_indices]
                    result=experiment(output,f'cv_{"clean18_" if clean else ""}{scheme}_{model}_{"joint" if joint else "fixed"}_{fold}',train,model,joint,test)
                    fits.append(result)
                held=[r['held_out_evaluation'] for r in fits]
                baseline=[r['report']['baseline'] for r in fits]
                matrices=[{k:np.array(v) for k,v in r['matrices'].items()} for r in fits]
                weighted=lambda key:float(np.sqrt(sum(r['pairs']*r[key]**2 for r in held)/sum(r['pairs'] for r in held)))
                records.append({'scheme':scheme,'model':model,'joint':joint,
                                'heldout_vertical_rms_px':weighted('vertical_rms_px'),
                                'heldout_vertical_at_f240_rms_px':weighted('vertical_rms_at_f240_px'),
                                'heldout_joint_stereo_rms_px':weighted('stereo_joint_pose_rms_px'),
                                'heldout_left_rms_px':weighted('left_rms_px'),
                                'heldout_right_rms_px':weighted('right_rms_px'),
                                'baseline_min_mm':min(baseline),'baseline_max_mm':max(baseline),'baseline_std_mm':float(np.std(baseline)),
                                'fold_radial_checks':[{s:distortion_diagnostics(m['K_'+s],m['D_'+s]) for s in ('left','right')} for m in matrices],
                                'fold_experiments':[r['experiment_name'] for r in fits]})
    write_json(output/('cross_validation_clean18_summary.json' if clean else 'cross_validation_summary.json'),records)
    for r in records:
        print('CVSUMMARY',r['scheme'],r['model'],r['joint'],round(r['heldout_joint_stereo_rms_px'],4),
              round(r['heldout_vertical_rms_px'],4),round(r['baseline_min_mm'],2),round(r['baseline_max_mm'],2),flush=True)


def diagnostics(output):
    obs,prov=observations(output,'accurate',True)
    master=json.loads((output/'experiments/08_accurate_k1k2_joint.json').read_text())
    m={k:np.asarray(v) for k,v in master['matrices'].items()}
    records=[]
    for o,p in zip(obs,prov):
        poses=[]
        for side in ('left','right'):
            _,rv,tv=cv2.solvePnP(BOARD.object_points(),getattr(o,side),m['K_'+side],m['D_'+side])
            poses.append((cv2.Rodrigues(rv)[0],tv))
        Ri=poses[1][0]@poses[0][0].T
        Ti=poses[1][1]-Ri@poses[0][1]
        records.append({'n':p['n'],'pair_id':o.pair_id,'individual_R':Ri.tolist(),'individual_T_mm':Ti.tolist(),
                        'individual_baseline_mm':float(np.linalg.norm(Ti)),
                        'rotation_difference_from_rig_degrees':float(np.degrees(np.linalg.norm(cv2.Rodrigues(Ri@m['R'].T)[0]))),
                        'translation_difference_from_rig_mm':float(np.linalg.norm(Ti-m['T'])),
                        'board_normal_left':poses[0][0][:,2].tolist(),'board_center_depth_mm':float(poses[0][1][2,0])})
    write_json(output/'pose_consistency.json',records)
    source_rows=[]
    for p in sorted((output/'experiments').glob('0*.json')):
        d=json.loads(p.read_text()); mats={k:np.asarray(v) for k,v in d['matrices'].items()}
        source_rows.append({'experiment':p.stem,'distortion_checks':{s:distortion_diagnostics(mats['K_'+s],mats['D_'+s]) for s in ('left','right')}})
    write_json(output/'distortion_model_checks.json',source_rows)


def recovered_observations(output):
    arrays=np.load(output/'recovery_detections.npz')
    selected={1:{'left':'clahe_s3','right':'clahe_s2'},24:{'left':'crop_s2','right':'crop_s1'}}
    results=[]
    for n,meta,images in load_images():
        if n in selected:
            results.append(Observation(meta['pair_id'],*[arrays[f'{n:02}_{s}_{selected[n][s]}'] for s in ('left','right')],SIZE))
    write_json(output/'recovery_selected.json',{'selection':selected,
               'rule':'Full visible 9x6 grid; lowest homography error among visually inspected plausible SB grids. No stereo residual or baseline used for selecting corner variant.'})
    return results


def validation(output):
    obs,prov=observations(output,'accurate',True)
    recovered=recovered_observations(output)
    # The original 1px vertical quality criterion identifies these two images.
    # They remain in separate validation outputs and are never presented as good.
    outlier_numbers=[11,22]
    good=[o for o,p in zip(obs,prov) if p['n'] not in outlier_numbers]
    bad=[o for o,p in zip(obs,prov) if p['n'] in outlier_numbers]
    hold_numbers=[4,10,14,18,21]
    train=[o for o,p in zip(obs,prov) if p['n'] not in outlier_numbers+hold_numbers]
    held=[o for o,p in zip(obs,prov) if p['n'] in hold_numbers]
    write_json(output/'validation_design.json',{'outlier_numbers':outlier_numbers,
               'outlier_rule':'vertical RMS > original 1px per-pair quality target in corrected full-set fit; retrospective diagnostic exclusion',
               'holdout_numbers':hold_numbers,'recovered_holdout_numbers':[1,24],
               'selection_limit':'Retrospective reuse of one capture session; not external future-image validation. Baseline proximity is not model-selection criterion.',
               'model_selection':'k1k2 with separate mono fits and FIX_INTRINSIC: stable sensor-wide radial mapping in every CV fold; slightly best heldout joint stereo RMS in both split schemes.'})
    for model in ('full5','k1k2','k1','zero'):
        experiment(output,f'09_{model}_fixed20_recovered_holdout',obs,model,False,recovered)
        experiment(output,f'10_{model}_fixed18_recovered_holdout',good,model,False,recovered+bad)
        experiment(output,f'11_{model}_fixed13_holdout5',train,model,False,held)
    experiment(output,'12_k1k2_fixed22_including_recovered',obs+recovered,'k1k2',False)
    experiment(output,'13_k1k2_joint18',good,'k1k2',True,recovered+bad)


def bundle_fit(initial,obs,constrain_baseline=False,free_intrinsics=False):
    """Separate exact-95mm experiment; optimize reprojection rather than edit T.

    Board poses, camera rotation, translation direction and optionally both
    cameras' intrinsics are jointly optimized. Raw-pixel reprojection is the loss.
    """
    from scipy.sparse import lil_matrix
    m={k:np.asarray(v,np.float64) for k,v in initial['matrices'].items()}
    kl,dl,kr,dr=m['K_left'],m['D_left'],m['K_right'],m['D_right']
    obj=BOARD.object_points().astype(np.float64)
    intr=np.r_[kl[0,0],kl[1,1],kl[0,2],kl[1,2],dl.ravel()[:2],kr[0,0],kr[1,1],kr[0,2],kr[1,2],dr.ravel()[:2]]
    rvec=cv2.Rodrigues(m['R'])[0].ravel()
    t=m['T'].ravel()
    if constrain_baseline:
        translation=np.array([np.arctan2(t[1],t[0]),np.arctan2(t[2],np.hypot(t[0],t[1]))])
    else:
        translation=t
    offset=12 if free_intrinsics else 0
    riglen=5 if constrain_baseline else 6
    poses=[]
    for o in obs:
        _,rv,tv=cv2.solvePnP(obj,o.left,kl,dl)
        poses.extend(np.r_[rv.ravel(),tv.ravel()])
    x0=np.r_[intr if free_intrinsics else [],rvec,translation,poses]
    def unpack(x):
        if free_intrinsics:
            kvals=x[:12]
            cams=[]
            for i in (0,6):
                fx,fy,cx,cy,k1,k2=kvals[i:i+6]
                cams.extend([np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]),np.array([k1,k2,0.,0.,0.])])
            kl2,dl2,kr2,dr2=cams
        else:
            kl2,dl2,kr2,dr2=kl,dl,kr,dr
        R=cv2.Rodrigues(x[offset:offset+3])[0]
        u=x[offset+3:offset+riglen]
        if constrain_baseline:
            az,el=u
            T=95*np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
        else: T=u
        return kl2,dl2,kr2,dr2,R,T.reshape(3,1)
    def fun(x):
        kl2,dl2,kr2,dr2,R,T=unpack(x)
        residual=[]
        for i,o in enumerate(obs):
            start=offset+riglen+6*i
            rv=x[start:start+3];tv=x[start+3:start+6].reshape(3,1)
            rvr=cv2.Rodrigues(R@cv2.Rodrigues(rv)[0])[0]
            tvr=R@tv+T
            lp=cv2.projectPoints(obj,rv,tv,kl2,dl2)[0].reshape(-1,2)
            rp=cv2.projectPoints(obj,rvr,tvr,kr2,dr2)[0].reshape(-1,2)
            residual.extend((lp-o.left.reshape(-1,2)).ravel())
            residual.extend((rp-o.right.reshape(-1,2)).ravel())
        return np.array(residual)
    sparsity=lil_matrix((216*len(obs),len(x0)),dtype=int)
    for i in range(len(obs)):
        row=216*i;start=offset+riglen+6*i
        sparsity[row:row+216,start:start+6]=1
        sparsity[row+108:row+216,offset:offset+riglen]=1
        if free_intrinsics:
            sparsity[row:row+108,:6]=1;sparsity[row+108:row+216,6:12]=1
    opt=least_squares(fun,x0,jac_sparsity=sparsity.tocsr(),x_scale='jac',ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=800)
    kl,dl,kr,dr,R,T=unpack(opt.x)
    rect=rectification(kl,dl,kr,dr,R,T)
    result=json.loads(json.dumps(initial))
    t0,t1,t2=T.ravel()
    E=np.array([[0,-t2,t1],[t2,0,-t0],[-t1,t0,0]])@R
    F=np.linalg.inv(kr).T@E@np.linalg.inv(kl)
    result['matrices']={k:v.tolist() for k,v in dict(K_left=kl,D_left=dl.reshape(1,-1),K_right=kr,D_right=dr.reshape(1,-1),
                           R=R,T=T,E=E,F=F,R1=rect[0],R2=rect[1],P1=rect[2],P2=rect[3],Q=rect[4]).items()}
    result['training_evaluation']=evaluate(obs,kl,dl,kr,dr,R,T,rect)
    train=result['training_evaluation']
    result['roi_left']=list(rect[5]);result['roi_right']=list(rect[6])
    result['rectification_coverage']=coverage(kl,dl,kr,dr,R,T)
    baseline=float(np.linalg.norm(T))
    result['report'].update(left_reprojection_rms_px=train['left_rms_px'],right_reprojection_rms_px=train['right_rms_px'],
                           stereo_rms_px=float(np.sqrt(np.mean(opt.fun.reshape(-1,2)**2)*2)),vertical_rms_px=train['vertical_rms_px'],
                           baseline=baseline,baseline_cm=baseline/10,baseline_relative_difference=abs(baseline-95)/95,
                           relative_rotation_degrees=float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))))
    result['experiment_settings'].update(baseline_constraint=95 if constrain_baseline else None,free_intrinsics_in_scipy=free_intrinsics)
    result['optimizer']={'success':bool(opt.success),'status':int(opt.status),'message':opt.message,'evaluations':int(opt.nfev),
                         'optimality':float(opt.optimality),'initial_stereo_rms_px':float(np.sqrt(np.mean(fun(x0).reshape(-1,2)**2)*2))}
    if not opt.success:
        result['report']['warnings'].append('SciPy optimizer did not converge; diagnostic only')
        result['report']['numeric_checks_passed']=False
    return result


def constraints(output):
    obs,prov=observations(output,'accurate',True)
    good=[o for o,p in zip(obs,prov) if p['n'] not in (11,22)]
    recovered=recovered_observations(output)
    for count,train in ((20,obs),(18,good)):
        initial=json.loads((output/f'experiments/09_k1k2_fixed20_recovered_holdout.json' if count==20 else output/f'experiments/10_k1k2_fixed18_recovered_holdout.json').read_text())
        for constrained,free in ((False,False),(True,False),(True,True)):
            name=f'14_bundle{count}_{"baseline95" if constrained else "unconstrained"}_{"freeK" if free else "fixedK"}'
            result=bundle_fit(initial,train,constrained,free)
            result['experiment_name']=name
            m={k:np.asarray(v) for k,v in result['matrices'].items()}
            rect=rectification(m['K_left'],m['D_left'],m['K_right'],m['D_right'],m['R'],m['T'])
            result['held_out_evaluation']=evaluate(recovered,m['K_left'],m['D_left'],m['K_right'],m['D_right'],m['R'],m['T'],rect)
            result['test_pair_ids']=[o.pair_id for o in recovered]
            write_json(output/'experiments'/f'{name}.json',result)
            print('BUNDLE',name,result['optimizer'],'stereo',result['report']['stereo_rms_px'],
                  'vertical',result['report']['vertical_rms_px'],'heldout',result['held_out_evaluation']['vertical_rms_px'],flush=True)


def stability(output):
    obs,prov=observations(output,'accurate',True)
    good=[o for o,p in zip(obs,prov) if p['n'] not in (11,22)]
    rng=np.random.default_rng(95022)
    records=[]
    for trial in range(100):
        indices=rng.choice(len(good),len(good),replace=True)
        chosen=[good[i] for i in indices]
        objects=[BOARD.object_points() for _ in chosen]
        left=[o.left for o in chosen];right=[o.right for o in chosen]
        flags=MODELS['k1k2']
        _,kl,dl,*_=cv2.calibrateCamera(objects,left,SIZE,None,None,flags=flags,criteria=CRITERIA)
        _,kr,dr,*_=cv2.calibrateCamera(objects,right,SIZE,None,None,flags=flags,criteria=CRITERIA)
        sr,kl,dl,kr,dr,R,T,*_=cv2.stereoCalibrate(objects,left,right,kl,dl,kr,dr,SIZE,flags=cv2.CALIB_FIX_INTRINSIC,criteria=CRITERIA)
        records.append({'trial':trial,'resampled_indices':indices.tolist(),'baseline_mm':float(np.linalg.norm(T)),
                        'stereo_rms_px':float(sr),'rotation_degrees':float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))),
                        'K_left':kl.tolist(),'K_right':kr.tolist(),'T':T.tolist(),
                        'distortion_checks':{s:distortion_diagnostics(k,d) for s,k,d in (('left',kl,dl),('right',kr,dr))}})
        if trial%20==19: print('BOOTSTRAP',trial+1,flush=True)
    baselines=[r['baseline_mm'] for r in records]
    write_json(output/'bootstrap100.json',{'seed':95022,'model':'k1k2 fixed18','sample_count':100,
               'limitation':'Empirical stability under frame resampling, conditional on the chosen data/model; not an absolute distance confidence interval.',
               'baseline_percentiles_mm':{str(q):float(np.percentile(baselines,q)) for q in (2.5,25,50,75,97.5)},
               'baseline_std_mm':float(np.std(baselines)),
               'nonmonotonic_trials':sum(any(not d['radial_monotonic_on_sensor'] for d in r['distortion_checks'].values()) for r in records),
               'trials':records})


def salvage06(output):
    """A visually initialized grid is refined against real pixels, never filled.

    The initialization and every variant are preserved for review. This frame is
    diagnostic holdout only and cannot change the selected camera parameters.
    """
    n,meta,images=load_images()[5]
    gray=cv2.cvtColor(images[1],cv2.COLOR_BGR2GRAY)
    objects=np.array([[0,0],[8,0],[0,5],[8,5]],np.float32)
    guessed=np.array([[139.,164.],[225.,172.],[133.5,205.9],[225.3,215.]],np.float32)
    H=cv2.getPerspectiveTransform(objects,guessed)
    xy=BOARD.object_points()[:,:2]/22
    initial=cv2.perspectiveTransform(xy.reshape(-1,1,2),H)
    candidates={}
    for window in (2,3,4):
        candidates[f'manual_init_subpix{window}']=cv2.cornerSubPix(gray,initial.copy(),(window,window),(-1,-1),(3,60,1e-5))
    target=25*objects+np.array([50,50],np.float32)
    warpH=cv2.getPerspectiveTransform(guessed,target)
    warped=cv2.warpPerspective(gray,warpH,(300,225))
    png(output/'pair06_right_warped.png',warped)
    for variant,g in [('warp',warped),('warp_clahe',cv2.createCLAHE(2.,(4,4)).apply(warped))]:
        found,p=cv2.findChessboardCornersSB(g,(9,6),cv2.CALIB_CB_NORMALIZE_IMAGE|cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY)
        if found:
            candidates[variant]=cv2.perspectiveTransform(p.reshape(-1,1,2),np.linalg.inv(warpH)).astype(np.float32)
    records=[]
    for name,p in candidates.items():
        p,_=canonical(p)
        candidates[name]=p
        scores=[]
        for point in p.reshape(-1,2):
            offsets=np.array([[-2,-2],[2,-2],[-2,2],[2,2]],np.float32)
            vals=[float(cv2.getRectSubPix(gray,(1,1),tuple(point+off))[0,0]) for off in offsets]
            scores.append((vals[0]+vals[3]-vals[1]-vals[2])/2)
        expected=np.array([(-1)**(row+col) for row in range(6) for col in range(9)])
        aligned=np.asarray(scores)*expected
        polarity=max(float(np.mean(aligned>0)),float(np.mean(aligned<0)))
        records.append({'variant':name,'grid':grid_metrics(p,BOARD),'checker_saddle_polarity_fraction':polarity,
                        'checker_saddle_absolute_median':float(np.median(np.abs(scores))),
                        'checker_saddle_absolute_min':float(np.min(np.abs(scores)))})
        png(output/'corners'/f'06_right_salvage_{name}.png',numbered_corners(images[1],p,BOARD,'06 RIGHT '+name,4))
    np.savez_compressed(output/'salvage06_detections.npz',**candidates)
    write_json(output/'salvage06_audit.json',{'guessed_extreme_inner_corners':guessed.tolist(),
               'initial_homography':H.tolist(),'candidates':records,'selection_status':'awaiting direct visual verification; no calibration fit uses this image'})
    print('SALVAGE06',records,flush=True)


def hypotheses(output):
    obs,prov=observations(output,'accurate',True)
    good=[o for o,p in zip(obs,prov) if p['n'] not in (11,22)]
    transforms={
        '15_right_horizontal_mirror_hypothesis':lambda p:np.c_[239-p.reshape(-1,2)[:,0],p.reshape(-1,2)[:,1]],
        '15_right_vertical_mirror_hypothesis':lambda p:np.c_[p.reshape(-1,2)[:,0],319-p.reshape(-1,2)[:,1]],
        '15_right_180_pixel_rotation_hypothesis':lambda p:np.c_[239-p.reshape(-1,2)[:,0],319-p.reshape(-1,2)[:,1]],
    }
    for name,transform in transforms.items():
        transformed=[Observation(o.pair_id,o.left,transform(o.right).astype(np.float32),SIZE) for o in good]
        result=experiment(output,name,transformed,'k1k2',False)
        result['input_space']='diagnostic_transformed_coordinates_not_deployable'
        result['hypothesis_only']=True
        write_json(output/'experiments'/f'{name}.json',result)
    swapped=[Observation(o.pair_id,o.right,o.left,SIZE) for o in good]
    result=experiment(output,'15_swapped_camera_labels_hypothesis',swapped,'k1k2',False)
    result['orientation']=OrientationSettings('ccw90','cw90').to_dict()
    result['camera_mapping']='Requires swapping raw input streams; physical identities unverified'
    result['hypothesis_only']=True
    write_json(output/'experiments/15_swapped_camera_labels_hypothesis.json',result)


def finalize(output):
    """Export the selected unconstrained calibration and all-frame previews."""
    from .calibration import validate_result
    from .rectify import rectify_pair
    obs,prov=observations(output,'accurate',True)
    recovered=recovered_observations(output)
    good=[o for o,p in zip(obs,prov) if p['n'] not in (11,22)]
    bad=[o for o,p in zip(obs,prov) if p['n'] in (11,22)]
    result=json.loads((output/'experiments/10_k1k2_fixed18_recovered_holdout.json').read_text())
    m={k:np.asarray(v) for k,v in result['matrices'].items()}
    kl,dl,kr,dr,R,T=(m[k] for k in ('K_left','D_left','K_right','D_right','R','T'))
    rect=rectification(kl,dl,kr,dr,R,T)
    result['automatic_recovered_holdout_evaluation']=evaluate(recovered,kl,dl,kr,dr,R,T,rect)
    # Accepted only after the warped SB grid and all 54 checker saddles were
    # inspected. Manual initialization itself was rejected, not treated as data.
    if (output/'salvage06_detections.npz').exists():
        _,meta,_=load_images()[5]
        lp=np.load(output/'recovery_detections.npz')['06_left_gamma05_s1']
        rp=np.load(output/'salvage06_detections.npz')['warp']
        six=Observation(meta['pair_id'],lp,rp,SIZE)
        result['assisted_recovered_holdout_evaluation']=evaluate([six],kl,dl,kr,dr,R,T,rect)
        recovered=recovered+[six]
        salvage=json.loads((output/'salvage06_audit.json').read_text())
        salvage['selection_status']='Accepted warp SB after visual inspection and checker saddle polarity 54/54; never used for fitting camera parameters'
        salvage['selected_variant']='warp'
        write_json(output/'salvage06_audit.json',salvage)
    result['held_out_evaluation']=evaluate(recovered,kl,dl,kr,dr,R,T,rect)
    result['excluded_outlier_evaluation']=evaluate(bad,kl,dl,kr,dr,R,T,rect)
    result['test_pair_ids']=[o.pair_id for o in recovered]
    result['rectification_coverage']=coverage(kl,dl,kr,dr,R,T)
    cv18=json.loads((output/'cross_validation_clean18_summary.json').read_text())
    result['internal_cross_validation']=[r for r in cv18 if r['model']=='k1k2']
    result['selection_reason']='Unconstrained k1/k2 radial model with fixed separately estimated intrinsics; all CV fold radial mappings remain monotonic; stable heldout geometry; two 1px vertical outliers retained in a separate diagnostic evaluation.'
    result['report'].update(validation_status='internal_geometry_validated_external_metric_accuracy_unverified',
                            quality_evaluation='train18; additional automatic holdout2 plus assisted holdout1; retrospective 5-fold cross-validation',
                            corner_policy='experimental_accurate_SB_with_inspected_180_alignment',
                            physical_distance_accuracy_verified=False,
                            per_pair=result['training_evaluation']['per_pair'],
                            limitations=['Absolute distance accuracy has no independently measured depth validation.',
                                         'Camera identity and common mirroring are unverified; disparity is negative for the supplied stream order.',
                                         'Pair 11 and pair 22 fail the per-pair 1px vertical target after coordinate repair.'])
    result['report'].update(expected_baseline_cm=9.5,total_pairs=24,excluded_pairs=6,detection_failed_pairs=1,
                            per_pair=[{'pair_id':p['pair_id'],'left_reprojection_rms_px':p['left_rms_px'],
                                       'right_reprojection_rms_px':p['right_rms_px'],'vertical_rms_px':p['vertical_rms_px'],
                                       'vertical_max_px':p['vertical_max_px']} for p in result['training_evaluation']['per_pair']],
                            vertical_p95_px=result['training_evaluation']['vertical_p95_px'],
                            vertical_max_px=result['training_evaluation']['vertical_max_px'])
    dy=[]
    for o in good:
        a=cv2.undistortPoints(o.left,kl,dl,R=rect[0],P=rect[2]).reshape(-1,2)
        b=cv2.undistortPoints(o.right,kr,dr,R=rect[1],P=rect[3]).reshape(-1,2)
        dy.extend(np.abs(a[:,1]-b[:,1]).tolist())
    result['report']['vertical_mean_abs_px']=float(np.mean(dy))
    result['report']['excluded']=[{'pair_directory':str(DATA/('pair_'+p['pair_id'])),
                                  'reason':'held out for validation' if p['n'] in (1,6,24) else ('full board cropped in RIGHT' if p['n']==7 else 'corrected per-pair vertical RMS exceeds original 1px target')}
                                 for p in json.loads((output/'audit.json').read_text()) if p['n'] in (1,6,7,11,22,24)]
    result['calibration_method']='unconstrained_baseline_k1k2_fixed_intrinsics'
    result['baseline_constraint_mm']=None
    result['report']['physical_use_approved']=False
    result['distortion_checks']={s:distortion_diagnostics(m['K_'+s],m['D_'+s]) for s in ('left','right')}
    result['experiment_settings']['termination_criteria']=list(CRITERIA)
    validate_result(result)
    result['geometric_checks']={'R_determinant':float(np.linalg.det(R)),
                                'R_orthogonality_error':float(np.linalg.norm(R.T@R-np.eye(3))),
                                'right_camera_center_in_left_coordinates_mm':(-R.T@T).ravel().tolist()}
    test_xyz=np.array([[-100,-80,500],[0,0,1000],[100,80,2000],[50,-30,700]],np.float64)
    homogeneous=np.c_[test_xyz,np.ones(len(test_xyz))]
    pl=(rect[2]@homogeneous.T).T;pr=(rect[3]@homogeneous.T).T
    pl=pl[:,:2]/pl[:,2:3];pr=pr[:,:2]/pr[:,2:3]
    reconstructed=cv2.perspectiveTransform(np.c_[pl,pl[:,0]-pr[:,0]].reshape(-1,1,3),rect[4]).reshape(-1,3)
    result['geometric_checks']['Q_exact_projection_roundtrip_max_mm']=float(np.max(np.abs(reconstructed-test_xyz)))
    result['dataset']=str(DATA)
    result['source_sha256']=json.loads((output/'source_hashes_before.json').read_text())
    write_json(output/'calibration_session02_best.json',result)
    (output/'rectified').mkdir(exist_ok=True)
    allobs={o.pair_id:o for o in obs+recovered}
    errors={r['pair_id']:r for r in result['training_evaluation']['per_pair']+result['held_out_evaluation']['per_pair']+result['excluded_outlier_evaluation']['per_pair']}
    qerrors=[]
    crop_sheets=[]
    for n,meta,images in load_images():
        directory=DATA/('pair_'+meta['pair_id'])
        _,_,raw=load_pair(directory/'pair.json')
        corrected=rectify_pair(*raw,result)
        group='TRAIN18' if n not in (1,6,7,11,22,24) else ('HOLDOUT' if n in (1,6,24) else ('OUTLIER' if n in (11,22) else 'CROPPED'))
        preview=np.hstack(corrected)
        for y in range(20,320,20): cv2.line(preview,(0,y),(479,y),(0,180,0),1)
        scaled=cv2.resize(preview,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
        header=f'{n:02} {group} | '
        if meta['pair_id'] in errors: header+=f'vertical RMS {errors[meta["pair_id"]]["vertical_rms_px"]:.3f}px'
        else: header+='partial board; no full-grid numerical evaluation'
        scaled=cv2.copyMakeBorder(scaled,42,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(scaled,header,(8,27),0,.65,(255,255,255),1)
        png(output/'rectified'/f'{n:02}_rectified.png',scaled)
        if meta['pair_id'] in allobs:
            o=allobs[meta['pair_id']]
            lr=cv2.undistortPoints(o.left,kl,dl,R=rect[0],P=rect[2]).reshape(-1,2)
            rr=cv2.undistortPoints(o.right,kr,dr,R=rect[1],P=rect[3]).reshape(-1,2)
            lu=cv2.undistortPoints(o.left,kl,dl).reshape(-1,2)
            ru=cv2.undistortPoints(o.right,kr,dr).reshape(-1,2)
            h=cv2.triangulatePoints(np.c_[np.eye(3),np.zeros(3)],np.c_[R,T.ravel()],lu.T,ru.T)
            xyz=(h[:3]/h[3]).T
            vd=np.c_[lr[:,0],lr[:,1],lr[:,0]-rr[:,0]].reshape(-1,1,3)
            qxyz=cv2.perspectiveTransform(vd.astype(np.float64),rect[4]).reshape(-1,3)
            qerrors.append({'n':n,'Q_vs_direct_xyz_rms_mm':float(np.sqrt(np.mean(np.sum((qxyz-(rect[0]@xyz.T).T)**2,axis=1)))),
                            'positive_Q_depth_fraction':float(np.mean(qxyz[:,2]>0)),
                            'vertical_mean_signed_px':float(np.mean(lr[:,1]-rr[:,1])),
                            'vertical_std_px':float(np.std(lr[:,1]-rr[:,1]))})
            panels=[numbered_corners(im,cp.reshape(-1,1,2),BOARD,f'{n:02} {s} {group} RECTIFIED',3)
                    for s,im,cp in (('LEFT',corrected[0],lr),('RIGHT',corrected[1],rr))]
            png(output/'rectified'/f'{n:02}_rectified_numbered.png',np.hstack(panels))
            crops=[]
            for image,points,side in ((images[0],o.left,'LEFT'),(images[1],o.right,'RIGHT')):
                p=points.reshape(-1,2)
                x0=max(0,int(p[:,0].min())-12);x1=min(240,int(p[:,0].max())+13)
                y0=max(0,int(p[:,1].min())-12);y1=min(320,int(p[:,1].max())+13)
                patch=image[y0:y1,x0:x1]
                translated=p-np.array([x0,y0])
                debugscale=min(680/(x1-x0),430/(y1-y0))
                panel=numbered_corners(patch,translated.reshape(-1,1,2),BOARD,f'{n:02} {side}',debugscale)
                panel=cv2.copyMakeBorder(panel,0,max(0,500-panel.shape[0]),0,max(0,720-panel.shape[1]),cv2.BORDER_CONSTANT)
                crops.append(panel[:500,:720])
            crop_sheets.append((n,np.hstack(crops)))
    for start in range(0,len(crop_sheets),4):
        batch=crop_sheets[start:start+4]
        png(output/f'board_detail_{batch[0][0]:02}_{batch[-1][0]:02}.png',np.vstack([b[1] for b in batch]))
    write_json(output/'q_validation.json',qerrors)
    write_json(output/'runtime_manifest.json',{'python':__import__('sys').version,'numpy':np.__version__,
               'scipy':__import__('scipy').__version__,'opencv':cv2.__version__,
               'source_code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'app/stereo/diagnose.py',ROOT/'app/stereo/dataset.py',ROOT/'app/stereo/orientation.py',ROOT/'app/stereo/rectify.py',ROOT/'app/stereo/calibration.py']},
               'scope':'Experiment and report modules were added by this task; calibration.py received concurrent outside changes. Core experimental fitting is implemented here independently.'})
    rows=[]
    for p in sorted((output/'experiments').glob('*.json')):
        d=json.loads(p.read_text())
        d['experiment_settings']['termination_criteria']=list(CRITERIA)
        if d.get('optimizer') and not d['optimizer']['success']:
            d['report']['numeric_checks_passed']=False
            if 'SciPy optimizer did not converge; diagnostic only' not in d['report']['warnings']:
                d['report']['warnings'].append('SciPy optimizer did not converge; diagnostic only')
        write_json(p,d)
        r=d['report'];held=d.get('held_out_evaluation') or {}
        rows.append({'experiment':p.stem,'n_train':r['used_pairs'],'n_test':held.get('pairs',0),
                    'left_rms_px':r['left_reprojection_rms_px'],'right_rms_px':r['right_reprojection_rms_px'],
                    'stereo_rms_px':r['stereo_rms_px'],'vertical_rms_px':r['vertical_rms_px'],'baseline_mm':r['baseline'],
                    'test_stereo_rms_px':held.get('stereo_joint_pose_rms_px'),'test_vertical_rms_px':held.get('vertical_rms_px'),
                    'baseline_constraint_mm':d['experiment_settings'].get('baseline_constraint'),
                    'optimizer_converged':(d.get('optimizer') or {}).get('success'),
                    'numeric_checks_passed':r['numeric_checks_passed']})
    with (output/'experiment_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    preserved=hashes()==json.loads((output/'source_hashes_before.json').read_text())
    write_json(output/'source_preservation.json',{'preserved':preserved,'after':hashes()})
    if not preserved: raise RuntimeError('Original data hashes changed')
    print('SELECTED', {k:v for k,v in result['report'].items() if k!='per_pair'},
          'HOLDOUT', {k:v for k,v in result['held_out_evaluation'].items() if k!='per_pair'},flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['audit','suite','recover','cv','cv18','diagnostics','validation','constraints','stability','hypotheses','finalize','salvage06'])
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    {'audit':audit,'suite':suite,'recover':recover,'cv':cross_validation,'diagnostics':diagnostics,
     'validation':validation,'constraints':constraints,'stability':stability,
     'cv18':lambda p:cross_validation(p,True),'hypotheses':hypotheses,'finalize':finalize,'salvage06':salvage06}[args.stage](args.output.resolve())


if __name__ == '__main__':
    main()
