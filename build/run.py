"""3D Jimmy: run the build inside Blender (JM-491).

In Blender's Python console, or through the Blender connector:
    import sys; sys.path.insert(0, r"<this folder>"); import run; run.go()
"""
import importlib
import os
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), 'renders')
NAMES = ('j3d_common', 'j3d_mats', 'j3d_model', 'j3d_scene', 'j3d_rig', 'j3d_moves', 'j3d_export')


def mods():
    out = {}
    for n in NAMES:
        try:
            out[n] = importlib.reload(importlib.import_module(n))
        except ModuleNotFoundError:
            pass
    return out


def sheet(m, items, prefix, size):
    """Render a list of (action, where in it 0..1, view): one picture each."""
    import bpy
    out = []
    sc = m['j3d_scene']
    sc.setup(size)
    for i, (action, at, view) in enumerate(items):
        act = bpy.data.actions[action]
        m['j3d_moves'].show(action, act.frame_start + (act.frame_end - act.frame_start) * at)
        sc.badge_view() if view == 'badge' else sc.view(view)
        path = os.path.join(OUT, '%s%02d_%s.png' % (prefix, i, action))
        bpy.context.scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        out.append(path)
    return out


def go(steps=('model', 'rig', 'moves'), shots=('front', 'three'), prefix='t_', size=800, action=None, frame=None, only=None, **kw):
    try:
        m = mods()
        t = time.time()
        r = {}
        if 'model' in steps:
            m['j3d_common'].wipe()
            r['model'] = m['j3d_model'].build()
        if 'rig' in steps:
            r['rig'] = m['j3d_rig'].build()
        if 'moves' in steps and 'j3d_moves' in m:
            r['moves'] = m['j3d_moves'].build()
        if action and 'j3d_moves' in m:
            r['frame'] = m['j3d_moves'].show(action, frame)
        if shots:
            r['shots'] = m['j3d_scene'].shots(OUT, list(shots), prefix=prefix, size=size, only=only)['files']
        if kw.get('sheet'):
            r['sheet'] = sheet(m, kw['sheet'], prefix, size)
        r['secs'] = round(time.time() - t, 1)
        return r
    except Exception:
        return {'error': traceback.format_exc()[-3500:]}
