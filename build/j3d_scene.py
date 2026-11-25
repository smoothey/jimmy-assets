"""3D Jimmy: camera, light and test renders (JM-491). Runs inside Blender."""
import math
import os

import bpy
from mathutils import Vector

VIEWS = {
    # name: the way the camera looks (his front faces -Y)
    'front': (0, 1, 0),
    'three': (0.5, 0.86, -0.18),
    'three_l': (-0.5, 0.86, -0.18),
    'side': (1, 0, 0),
    'side_l': (-1, 0, 0),
    'back': (-0.35, -0.9, -0.2),
    'top': (0, 0.35, -0.94),
    'low': (0.2, 0.9, 0.38),
    'face': (0, 1, 0),
}


def bounds(names=None):
    """The box round Jimmy as he is posed now (hidden things left out)."""
    dg = bpy.context.evaluated_depsgraph_get()
    lo = Vector((1e9, 1e9, 1e9))
    hi = -lo
    for o in bpy.data.objects:
        if o.type != 'MESH' or not o.get('jimmy') or o.hide_render:
            continue
        if names and not any(o.name.startswith(n) for n in names):
            continue
        e = o.evaluated_get(dg)
        for c in e.bound_box:
            w = e.matrix_world @ Vector(c)
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    return lo, hi


def _obj(name, data_new):
    o = bpy.data.objects.get(name)
    if not o:
        o = bpy.data.objects.new(name, data_new())
        bpy.context.scene.collection.objects.link(o)
    return o


def setup(size=900):
    sc = bpy.context.scene
    for n in ('Camera', 'Light'):
        o = bpy.data.objects.get(n)
        if o and not o.get('jimmy_scene'):
            bpy.data.objects.remove(o, do_unlink=True)
    cam = _obj('JimmyCam', lambda: bpy.data.cameras.new('JimmyCam'))
    cam['jimmy_scene'] = True
    sc.camera = cam
    sun = _obj('JimmySun', lambda: bpy.data.lights.new('JimmySun', 'SUN'))
    sun['jimmy_scene'] = True
    sun.data.energy = 3.14
    sun.data.angle = math.radians(2)
    sun.rotation_euler = Vector((0.35, 0.80, -0.50)).to_track_quat('-Z', 'Y').to_euler()
    w = sc.world or bpy.data.worlds.new('World')
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes.get('Background')
    if bg:
        bg.inputs['Color'].default_value = (0, 0, 0, 1)
        bg.inputs['Strength'].default_value = 0.0
    sc.render.engine = 'BLENDER_EEVEE'
    sc.render.film_transparent = True
    sc.render.resolution_x = sc.render.resolution_y = size
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.view_settings.view_transform = 'Standard'
    sc.view_settings.look = 'None'
    sc.render.fps = 30
    return cam


def view(name, only=None, pad=1.12):
    """Point the camera along a named view and fit Jimmy (or the named parts) in the frame."""
    cam = bpy.context.scene.camera
    d = Vector(VIEWS[name]).normalized()
    lo, hi = bounds(only)
    c = (lo + hi) / 2
    q = d.to_track_quat('-Z', 'Y')
    cam.rotation_euler = q.to_euler()
    cam.location = c - d * 30
    cam.data.type = 'ORTHO'
    right = q @ Vector((1, 0, 0))
    up = q @ Vector((0, 1, 0))
    corners = [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]
    w = max(abs((p - c).dot(right)) for p in corners) * 2
    h = max(abs((p - c).dot(up)) for p in corners) * 2
    cam.data.ortho_scale = max(w, h) * pad
    cam.data.clip_end = 100


def badge_view():
    """The drawing's own framing: his head as wide in the frame as it is in the badge."""
    cam = bpy.context.scene.camera
    arm = bpy.data.objects['Jimmy']
    bpy.context.view_layer.update()
    hc = arm.matrix_world @ arm.pose.bones['head'].head + Vector((0, -0.05, 0.13))
    d = Vector((0, 1, 0))
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    cam.location = Vector((hc.x - 0.004, hc.y, hc.z - 0.030)) - d * 30
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 1.23
    cam.data.clip_end = 100


def shots(folder, names, prefix='', size=900, frame=None, only=None):
    setup(size)
    if frame is not None:
        bpy.context.scene.frame_set(frame)
    out = []
    for n in names:
        badge_view() if n == 'badge' else view(n, only)
        path = os.path.join(folder, prefix + n + '.png')
        bpy.context.scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        out.append(path)
    return {'files': out}
