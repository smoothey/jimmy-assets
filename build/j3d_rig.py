"""3D Jimmy: his skeleton, skinning and sitting rest pose (JM-491). Runs inside Blender.

Bones ending .L are on his left (+X, the right of the screen from the front).
He is skinned standing, then sat down as the drawing has him, and the sitting
pose is made his rest pose: every move after that is a change from sitting.
"""
import math

import bpy
from mathutils import Matrix, Quaternion, Vector

import j3d_mats
import j3d_model as mdl
from j3d_common import collection

V = Vector
ARM = 'Jimmy'
SKIN = ('hips', 'spine', 'chest', 'neck', 'head', 'tail.1', 'tail.2', 'tail.3', 'tail.4', 'tail.5',
        'upper_arm', 'forearm', 'paw', 'thigh', 'shin', 'hock', 'foot')


def _bones():
    """name: (head, tail, parent, joined to parent)"""
    m = mdl
    b = {
        'root': (V((0, 0, 0)), V((0, -0.25, 0)), None, False),
        'hips': (m.HIP, m.SPINE, 'root', False),
        'spine': (m.SPINE, m.CHEST, 'hips', True),
        'chest': (m.CHEST, m.NECK, 'spine', True),
        'neck': (m.NECK, m.HEADJ, 'chest', True),
        'head': (m.HEADJ, m.HEAD_END, 'neck', True),
    }
    for i in range(5):
        b['tail.%d' % (i + 1)] = (m.TAIL[i], m.TAIL[i + 1], 'hips' if i == 0 else 'tail.%d' % i, i > 0)
    for side, sx in (('L', 1), ('R', -1)):
        def X(v):
            return V((sx * v.x, v.y, v.z))
        b['upper_arm.' + side] = (X(m.SH), X(m.EL), 'chest', False)
        b['forearm.' + side] = (X(m.EL), X(m.WR), 'upper_arm.' + side, True)
        b['paw.' + side] = (X(m.WR), X(m.TOE), 'forearm.' + side, True)
        b['thigh.' + side] = (X(m.HP), X(m.KN), 'hips', False)
        b['shin.' + side] = (X(m.KN), X(m.HK), 'thigh.' + side, True)
        b['hock.' + side] = (X(m.HK), X(m.BALL), 'shin.' + side, True)
        b['foot.' + side] = (X(m.BALL), X(m.TOEH), 'hock.' + side, True)
        eb = m.HC + X(m.EAR_BASE)
        b['ear.' + side] = (eb, eb + (m.HC + X(m.EAR_TIP) - eb) * 0.6, 'head', False)
        eo = bpy.data.objects.get('eye_' + side)
        ec = V(eo['c']) if eo and eo.get('c') else m.HC + X(m.EYE)
        for n in ('eye', 'lid_up', 'lid_lo'):
            b[n + '.' + side] = (ec, ec + V((0, -0.12, 0)), 'head', False)
        bc = m.HC + V((sx * 0.125, -0.215, m.BROW_Z + 0.015))
        b['brow.' + side] = (bc, bc + V((0, 0, 0.05)), 'head', False)
    mo = bpy.data.objects.get('mouth_in')
    mc = V(mo['c']) if mo and mo.get('c') else m.HC + V((0, -0.18, m.MOUTH_Z))
    b['mouth'] = (mc, mc + V((0, -0.08, 0)), 'head', False)
    b['headset'] = (m.HC + V((0, 0.05, -0.035)), m.HC + V((0, 0.05, 0.10)), 'head', False)
    b['mic'] = (m.HC + m.MIC_FROM, m.HC + m.MIC_TO, 'headset', False)
    b['hat'] = (m.HC + V((0.012, -0.015, 0.258)), m.HC + V((0.022, -0.022, 0.598)), 'head', False)
    return b


def build_armature():
    old = bpy.data.objects.get(ARM)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    arm = bpy.data.armatures.new(ARM)
    obj = bpy.data.objects.new(ARM, arm)
    obj['jimmy'] = True
    collection().objects.link(obj)
    obj.show_in_front = True
    arm.display_type = 'STICK'
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    spec = _bones()
    # The tag hangs from the collar: find where the model put it.
    tag = bpy.data.objects.get('tag')
    if tag:
        zs = [v.co for v in tag.data.vertices]
        top = max(zs, key=lambda c: c.z)
        lo = min(zs, key=lambda c: c.z)
        spec['tag'] = (V((0, top.y, top.z)), V((0, lo.y, lo.z + 0.05)), 'neck', False)
    for name, (h, t, parent, joined) in spec.items():
        e = arm.edit_bones.new(name)
        e.head, e.tail = h, t
        e.roll = 0
    for name, (h, t, parent, joined) in spec.items():
        if parent:
            e = arm.edit_bones[name]
            e.parent = arm.edit_bones[parent]
            e.use_connect = joined
    bpy.ops.object.mode_set(mode='OBJECT')
    for b in arm.bones:
        b.use_deform = b.name.split('.')[0] in ('hips', 'spine', 'chest', 'neck', 'head', 'tail', 'upper_arm', 'forearm', 'paw', 'thigh', 'shin', 'hock', 'foot')
    return obj


def skin(arm):
    body = bpy.data.objects['body']
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.get('jimmy'):
            for m in list(o.modifiers):
                o.modifiers.remove(m)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    body.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    for b in arm.data.bones:
        b.use_deform = True
    for o in bpy.data.objects:
        if o.type != 'MESH' or not o.get('jimmy') or o is body:
            continue
        o.parent = arm
        mod = o.modifiers.new('rig', 'ARMATURE')
        mod.object = arm
    weights = {g.name for g in body.vertex_groups}
    return sorted(weights)


def add_lines():
    line = bpy.data.materials['J_line']
    thin = ('whiskers', 'brow', 'mouth_line', 'chin_line', 'lash', 'eye_rim', 'glint', 'pupil', 'eye_', 'tag_text', 'teeth', 'tongue', 'mouth_in', 'lid_')
    fine = ('pads', 'nose', 'cup_glow', 'hat_', 'boom', 'band', 'tag', 'buckle')
    for o in bpy.data.objects:
        if o.type != 'MESH' or not o.get('jimmy') or o.name.startswith(thin):
            continue
        j3d_mats.outline(o, line, 0.0045 if o.name.startswith(fine) else 0.008)


# ------------------------------------------------------------ posing ----
def _upd():
    bpy.context.view_layer.update()


def turn(arm, name, axis, degrees):
    """Turn a bone about an axis of the room, through its own joint."""
    pb = arm.pose.bones[name]
    _upd()
    m = pb.matrix.copy()
    r = Matrix.Rotation(math.radians(degrees), 4, V(axis))
    loc = m.translation.copy()
    new = r @ m
    new.translation = loc
    pb.matrix = new
    _upd()


def aim(arm, name, direction):
    """Point a bone along a direction in the room, with as little twist as it can."""
    pb = arm.pose.bones[name]
    _upd()
    m = pb.matrix.copy()
    y = m.to_3x3().col[1].normalized()
    q = y.rotation_difference(V(direction).normalized())
    loc = m.translation.copy()
    new = q.to_matrix().to_4x4() @ m
    new.translation = loc
    pb.matrix = new
    _upd()


def level(arm, name):
    """Give a bone back the way it faced before anything above it moved."""
    pb = arm.pose.bones[name]
    _upd()
    m = pb.bone.matrix_local.copy()
    m.translation = pb.matrix.translation.copy()
    pb.matrix = m
    _upd()


def move(arm, name, to):
    pb = arm.pose.bones[name]
    _upd()
    m = pb.matrix.copy()
    m.translation = V(to)
    pb.matrix = m
    _upd()


SIT = {
    'hips_at': (0, 0.20, 0.215),
    'hips': 48, 'spine': 11, 'chest': 6,          # degrees each pitches up (65 in all)
    'neck': (0, -0.16, 1.0),
    'arm': ((-0.16, 0.10, -1.0), (0.02, -0.16, -1.0), (0.0, -1.0, -0.22)),       # upper arm, forearm, paw (his left; right mirrored)
    'leg': ((0.42, -0.86, 0.20), (-0.10, 0.62, -0.78), (0.06, -1.0, -0.20), (0.05, -1.0, -0.02)),   # thigh, shin, hock, foot
    'tail': ((-0.92, 0.30, -0.20), (-0.55, -0.25, 0.80), (-0.10, -0.18, 1.0), (0.06, -0.06, 1.0), (-0.66, 0.0, 0.75)),
    'tag': 20,                                    # degrees the tag swings out to lie on his chest
}


def sit(arm, p=SIT):
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    move(arm, 'hips', p['hips_at'])
    for n in ('hips', 'spine', 'chest'):
        turn(arm, n, (1, 0, 0), -p[n])
    aim(arm, 'neck', p['neck'])
    level(arm, 'head')
    level(arm, 'tag')
    turn(arm, 'tag', (1, 0, 0), -p['tag'])
    for side, sx in (('L', 1), ('R', -1)):
        for n, d in zip(('upper_arm', 'forearm', 'paw'), p['arm']):
            aim(arm, n + '.' + side, (sx * d[0], d[1], d[2]))
        for n, d in zip(('thigh', 'shin', 'hock', 'foot'), p['leg']):
            aim(arm, n + '.' + side, (sx * d[0], d[1], d[2]))
    for i, d in enumerate(p['tail']):
        aim(arm, 'tail.%d' % (i + 1), d)
    _upd()
    return {n: [round(c, 3) for c in arm.pose.bones[n].tail] for n in ('paw.L', 'foot.L', 'hock.L', 'tail.5', 'head', 'chest', 'neck', 'upper_arm.L', 'thigh.L', 'shin.L')}


def make_rest(arm):
    """Make the pose he is in now his rest pose. Returns what undoes it (his standing pose)."""
    undo = {pb.name: pb.matrix_basis.inverted() for pb in arm.pose.bones}
    dg = bpy.context.evaluated_depsgraph_get()
    for o in bpy.data.objects:
        if o.type != 'MESH' or not o.get('jimmy'):
            continue
        for m in o.modifiers:
            m.show_viewport = m.type == 'ARMATURE'
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    for o in bpy.data.objects:
        if o.type != 'MESH' or not o.get('jimmy'):
            continue
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        old = o.data
        o.data = me
        me.name = o.name
        bpy.data.meshes.remove(old)
        for m in o.modifiers:
            m.show_viewport = True
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    _upd()
    return undo


def build(step='all'):
    arm = build_armature()
    groups = skin(arm)
    info = sit(arm)
    undo = make_rest(arm)
    add_lines()
    # The gradients on his eyes and ears go by height: move them to where he sits now.
    r = mdl.EYE_R
    ez = arm.data.bones['eye.L'].head_local.z
    j3d_mats.set_height('eye', ez - r * 0.70, ez + r * 0.75)
    j3d_mats.set_height('pupil', ez - r * 0.50, ez + r * 0.20)
    zs = [v.co.z for v in bpy.data.objects['ear_in_L'].data.vertices]
    j3d_mats.set_height('ear_in', min(zs), max(zs))
    arm['stand'] = {k: [x for r in v for x in r] for k, v in undo.items()}
    # His party hat is there all the time and shown by one switch on the rig.
    arm['hat'] = 0.0
    arm.id_properties_ui('hat').update(min=0.0, max=1.0, description='1 shows his party hat')
    for o in bpy.data.objects:
        if not o.get('hat'):
            continue
        for path in ('hide_viewport', 'hide_render'):
            o.driver_remove(path)
            d = o.driver_add(path).driver
            d.type = 'SCRIPTED'
            v = d.variables.new()
            v.name = 'hat'
            v.type = 'SINGLE_PROP'
            v.targets[0].id = arm
            v.targets[0].data_path = '["hat"]'
            d.expression = 'hat < 0.5'
    return {'groups': len(groups), 'sit': info}
