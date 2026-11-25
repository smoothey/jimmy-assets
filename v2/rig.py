"""Jimmy from his own layers (JM-491): build him in Blender, rig him, give him the flat rig's moves.

Reads what gen.py made (parts.npz, parts.json, the pictures) and moves.json (from
dump-moves.cjs). Every part and bone has the flat rig's own name, so its actions and
poses carry over one for one. The differences are the ones 3D makes possible: a look to
the side is a real turn of his head, and a nod is a real nod.

    import sys; sys.path.insert(0, r"<this folder>"); import rig; rig.build()
"""
import json
import math
import os

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = 'Jimmy2'
FPS = 30
LINE = (0.208, 0.086, 0.102)
SHOWN = ('closed-left', 'closed-right', 'mouth-talk', 'mouth-wide', 'hat')     # parts that fade in and out
FADED = SHOWN + ('paw', 'leg')   # with his whole body, his raised arm and the front leg it replaces swap over
ARM_DOWN = 132                   # degrees his raised arm folds down to hang behind his body, out of sight
ARM_MS = 340                     # how long his paw takes to come up, and to go back down
YAW, PITCH = 9.0, 4.0            # degrees his head turns and nods for each unit the flat rig slides it
SLOPES = {}                      # part: how the surface under it slopes (filled in from parts.json)
EYEBALLS = {}                    # pupil: the ball it turns on, {'centre': [x, y, z], 'R': radius}
EXTRA_DEPTH = {'ear-left': 0.075, 'ear-right': 0.075, 'tail': 0.16, 'tail-mid': 0.16, 'tail-tip': 0.16, 'paw': 0.13, 'forearm': 0.13}


def lin(c):
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]


def smooth(a, lo, hi):
    t = np.clip((a - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def meta():
    with open(os.path.join(HERE, 'parts.json')) as f:
        return json.load(f)


def room(m, px, py, depth=0.0):
    """A point of the drawing (flat-rig units) in the room."""
    return Vector(((px * m['U'] - m['SRC'] / 2) * m['S'], depth, (m['SRC'] - py * m['U']) * m['S']))


# ------------------------------------------------------------- materials ----
def image(name):
    return bpy.data.images.load(os.path.join(HERE, name), check_existing=True)


def material(name, part, side, m):
    mat = bpy.data.materials.new('J2_%s_%s' % (name, side))
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    N, L = nt.nodes.new, nt.links.new
    out, em = N('ShaderNodeOutputMaterial'), N('ShaderNodeEmission')

    def tex(file, uv, clip=True):
        t = N('ShaderNodeTexImage')
        t.image = image(file)
        t.extension = 'CLIP' if clip else 'EXTEND'
        u = N('ShaderNodeUVMap')
        u.uv_map = uv
        L(u.outputs[0], t.inputs['Vector'])
        return t, u

    def attr(aname, kind='GEOMETRY'):
        a = N('ShaderNodeAttribute')
        a.attribute_type = kind
        a.attribute_name = aname
        return a

    def mix(a, b, fac):
        x = N('ShaderNodeMix')
        x.data_type = 'RGBA'
        for sock, v in ((x.inputs['A'], a), (x.inputs['B'], b)):
            if isinstance(v, (tuple, list)):
                sock.default_value = v
            else:
                L(v, sock)
        L(fac, x.inputs['Factor'])
        return x.outputs['Result']

    def times(a, b):
        x = N('ShaderNodeMath')
        x.operation = 'MULTIPLY'
        L(a, x.inputs[0])
        L(b, x.inputs[1])
        return x.outputs[0]

    if part.get('plain'):                         # one flat colour, seen from either side (the inside of his eye sockets)
        em.inputs['Color'].default_value = (*lin(part['colour']), 1)
        L(em.outputs[0], out.inputs['Surface'])
        mat.use_backface_culling = False
        return mat
    if side == 'front':
        art, uvn = tex(name + '.png', 'UVMap')
        colour, alpha = art.outputs['Color'], art.outputs['Alpha']
        if part.get('side'):                      # his head: the drawing on the front, Gemini's side view round the sides
            st, _ = tex('side.png', 'side', clip=False)
            w = attr('mixw').outputs['Fac']
            colour = mix(colour, st.outputs['Color'], w)
            mx = N('ShaderNodeMath')
            mx.operation = 'MAXIMUM'
            L(alpha, mx.inputs[0])
            L(w, mx.inputs[1])
            alpha = mx.outputs[0]
        if part['two_sided'] and not part.get('side'):
            # Seen from the side, the rim of a part shows the last few pixels of the drawing's edge, stretched and
            # half see-through. Where the rim faces the camera, paint it the drawing's line colour instead. From
            # straight on the rim is edge-on, so this does nothing and the drawing is untouched.
            geo, dot, ramp = N('ShaderNodeNewGeometry'), N('ShaderNodeVectorMath'), N('ShaderNodeMapRange')
            dot.operation = 'DOT_PRODUCT'
            L(geo.outputs['Normal'], dot.inputs[0])
            L(geo.outputs['Incoming'], dot.inputs[1])
            ramp.interpolation_type = 'SMOOTHSTEP'
            ramp.inputs['From Min'].default_value, ramp.inputs['From Max'].default_value = 0.40, 0.60
            L(dot.outputs['Value'], ramp.inputs['Value'])
            rim = times(attr('rim').outputs['Fac'], ramp.outputs['Result'])
            colour = mix(colour, (*lin(LINE), 1), rim)
            mx = N('ShaderNodeMath')
            mx.operation = 'MAXIMUM'
            L(alpha, mx.inputs[0])
            L(rim, mx.inputs[1])
            alpha = mx.outputs[0]
        if part.get('clip'):                      # pupils and lids show only inside his eyes, wherever they have slid to
            mask = N('ShaderNodeTexImage')
            mask.image = image(m.get('mask') or part['clip'] + '.png')
            mask.extension = 'CLIP'
            off = N('ShaderNodeCombineXYZ')
            L(attr('ox', 'OBJECT').outputs['Fac'], off.inputs['X'])
            L(attr('oz', 'OBJECT').outputs['Fac'], off.inputs['Y'])
            add = N('ShaderNodeVectorMath')
            L(uvn.outputs[0], add.inputs[0])
            L(off.outputs[0], add.inputs[1])
            L(add.outputs[0], mask.inputs['Vector'])
            alpha = times(alpha, mask.outputs['Alpha'])
        if name in FADED:
            alpha = times(alpha, attr('show', 'OBJECT').outputs['Fac'])
    else:
        # The back: the drawing's line colour at the outline, his own colour further round.
        colour = mix((*lin(LINE), 1), (*lin(part['colour']), 1), attr('shade').outputs['Fac'])
        alpha = attr('inside').outputs['Fac']
        if m.get('back') and name in ('head', 'ear-left', 'ear-right'):
            bt, _ = tex('back.png', 'back', clip=False)
            if part.get('side'):
                st, _ = tex('side.png', 'side', clip=False)
                colour = mix(st.outputs['Color'], bt.outputs['Color'], attr('mixw').outputs['Fac'])
            else:
                colour = bt.outputs['Color']
        if name in FADED:
            alpha = times(alpha, attr('show', 'OBJECT').outputs['Fac'])
    tr, ms = N('ShaderNodeBsdfTransparent'), N('ShaderNodeMixShader')
    L(colour, em.inputs['Color'])
    L(alpha, ms.inputs['Fac'])
    L(tr.outputs[0], ms.inputs[1])
    L(em.outputs[0], ms.inputs[2])
    L(ms.outputs[0], out.inputs['Surface'])
    mat.use_backface_culling = True
    return mat


# ---------------------------------------------------------------- meshes ----
def fattr(me, name, values):
    a = me.color_attributes.new(name, 'FLOAT_COLOR', 'POINT')
    a.data.foreach_set('color', np.repeat(np.asarray(values, 'f4')[:, None], 4, 1).ravel())


def make_part(coll, d, part, m, arm):
    name = part['name']
    vf, uv, f, g, inside = d[name + '_vf'], d[name + '_uv'], d[name + '_f'], d[name + '_g'], d[name + '_in']
    n = len(vf)
    two = part['two_sided']
    if part.get('lid'):                           # lids rest open: up the slope of his brow, above the eye, out of sight
        vf = vf.copy()
        up = 1.05 * part['box'][3] * m['U'] * m['S']
        vf[:, 2] += up
        vf[:, 1] += part['slope'][1] * up
    ff = d[name + '_ff'] if name + '_ff' in d else f          # the front may have holes (his eye sockets)
    verts = np.concatenate([vf, d[name + '_vb']]) if two else vf
    faces = np.concatenate([ff, f[:, ::-1] + n]) if two else ff
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.shade_smooth()
    loops = np.zeros(len(me.loops), 'i4')
    me.loops.foreach_get('vertex_index', loops)
    uv_all = np.concatenate([uv, uv]) if two else uv
    me.uv_layers.new(name='UVMap').data.foreach_set('uv', uv_all[loops].ravel())
    fattr(me, 'shade', np.clip(np.concatenate([g, g]) * 1.6, 0, 1) if two else g)
    fattr(me, 'inside', np.concatenate([inside, inside]).astype('f4') if two else inside.astype('f4'))
    co = verts
    if m.get('back') and name in ('head', 'ear-left', 'ear-right'):
        b = m['back']
        u = (b['cx'] - co[:, 0] / (b['head_width'] / b['width'])) / b['size']
        v = 1 - (b['top'] + (m['head_top_z'] - co[:, 2]) / (b['head_height'] / (b['bottom'] - b['top']))) / b['size']
        me.uv_layers.new(name='back').data.foreach_set('uv', np.stack([u, v], 1)[loops].ravel())
    if part.get('side'):
        me.uv_layers.new(name='side').data.foreach_set('uv', np.concatenate([d[name + '_uvsf'], d[name + '_uvsb']])[loops].ravel())
        nor = np.zeros(len(me.vertices) * 3, 'f4')
        me.vertices.foreach_get('normal', nor)
        nor = nor.reshape(-1, 3)
        w = np.zeros(len(verts), 'f4')
        w[:n] = smooth(np.abs(nor[:n, 0]), 0.72, 0.93) * inside           # front: how much is side view
        w[n:] = smooth(nor[n:, 1], 0.25, 0.70)                             # back: how much is back view (the rest is side view)
        fattr(me, 'mixw', w)
    if two and not part.get('side'):               # how much of a rim each point is: steep to the drawing's plane
        nor = np.zeros(len(me.vertices) * 3, 'f4')
        me.vertices.foreach_get('normal', nor)
        w = np.zeros(len(verts), 'f4')
        w[:n] = smooth(1 - np.abs(nor.reshape(-1, 3)[:n, 1]), 0.60, 0.75) * inside
        fattr(me, 'rim', w)
    me.materials.append(material(name, part, 'front', m))
    if two:
        me.materials.append(material(name, part, 'back', m))
        mi = np.zeros(len(me.polygons), 'i4')
        mi[len(ff):] = 1
        me.polygons.foreach_set('material_index', mi)
    o = bpy.data.objects.new('j2_' + name, me)
    o['jimmy2'] = True
    coll.objects.link(o)
    bones = part.get('bones') or [name]
    if len(bones) == 1:
        o.vertex_groups.new(name=bones[0]).add(range(len(verts)), 1.0, 'REPLACE')
    else:
        wt = d[name + '_wt']
        wt = np.concatenate([wt, wt]) if two else wt
        for k, b in enumerate(bones):
            grp = o.vertex_groups.new(name=b)
            col = wt[:, k]
            for level in np.unique(np.round(col, 2)):
                if level > 0:
                    grp.add(np.nonzero(np.round(col, 2) == level)[0].tolist(), float(level), 'REPLACE')
    o.parent = arm
    mod = o.modifiers.new('rig', 'ARMATURE')
    mod.object = arm
    return o


# -------------------------------------------------------------- skeleton ----
def make_armature(coll, m):
    data = bpy.data.armatures.new(NAME)
    arm = bpy.data.objects.new(NAME, data)
    arm['jimmy2'] = True
    coll.objects.link(arm)
    arm.show_in_front = True
    data.display_type = 'STICK'
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    for name, r in m['rig'].items():
        e = data.edit_bones.new(name)
        e.head = room(m, r['pivot'][0], r['pivot'][1], EXTRA_DEPTH.get(name, 0.0))
        if name in EYEBALLS:                      # a pupil turns about the middle of its eyeball
            e.head = Vector(EYEBALLS[name]['centre'])
        e.tail = e.head + Vector((0, 0, 0.06))
        e.roll = 0
    for name, r in m['rig'].items():
        if r['parent'] in m['rig']:
            data.edit_bones[name].parent = data.edit_bones[r['parent']]
    bpy.ops.object.mode_set(mode='OBJECT')
    for n in SHOWN + ('arm-up',):
        arm[n] = 0.0
        arm.id_properties_ui(n).update(min=0.0, max=1.0)
    return arm


def drive(obj, prop, arm, expr, bone=None, channel=None, armprop=None):
    obj[prop] = 0.0
    fc = obj.driver_add('["%s"]' % prop)
    dr = fc.driver
    dr.type = 'SCRIPTED'
    v = dr.variables.new()
    v.name = 'v'
    if bone:
        v.type = 'TRANSFORMS'
        v.targets[0].id = arm
        v.targets[0].bone_target = bone
        v.targets[0].transform_type = channel
        v.targets[0].transform_space = 'LOCAL_SPACE'
    else:
        v.type = 'SINGLE_PROP'
        v.targets[0].id = arm
        v.targets[0].data_path = '["%s"]' % armprop
    dr.expression = expr


def local_axis(arm, bone, axis):
    """Which of a bone's own axes lies along a room axis, and which way."""
    v = arm.data.bones[bone].matrix_local.to_3x3().inverted() @ Vector(axis)
    i = max(range(3), key=lambda k: abs(v[k]))
    return 'XYZ'[i], (1 if v[i] > 0 else -1)


# ----------------------------------------------------------------- moves ----
def ease(t):
    return 0.5 - math.cos(math.pi * t) / 2


def bone_matrix(arm, m, name, v):
    """A bone's pose (its own space) for one flat-rig key."""
    r = m['rig'][name]
    U = m['U'] * m['S']
    R = arm.data.bones[name].matrix_local.to_3x3()
    inv = R.inverted()
    w, h = r['box'][2], r['box'][3]
    dx = v.get('dx', 0.0) + v.get('dxw', 0.0) * w
    dy = v.get('dy', 0.0) + v.get('dyh', 0.0) * h
    roll, yaw, pitch = v.get('r', 0.0), 0.0, 0.0
    sx, sy = v.get('sx', 0.0), v.get('sy', 0.0)
    if name in EYEBALLS:                         # a slide of the flat pupil is a turn of his eyeball here
        rad = EYEBALLS[name]['R']
        q = Matrix.Rotation(dx * U / rad, 3, 'Z') @ Matrix.Rotation(dy * U / rad, 3, 'X')
        return (inv @ q @ R).to_4x4()
    if name == 'head':                           # a slide of the flat head is a real turn or nod here
        yaw, pitch, dx, dy, sx = dx * YAW, dy * PITCH, 0.0, 0.0, 0.0
    elif name in ('eyes', 'closed-left', 'closed-right', 'brow-left', 'brow-right', 'mouth-talk', 'mouth-wide', 'hat'):
        dx = 0.0                                 # the flat rig slides these to fake a turn; his head really turns
        dy = v.get('dyh', 0.0) * h
    elif name in ('ear-left', 'ear-right', 'headset-left', 'headset-right'):
        dx, sx = 0.0, 0.0
    q = Matrix.Rotation(math.radians(yaw), 3, 'Z') @ Matrix.Rotation(math.radians(pitch), 3, 'X') @ Matrix.Rotation(math.radians(roll), 3, 'Y')
    out = (inv @ q @ R).to_4x4()
    a, b = SLOPES.get(name, (0.0, 0.0))            # parts that lie on his face slide along it, not through it
    out.translation = inv @ Vector((dx * U, a * dx * U + b * -dy * U, -dy * U))
    s = inv @ Matrix.Diagonal((1 + sx, 1, 1 + sy)) @ R
    return out @ Matrix.Diagonal((abs(s[0][0]), abs(s[1][1]), abs(s[2][2]), 1))


def value_at(track, t):
    """What a flat-rig track says at t (0 to 1), eased between its keys as the flat rig eases."""
    for (t0, v0), (t1, v1) in zip(track, track[1:]):
        if t0 <= t <= t1:
            k = ease((t - t0) / (t1 - t0)) if t1 > t0 else 0.0
            return {q: v0.get(q, 0.0) + (v1.get(q, 0.0) - v0.get(q, 0.0)) * k for q in set(v0) | set(v1)}
    return dict(track[-1][1])


def fcurves(action):
    out = []
    for layer in getattr(action, 'layers', []):
        for strip in layer.strips:
            for bag in getattr(strip, 'channelbags', []):
                out += list(bag.fcurves)
    return out or list(getattr(action, 'fcurves', []))


def write(arm, m, name, ms, tracks, loop=False):
    old = bpy.data.actions.get(name)
    if old:
        bpy.data.actions.remove(old)
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    if not arm.animation_data:
        arm.animation_data_create()
    arm.animation_data.action = act
    frames = max(1.0, ms / 1000 * FPS)
    if m.get('tall'):
        # His paw stays down unless the move uses it (Ant, 23 Nov 2026): then it comes up at the start and
        # goes back down at the end. Down, his arm hangs behind his body and his front leg shows instead.
        tracks = dict(tracks)
        uses = 'paw' in tracks or 'forearm' in tracks
        lift = min(0.3, ARM_MS / ms) if ms else 0.0
        up = (lambda t: 1.0) if uses and not ms else (lambda t: min(1.0, t / lift, (1 - t) / lift)) if uses else (lambda t: 0.0)
        paw = tracks.get('paw') or [[0, {}], [1, {}]]
        times = sorted({t for t, _ in paw} | ({0.0, lift, 1 - lift, 1.0} if uses and ms else {0.0, 1.0}))
        tracks['paw'] = [[t, {**value_at(paw, t), 'r': value_at(paw, t).get('r', 0.0) + (1 - up(t)) * ARM_DOWN}] for t in times]
        for t in times:
            arm['arm-up'] = float(up(t))
            arm.keyframe_insert('["arm-up"]', frame=1 + t * frames)
    for part, track in tracks.items():
        if part not in m['rig']:
            continue                              # the badge's own bubbles
        pb = arm.pose.bones[part]
        fades = part in SHOWN and any('o' in v for _, v in track)
        moves = any(k != 'o' for _, v in track for k in v)
        for t, v in track:
            f = 1 + t * frames
            if moves:
                loc, rot, sc = bone_matrix(arm, m, part, v).decompose()
                pb.location, pb.rotation_quaternion, pb.scale = loc, rot, sc
                for path in ('location', 'rotation_quaternion', 'scale'):
                    pb.keyframe_insert(path, frame=f, group=part)
            if fades:
                arm[part] = float(v.get('o', 0.0))
                arm.keyframe_insert('["%s"]' % part, frame=f)
    for fc in fcurves(act):
        for kp in fc.keyframe_points:
            kp.interpolation = 'SINE'
            kp.easing = 'EASE_IN_OUT'
        fc.update()
    act['ms'] = ms
    act['loop'] = loop
    act.use_frame_range = True
    act.frame_start, act.frame_end = 1, 1 + frames
    reset(arm)
    return act


def reset(arm):
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    for n in SHOWN + ('arm-up',):
        arm[n] = 0.0
    if 'hips' in arm.pose.bones:                  # his whole body: his paw rests down
        arm.pose.bones['paw'].matrix_basis = bone_matrix(arm, meta(), 'paw', {'r': ARM_DOWN})


def make_moves(arm, m):
    with open(os.path.join(HERE, 'moves.json')) as f:
        flat = json.load(f)
    made = []
    for name, a in flat['ACTIONS'].items():
        write(arm, m, name, a['ms'], a['tracks'])
        made.append(name)
    for name, parts in flat['POSES'].items():
        write(arm, m, 'pose-' + name, 0, {p: [[0, v], [1, v]] for p, v in parts.items()})
        made.append('pose-' + name)
    # Moves the flat rig can't do: a proper look round, and looking up.
    px = flat['PUPIL']['x']
    look = lambda k: {'dx': k * 4.5}
    eyes = lambda k: {'dx': k * px * 0.8}
    write(arm, m, 'look-round', 4200, {
        'head': [[0, {}], [0.16, look(1)], [0.40, look(1)], [0.60, look(-1)], [0.84, look(-1)], [1, {}]],
        'pupil-left': [[0, {}], [0.08, eyes(1)], [0.40, eyes(1)], [0.50, eyes(-1)], [0.84, eyes(-1)], [0.94, {}], [1, {}]],
        'pupil-right': [[0, {}], [0.08, eyes(1)], [0.40, eyes(1)], [0.50, eyes(-1)], [0.84, eyes(-1)], [0.94, {}], [1, {}]],
        'tail-tip': [[0, {}], [0.3, {'r': 8}], [0.7, {'r': -8}], [1, {}]],
    })
    write(arm, m, 'look-up', 1800, {'head': [[0, {}], [0.25, {'dy': -5}], [0.75, {'dy': -5}], [1, {}]],
                                     'pupil-left': [[0, {}], [0.15, {'dy': -4}], [0.75, {'dy': -4}], [1, {}]],
                                     'pupil-right': [[0, {}], [0.15, {'dy': -4}], [0.75, {'dy': -4}], [1, {}]]})
    made += ['look-round', 'look-up']
    arm.animation_data.action = None
    reset(arm)
    return made


# ----------------------------------------------------------------- build ----
def build():
    for cname in (NAME, 'JimmyFlat'):
        old = bpy.data.collections.get(cname)
        if old:
            for o in list(old.objects):
                bpy.data.objects.remove(o, do_unlink=True)
            bpy.data.collections.remove(old)
    for mat in list(bpy.data.materials):
        if mat.name.startswith(('J2_', 'F_')):
            bpy.data.materials.remove(mat)
    # The first, sculpted version shares action names with this one. It lives on in Jimmy.blend and its
    # own build scripts; take it out of this session so the two don't mix.
    for o in list(bpy.data.objects):
        if o.get('jimmy'):
            bpy.data.objects.remove(o, do_unlink=True)
    v1 = bpy.data.collections.get('Jimmy')
    if v1:
        bpy.data.collections.remove(v1)
    for act in list(bpy.data.actions):
        bpy.data.actions.remove(act)
    for img in list(bpy.data.images):
        if img.filepath and os.path.dirname(bpy.path.abspath(img.filepath)) == HERE:
            img.reload()
    coll = bpy.data.collections.new(NAME)
    bpy.context.scene.collection.children.link(coll)
    m = meta()
    SLOPES.clear()
    SLOPES.update({p['name']: p['slope'] for p in m['parts'] if p.get('slope') and not p['two_sided']})
    EYEBALLS.clear()
    EYEBALLS.update({p['name']: p['eyeball'] for p in m['parts'] if p.get('eyeball')})
    d = np.load(os.path.join(HERE, 'parts.npz'))
    arm = make_armature(coll, m)
    objs = {p['name']: make_part(coll, d, p, m, arm) for p in m['parts']}
    frame = m['SRC'] * m['S']
    for p in m['parts']:
        o = objs[p['name']]
        if p['name'] in SHOWN:
            drive(o, 'show', arm, 'v', armprop=p['name'])
        elif p['name'] == 'paw' and m.get('tall'):
            drive(o, 'show', arm, 'min(1, v*3)', armprop='arm-up')
        elif p['name'] == 'leg':
            drive(o, 'show', arm, 'max(0, 1-v*3)', armprop='arm-up')
        elif p['name'] == 'paw':
            o['show'] = 1.0
        if p.get('clip'):
            ax, sx = local_axis(arm, p['name'], (1, 0, 0))
            az, sz = local_axis(arm, p['name'], (0, 0, 1))
            rest_up = 1.05 * p['box'][3] * m['U'] * m['S'] if p.get('lid') else 0.0
            drive(o, 'ox', arm, '%d*v/%f' % (sx, frame), bone=p['name'], channel='LOC_' + ax)
            drive(o, 'oz', arm, '(%f+%d*v)/%f' % (rest_up, sz, frame), bone=p['name'], channel='LOC_' + az)
    made = make_moves(arm, m)
    for name in made:
        bpy.data.actions[name]['j2'] = True
    empty = [i.name for i in bpy.data.images if i.filepath and os.path.dirname(bpy.path.abspath(i.filepath)) == HERE and (i.size[0] == 0 or not i.has_data and i.size[0] == 0)]
    return {'parts': len(objs), 'bones': len(arm.data.bones), 'actions': len(made), 'points': int(sum(len(o.data.vertices) for o in objs.values())), 'empty_pictures': empty}


def show(action=None, at=0.5):
    arm = bpy.data.objects[NAME]
    reset(arm)
    if action:
        act = bpy.data.actions[action]
        arm.animation_data.action = act
        f = act.frame_start + (act.frame_end - act.frame_start) * at
        bpy.context.scene.frame_set(int(f), subframe=f - int(f))
    else:
        arm.animation_data.action = None
        bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()


def camera(yaw=0.0, pitch=0.0, size=1024, scale=None, whole=False):
    """Look at him from yaw degrees round (0 = straight on, as the drawing; 180 = from behind)."""
    sc = bpy.context.scene
    for cname, hide in (('Jimmy', True), (NAME, False)):
        c = bpy.data.collections.get(cname)
        if c:
            c.hide_render = hide
    cam = sc.camera
    if cam is None:
        cam = bpy.data.objects.new('JimmyCam', bpy.data.cameras.new('JimmyCam'))
        sc.collection.objects.link(cam)
        sc.camera = cam
    m = meta()
    frame = m['SRC'] * m['S']
    sc.render.engine = 'BLENDER_EEVEE'
    tall = m.get('tall', m['SRC']) * m['S'] if whole else frame      # whole: all of him, down to the ground
    sc.render.resolution_x, sc.render.resolution_y = int(round(size * frame / tall)), size
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.view_settings.view_transform = 'Standard'
    try:
        sc.eevee.taa_render_samples = 32
    except Exception:
        pass
    a, p = math.radians(yaw), math.radians(pitch)
    dvec = Vector((math.sin(a) * math.cos(p), math.cos(a) * math.cos(p), -math.sin(p)))
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = scale or (tall if yaw == 0 and pitch == 0 else tall * 1.15)
    cam.data.clip_end = 100
    cam.rotation_euler = dvec.to_track_quat('-Z', 'Y').to_euler()
    cam.location = Vector((0, 0, frame - tall / 2)) - dvec * 30


def shot(path, yaw=0.0, pitch=0.0, action=None, at=0.5, size=1024, whole=False):
    show(action, at)
    camera(yaw, pitch, size, whole=whole)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


def clip(folder, action, yaw=0.0, size=512):
    """Render one action as PNG frames with a clear background: <folder>/<action>/0001.png ..."""
    act = bpy.data.actions[action]
    show(action, 0)
    camera(yaw, 0, size)
    out = os.path.join(folder, action)
    os.makedirs(out, exist_ok=True)
    sc = bpy.context.scene
    last = int(round(act.frame_end))
    for f in range(1, last + 1):
        sc.frame_set(f)
        sc.render.filepath = os.path.join(out, '%04d.png' % f)
        bpy.ops.render.render(write_still=True)
    return last
