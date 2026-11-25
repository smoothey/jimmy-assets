"""3D Jimmy: his moves (JM-491). Runs inside Blender.

Two kinds:
- Every action and pose of the flat rig (jimmy-rig.js), carried over by rule
  from moves.json (made by dump-moves.cjs). Change a move in jimmy-rig.js,
  dump again, build again, and the 3D move follows.
- Moves only a body can do (standing, walking, stretching...), written here.

Each becomes a Blender action. The flat rig's moves are all drawn on top of
his badge pose (sitting, left paw up), so each of those actions holds that
pose too and plays right by itself. Names: the flat rig's own ('wave'), with
'pose-' in front for the ones that are held ('pose-think').
"""
import json
import math
import os

import bpy
from mathutils import Matrix, Quaternion, Vector

import j3d_model as mdl

V = Vector
FPS = 30
UNIT = 0.58 / 162          # one unit of the flat rig's frame, in Blender units
SHUT_AT = 20               # degrees below the middle of the eye where shut lids meet
SMILE_AT = 24              # degrees above it where they meet for his happy eyes
HERE = os.path.dirname(os.path.abspath(__file__))


def ease(t):
    return 0.5 - math.cos(math.pi * t) / 2


# ------------------------------------------------------------- posing ----
class Poser:
    """Works out poses from the skeleton alone (no need to wait for Blender to redraw)."""

    def __init__(self, arm, basis=None):
        self.rest = {b.name: b.matrix_local.copy() for b in arm.data.bones}
        self.parent = {b.name: (b.parent.name if b.parent else None) for b in arm.data.bones}
        self.order = [b.name for b in arm.data.bones]
        self.basis = {n: Matrix.Identity(4) for n in self.order}
        if basis:
            for n, m in basis.items():
                self.basis[n] = m.copy()

    def copy(self):
        p = Poser.__new__(Poser)
        p.rest, p.parent, p.order = self.rest, self.parent, self.order
        p.basis = {n: m.copy() for n, m in self.basis.items()}
        return p

    def pre(self, name):
        """Where the bone would be with its own pose taken off."""
        p = self.parent[name]
        if p is None:
            return self.rest[name].copy()
        return self.pose(p) @ self.rest[p].inverted() @ self.rest[name]

    def pose(self, name):
        return self.pre(name) @ self.basis[name]

    def set_pose(self, name, m):
        self.basis[name] = self.pre(name).inverted() @ m

    def turn(self, name, axis, degrees):
        """Turn about an axis of the room: (1,0,0) pitches, (0,0,1) turns, (0,1,0) rolls."""
        m = self.pose(name)
        loc = m.translation.copy()
        n = Matrix.Rotation(math.radians(degrees), 4, V(axis)) @ m
        n.translation = loc
        self.set_pose(name, n)
        return self

    def twist(self, name, degrees):
        self.basis[name] = self.basis[name] @ Matrix.Rotation(math.radians(degrees), 4, 'Y')
        return self

    def aim(self, name, direction):
        m = self.pose(name)
        loc = m.translation.copy()
        y = m.to_3x3().col[1].normalized()
        n = y.rotation_difference(V(direction).normalized()).to_matrix().to_4x4() @ m
        n.translation = loc
        self.set_pose(name, n)
        return self

    def shift(self, name, vec):
        m = self.pose(name)
        m.translation = m.translation + V(vec)
        self.set_pose(name, m)
        return self

    def face(self, name, was, to):
        """Twist a bone about its own length until the side that faced `was` (at rest) faces `to`."""
        a = self.rest[name].to_3x3().inverted() @ V(was)
        m = self.pose(name).to_3x3()
        y = m.col[1].normalized()
        w = m @ a
        w = (w - y * w.dot(y)).normalized()
        t = V(to) - y * V(to).dot(y)
        if t.length < 1e-6:
            return self
        t.normalize()
        ang = math.atan2(y.dot(w.cross(t)), w.dot(t))
        self.basis[name] = self.basis[name] @ Matrix.Rotation(ang, 4, 'Y')
        return self

    def stretch(self, name, x=1.0, y=1.0, z=1.0):
        self.basis[name] = self.basis[name] @ Matrix.Diagonal((x, y, z, 1))
        return self

    # Plain words for the usual turns. Up, his left and clockwise-from-the-front are plus.
    def pitch(self, name, d):
        return self.turn(name, (1, 0, 0), -d)

    def yaw(self, name, d):
        return self.turn(name, (0, 0, 1), d)

    def roll(self, name, d):
        return self.turn(name, (0, 1, 0), d)


def badge_pose(arm):
    """Sitting with his left paw up, pads to the front, as the drawing has him."""
    p = Poser(arm)
    p.aim('upper_arm.L', (0.86, -0.34, -0.12))
    p.face('upper_arm.L', (0, 1, 0), (0, -1, 0.2))
    p.aim('forearm.L', (0.50, -0.20, 0.84))
    p.face('forearm.L', (0, 1, 0), (0, -1, 0))
    p.aim('paw.L', (0.30, 0.02, 0.95))
    p.face('paw.L', (0, 0, -1), (0, -1, 0))
    return p


def stand_pose(arm):
    st = arm.get('stand')
    basis = {}
    for n in arm.data.bones.keys():
        if st and n in st:
            f = list(st[n])
            basis[n] = Matrix([f[0:4], f[4:8], f[8:12], f[12:16]])
    return Poser(arm, basis)


# -------------------------------------------- the flat rig's moves, by rule ----
# Flat part -> [(bone, what, how much per unit)]. The flat rig's "left" is the
# left of the screen, which is his right (.R).
def _side(flat):
    return 'R' if flat.endswith('left') else 'L'


def rules(part, prop):
    s = _side(part)
    if part.startswith('lid-') and prop == 'dyh':
        return [('lid_up.' + s, 'close', 1 / 1.05), ('lid_lo.' + s, 'rise', 1 / 1.05)]
    if part.startswith('closed-') and prop == 'o':
        return [('lid_lo.' + s, 'smile', 1.0), ('lid_up.' + s, 'drop', 1.0), ('brow.' + s, 'dz', 0.004)]
    if part.startswith('brow-'):
        return {'dyh': [('brow.' + s, 'dz', -0.042)], 'r': [('brow.' + s, 'roll', 1.0)]}.get(prop, [])
    if part.startswith('ear-'):
        return {'r': [('ear.' + s, 'roll', 1.0)], 'sy': [('ear.' + s, 'len', 1.6)]}.get(prop, [])
    if part.startswith('pupil-'):
        return {'dx': [('eye.' + s, 'yaw', 24 / 8.5)], 'dy': [('eye.' + s, 'pitch', -18 / 6.5)]}.get(prop, [])
    if part == 'tail' and prop == 'r':
        return [('tail.1', 'roll', 0.6), ('tail.2', 'roll', 0.6)]
    if part == 'tail-mid' and prop == 'r':
        return [('tail.3', 'roll', 1.2)]
    if part == 'tail-tip' and prop == 'r':
        return [('tail.4', 'roll', 0.7), ('tail.5', 'roll', 0.7)]
    if part == 'paw' and prop == 'r':
        return [('upper_arm.L', 'roll', 1.0)]
    if part == 'forearm' and prop == 'r':
        return [('forearm.L', 'roll', 1.0)]
    if part == 'body':
        return {'sy': [('root', 'squash', 1.0)], 'dy': [('root', 'dz', -UNIT * 1.6)], 'r': [('root', 'roll', 1.0)]}.get(prop, [])
    if part == 'head':
        return {'r': [('head', 'roll', 0.7), ('neck', 'roll', 0.4)],
                'dy': [('head', 'pitch', -3.8), ('neck', 'pitch', -1.8)],
                'dx': [('head', 'yaw', 5.5), ('neck', 'yaw', 2.6)]}.get(prop, [])
    if part == 'tag' and prop == 'r':
        return [('tag', 'roll', 1.0)]
    if part == 'mouth-talk' and prop == 'o':
        return [('mouth', 'open', 0.62)]
    if part == 'mouth-wide' and prop == 'o':
        return [('mouth', 'open', 1.0)]
    if part == 'headset-left' and prop == 'dy':
        return [('mic', 'pitch', -1.6)]
    if part == 'hat' and prop == 'o':
        return [('*', 'hat', 1.0)]
    return []       # the flat rig's sideways slides that fake a turn, and the badge's own bubbles


def sample(track, t):
    """The flat rig's own reading of a track at time t (0 to 1)."""
    i = 0
    while i < len(track) - 2 and t > track[i + 1][0]:
        i += 1
    t0, a = track[i]
    t1, b = track[min(i + 1, len(track) - 1)]
    f = ease(min(1, max(0, (t - t0) / (t1 - t0)))) if t1 > t0 else 1
    keys = set(a) | set(b)
    return {k: a.get(k, 0) + (b.get(k, 0) - a.get(k, 0)) * f for k in keys}


def convert(tracks):
    """Flat tracks -> {bone: {time: {what: amount}}}, with a key wherever a part that drives the bone has one."""
    drives = {}        # bone -> [(part track, prop, what, factor)]
    for part, track in tracks.items():
        props = set()
        for _, v in track:
            props |= set(v)
        for prop in props:
            for bone, what, k in rules(part, prop):
                drives.setdefault(bone, []).append((track, prop, what, k))
    out = {}
    for bone, ds in drives.items():
        times = sorted({t for track, _, _, _ in ds for t, _ in track})
        out[bone] = {}
        for t in times:
            vals = {}
            for track, prop, what, k in ds:
                vals[what] = vals.get(what, 0.0) + sample(track, t).get(prop, 0.0) * k
            out[bone][t] = vals
    return out


def apply_values(base, bone, vals):
    """A bone's pose (4x4, its own space) for some amounts, on top of the base pose."""
    pre = base.pre(bone).to_3x3()
    inv = pre.inverted()
    b = base.basis[bone]
    pitch = vals.get('pitch', 0.0)
    yaw = vals.get('yaw', 0.0)
    roll = vals.get('roll', 0.0)
    loc = V((vals.get('dx', 0.0), vals.get('dy', 0.0), vals.get('dz', 0.0)))
    scale = V((1, 1, 1))
    # Lids. Shut, they meet in a line a little below the middle of the eye (a
    # sleeping curve); for his happy eyes they meet above it (an arch).
    if 'close' in vals:
        pitch -= (mdl.OPEN + SHUT_AT) * vals['close']
    if 'rise' in vals:
        pitch += (mdl.OPEN - SHUT_AT + 2) * vals['rise']
    if 'smile' in vals:
        pitch += (mdl.OPEN + SMILE_AT) * vals['smile']
    if 'drop' in vals:
        pitch -= (mdl.OPEN - SMILE_AT + 2) * vals['drop']
    if 'open' in vals:                       # his mouth coming forward out of his head
        o = vals['open']
        loc += V((0, -(mdl.MOUTH_IN + 0.006) * min(1.0, o * 1.6), -0.012 * o))
        s = 0.55 + 0.45 * o
        scale = V((s, 1, s))
    if 'len' in vals:
        scale = V((1, 1 + vals['len'], 1))
    q = Matrix.Rotation(math.radians(yaw), 3, 'Z') @ Matrix.Rotation(math.radians(-pitch), 3, 'X') @ Matrix.Rotation(math.radians(roll), 3, 'Y')
    rot = (inv @ q @ pre) @ b.to_3x3()
    m = rot.to_4x4()
    m.translation = b.translation + inv @ loc
    if 'squash' in vals:                     # root lies along -Y with Z up: squash his height, spread his width
        sq = vals['squash']
        scale = V((1 - sq * 0.5, 1 - sq * 0.5, 1 + sq))
    return m @ Matrix.Diagonal((*scale, 1))


# ------------------------------------------------------- writing actions ----
def fcurves(action):
    out = []
    if hasattr(action, 'layers'):
        for layer in action.layers:
            for strip in layer.strips:
                for bag in getattr(strip, 'channelbags', []):
                    out += list(bag.fcurves)
    if not out and hasattr(action, 'fcurves'):
        out = list(action.fcurves)
    return out


def write(arm, name, ms, keys, held=None, hat=None, loop=False):
    """keys: {bone: {time 0..1: 4x4 pose in the bone's own space}}; held: {bone: 4x4} keyed once at the start."""
    old = bpy.data.actions.get(name)
    if old:
        bpy.data.actions.remove(old)
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    if not arm.animation_data:
        arm.animation_data_create()
    arm.animation_data.action = act
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    frames = max(1.0, ms / 1000 * FPS)

    def key(bone, frame, m, what=('location', 'rotation_quaternion', 'scale')):
        pb = arm.pose.bones[bone]
        loc, rot, sc = m.decompose()
        pb.location, pb.rotation_quaternion, pb.scale = loc, rot, sc
        for path in what:
            pb.keyframe_insert(path, frame=frame, group=bone)

    for bone, m in (held or {}).items():
        if bone not in keys:
            key(bone, 1, m, ('rotation_quaternion',))
    for bone, ks in keys.items():
        prev = None
        for t in sorted(ks):
            m = ks[t]
            q = m.to_quaternion()
            if prev is not None and prev.dot(q) < 0:      # keep to the short way round
                loc, _, sc = m.decompose()
                q.negate()
                pb = arm.pose.bones[bone]
                pb.location, pb.rotation_quaternion, pb.scale = loc, q, sc
                for path in ('location', 'rotation_quaternion', 'scale'):
                    pb.keyframe_insert(path, frame=1 + t * frames, group=bone)
            else:
                key(bone, 1 + t * frames, m)
            prev = q
    if hat is not None:
        for t, v in hat:
            arm['hat'] = float(v)
            arm.keyframe_insert('["hat"]', frame=1 + t * frames)
        arm['hat'] = 0.0
    for fc in fcurves(act):
        for kp in fc.keyframe_points:
            kp.interpolation = 'SINE'
            kp.easing = 'EASE_IN_OUT'
        fc.update()
    act['ms'] = ms
    act['loop'] = loop
    act.frame_start, act.frame_end = 1, 1 + frames
    act.use_frame_range = True
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    return act


def from_flat(arm, base, name, ms, tracks, pose=False):
    conv = convert(tracks)
    hat = None
    if '*' in conv:
        hat = [(t, v.get('hat', 0)) for t, v in sorted(conv.pop('*').items())]
    keys = {bone: {t: apply_values(base, bone, vals) for t, vals in ks.items()} for bone, ks in conv.items()}
    held = {b: base.basis[b] for b in ('upper_arm.L', 'forearm.L', 'paw.L')}
    return write(arm, name, ms, keys, held=held, hat=hat)


# ------------------------------------------------ moves only a body can do ----
def cycle(n):
    return [k / n for k in range(n + 1)]


def body_moves(arm):
    """name: (ms, loop, {bone: {t: Poser}}) built from whole-body poses."""
    out = {}
    stand = stand_pose(arm)
    sitp = Poser(arm)
    out['pose-sit'] = (0, False, [(0, sitp)])
    out['pose-stand'] = (0, False, [(0, stand)])
    out['stand-up'] = (900, False, [(0, sitp), (1, stand)])
    out['sit-down'] = (900, False, [(0, stand), (1, sitp)])

    # Walking on the spot: a trot, opposite corners together.
    def step(ph):
        p = stand.copy()
        for side, off in (('L', 0.0), ('R', 0.5)):
            for front, o2 in ((True, 0.0), (False, 0.5)):
                a = 2 * math.pi * (ph + off + o2)
                swing = math.sin(a)
                lift = max(0.0, math.cos(a))
                if front:
                    p.pitch('upper_arm.' + side, 24 * swing)
                    p.pitch('forearm.' + side, -30 * lift)
                    p.pitch('paw.' + side, -30 * lift + 8 * swing)
                else:
                    p.pitch('thigh.' + side, 20 * swing)
                    p.pitch('shin.' + side, 26 * lift)
                    p.pitch('hock.' + side, -30 * lift)
                    p.pitch('foot.' + side, -20 * lift)
        b = math.sin(4 * math.pi * ph)
        p.shift('hips', (0, 0, 0.012 * b))
        p.roll('hips', 3 * math.sin(2 * math.pi * ph))
        p.pitch('head', -2.5 * b)
        p.yaw('spine', 3 * math.sin(2 * math.pi * ph))
        for i, k in enumerate((4, 6, 8, 9, 9)):
            p.turn('tail.%d' % (i + 1), (0, 0, 1), k * math.sin(2 * math.pi * ph - i * 0.7))
        return p
    out['walk'] = (800, True, [(t, step(t)) for t in cycle(8)])

    # A long stretch: front down and forward, tail up, then back.
    def stretch(k):
        p = stand.copy()
        p.shift('hips', (0, 0.05 * k, 0.02 * k))
        p.pitch('hips', -14 * k)
        p.pitch('chest', -16 * k)
        for s in ('L', 'R'):
            p.pitch('upper_arm.' + s, 58 * k)
            p.pitch('forearm.' + s, 10 * k)
            p.pitch('paw.' + s, -40 * k)
            p.pitch('thigh.' + s, 12 * k)
            p.pitch('shin.' + s, -8 * k)
        p.pitch('neck', 22 * k)
        p.pitch('head', 22 * k)
        for i in range(5):
            p.pitch('tail.%d' % (i + 1), 9 * k)
        return p
    out['stretch'] = (2600, False, [(0, stand), (0.3, stretch(1)), (0.7, stretch(1)), (1, stand)])

    # A real look round: head and neck turn, eyes lead.
    def look(yaw, pitch=0):
        p = badge_pose(arm)
        p.yaw('neck', yaw * 0.4).yaw('head', yaw * 0.6)
        p.pitch('head', pitch)
        return p
    out['look-round'] = (3600, False, [(0, look(0)), (0.18, look(48, 4)), (0.42, look(48, 4)), (0.62, look(-48, 4)), (0.84, look(-48, 4)), (1, look(0))])
    out['look-up'] = (1800, False, [(0, look(0)), (0.25, look(0, 30)), (0.75, look(0, 30)), (1, look(0))])
    out['pose-badge'] = (0, False, [(0, badge_pose(arm))])

    # Kneading the ground with his front paws, sitting.
    def knead(ph):
        p = sitp.copy()
        for s, off in (('L', 0.0), ('R', 0.5)):
            lift = max(0.0, math.sin(2 * math.pi * (ph + off)))
            p.pitch('upper_arm.' + s, 14 * lift)
            p.pitch('forearm.' + s, -22 * lift)
            p.pitch('paw.' + s, 18 * lift)
        p.roll('chest', 2.5 * math.sin(2 * math.pi * ph))
        return p
    out['knead'] = (900, True, [(t, knead(t)) for t in cycle(8)])
    return out


def from_posers(arm, name, ms, loop, frames):
    ident = Matrix.Identity(4)
    keys = {}
    moved = set()
    for _, p in frames:
        for n, m in p.basis.items():
            if any(abs(m[i][j] - ident[i][j]) > 1e-5 for i in range(4) for j in range(4)):
                moved.add(n)
    moved = moved or {'root'}
    for t, p in frames:
        for n in moved:
            keys.setdefault(n, {})[t] = p.basis[n]
    return write(arm, name, ms, keys, loop=loop)


def build():
    arm = bpy.data.objects['Jimmy']
    for a in list(bpy.data.actions):
        bpy.data.actions.remove(a)
    with open(os.path.join(HERE, 'moves.json')) as f:
        flat = json.load(f)
    base = badge_pose(arm)
    made = []
    for name, a in flat['ACTIONS'].items():
        from_flat(arm, base, name, a['ms'], a['tracks'])
        made.append(name)
    for name, parts in flat['POSES'].items():
        tracks = {part: [[0, v], [1, v]] for part, v in parts.items()}
        from_flat(arm, base, 'pose-' + name, 0, tracks)
        made.append('pose-' + name)
    for name, (ms, loop, frames) in body_moves(arm).items():
        from_posers(arm, name, ms, loop, frames)
        made.append(name)
    arm.animation_data.action = bpy.data.actions.get('pose-badge')
    bpy.context.scene.frame_set(1)
    return {'actions': len(made), 'names': made}


def show(name, frame=None):
    """Put him in an action at a frame (the middle of it if no frame is given)."""
    arm = bpy.data.objects['Jimmy']
    act = bpy.data.actions[name]
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    arm['hat'] = 0.0
    arm.animation_data.action = act
    f = frame if frame is not None else (act.frame_start + act.frame_end) / 2
    bpy.context.scene.frame_set(int(f), subframe=f - int(f))
    bpy.context.view_layer.update()
    return f
