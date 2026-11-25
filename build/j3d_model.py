"""3D Jimmy: the model (JM-491). Runs inside Blender.

He is built standing on four legs (the pose that skins best); the rig then
sits him down the way the drawing has him, and that becomes his rest pose.
Each part is its own object in one colour, so the drawn line can go round it.
"""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

import j3d_mats
from j3d_common import M, PAL, Shape, collection, frame, rigid

V = Vector

# ---- Where things are, standing (his left is +X; the right side is mirrored) ----
HIP = V((0, 0.22, 0.50))
SPINE = V((0, 0.06, 0.505))
CHEST = V((0, -0.10, 0.51))
NECK = V((0, -0.24, 0.60))
HEADJ = V((0, -0.35, 0.77))
HEAD_END = V((0, -0.35, 1.05))
HC = V((0, -0.40, 0.905))             # the middle of his head
HR = V((0.29, 0.25, 0.235))          # his head's half sizes

SH = V((0.115, -0.19, 0.46))         # front leg: shoulder, elbow, wrist, toes
EL = V((0.118, -0.15, 0.26))
WR = V((0.118, -0.18, 0.075))
TOE = V((0.118, -0.29, 0.035))
HP = V((0.12, 0.22, 0.44))           # hind leg: hip, knee, hock, ball, toes
KN = V((0.13, 0.10, 0.27))
HK = V((0.125, 0.26, 0.15))
BALL = V((0.125, 0.20, 0.045))
TOEH = V((0.125, 0.09, 0.035))
TAIL = [V(p) for p in ((0, 0.37, 0.56), (0, 0.50, 0.67), (0, 0.61, 0.82), (0, 0.65, 0.99), (0, 0.61, 1.15), (0, 0.52, 1.27))]
TAIL_R = [0.052, 0.056, 0.061, 0.066, 0.064, 0.052]

EAR_BASE = V((0.205, 0.03, 0.14))    # from the middle of his head
EAR_TIP = V((0.360, 0.05, 0.505))
EYE = V((0.128, -0.150, -0.005))     # from the middle of his head (how deep is set from the skin when he is built)
EYE_R = 0.118
EYE_SHOW = 0.0905                    # half the width of eye that shows
OPEN = 50                            # degrees a lid is turned back when open
MOUTH_IN = 0.062                     # how far inside his head his mouth waits
MIC_FROM = V((-0.312, -0.060, -0.085))
MIC_TO = V((-0.172, -0.283, -0.222))
NOSE_Z = -0.125                      # heights on his face, from the middle of his head
MOUTH_Z = -0.212
BROW_Z = 0.150

# How the sitting pose turns his chest (used to lay his chest fur the right way).
SIT_PITCH = math.radians(65)
SIT_F = V((0, -math.cos(SIT_PITCH), -math.sin(SIT_PITCH)))   # what will face forward once he sits
SIT_D = V((0, math.sin(SIT_PITCH), -math.cos(SIT_PITCH)))    # what will point down


def mx(v):
    return V((-v.x, v.y, v.z))


def add(name, mesh_or_shape, mat, bone=None, voxel=None, smooth=0):
    if isinstance(mesh_or_shape, Shape):
        obj = mesh_or_shape.to_object(name, voxel, smooth, mat)
    else:
        obj = bpy.data.objects.new(name, mesh_or_shape)
        obj['jimmy'] = True
        collection().objects.link(obj)
        obj.data.shade_smooth()
        obj.data.materials.append(mat)
    if bone:
        rigid(obj, bone)
    return obj


def sphere_part(c, r, keep, rot=None, seg=48):
    """Part of a ball: keep(v) picks the points to keep, v on a ball of size 1 before it is turned."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2, radius=1)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not keep(v.co)], context='VERTS')
    bmesh.ops.transform(bm, matrix=M(c, rot, (r, r, r)), verts=bm.verts[:])
    me = bpy.data.meshes.new('part')
    bm.to_mesh(me)
    bm.free()
    return me


def cap(c, r, ax, az, cx=0.0, cz=0.0, rings=6, seg=48, inner=0.0):
    """An oval patch on the front of a ball (radius r at c): half sizes ax, az and middle cx, cz, as parts of r.
    With inner (0 to 1) it is a ring: the middle is left open."""
    bm = bmesh.new()

    def pt(x, z):
        return V(c) + V((x, -math.sqrt(max(0.0, 1 - x * x - z * z)), z)) * r
    mid = None if inner else bm.verts.new(pt(cx, cz))
    prev = None
    for k in range(0 if inner else 1, rings + 1):
        f = inner + (1 - inner) * k / rings
        ring = [bm.verts.new(pt(cx + ax * f * math.cos(a), cz + az * f * math.sin(a))) for a in (2 * math.pi * j / seg for j in range(seg))]
        for j in range(seg):
            if prev is None:
                if mid is not None:
                    bm.faces.new((mid, ring[j], ring[(j + 1) % seg]))
            else:
                bm.faces.new((prev[j], ring[j], ring[(j + 1) % seg], prev[(j + 1) % seg]))
        prev = ring
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm_mesh(bm)


def tube(pts, radii, seg=10, caps=True):
    """A round tube through pts (a whisker, a brow, a boom)."""
    bm = bmesh.new()
    rings = []
    pts = [V(p) for p in pts]
    for i, p in enumerate(pts):
        d = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
        f = frame(d).to_3x3()
        r = radii[i] if isinstance(radii, (list, tuple)) else radii
        rings.append([bm.verts.new(p + f @ V((math.cos(a) * r, math.sin(a) * r, 0))) for a in (2 * math.pi * k / seg for k in range(seg))])
    for a, b in zip(rings, rings[1:]):
        for k in range(seg):
            bm.faces.new((a[k], a[(k + 1) % seg], b[(k + 1) % seg], b[k]))
    if caps:
        bm.faces.new(list(reversed(rings[0])))
        bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    me = bpy.data.meshes.new('tube')
    bm.to_mesh(me)
    bm.free()
    return me


def join_meshes(meshes, name):
    bm = bmesh.new()
    for me in meshes:
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
    out = bpy.data.meshes.new(name)
    bm.to_mesh(out)
    bm.free()
    return out


def lathe(profile, centre, axis, seg=40):
    """Spin a profile [(radius, height), ...] round an axis. Returns a bmesh."""
    bm = bmesh.new()
    f = frame(axis).to_3x3()
    rings = []
    for r, h in profile:
        rings.append([bm.verts.new(V(centre) + f @ V((math.cos(a) * r, math.sin(a) * r, h))) for a in (2 * math.pi * k / seg for k in range(seg))])
    for a, b in zip(rings, rings[1:]):
        for k in range(seg):
            bm.faces.new((a[k], a[(k + 1) % seg], b[(k + 1) % seg], b[k]))
    bm.faces.new(list(reversed(rings[0])))
    bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def bm_mesh(bm, name='m'):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def surface(obj, x, z, lift=0.0):
    """The point on obj's front at (x, z) from the middle of his head, lifted off the skin a little."""
    o = V((HC.x + x, HC.y - 2.0, HC.z + z))
    hit, loc, normal, _ = obj.ray_cast(o, V((0, 1, 0)))
    if not hit:
        return V((HC.x + x, HC.y - HR.y, HC.z + z))
    return loc + normal * lift


# ---------------------------------------------------------------- body ----
def build_body(mats):
    s = Shape()
    # One side's legs, then mirrored.
    s.ball(SH, (0.092, 0.10, 0.11))
    s.limb(SH, EL, 0.078, 0.062)
    s.limb(EL, WR, 0.062, 0.054)
    s.ball(V((0.118, -0.225, 0.050)), (0.076, 0.090, 0.050))                     # front paw
    for dx, dy in ((-0.047, 0.014), (-0.016, 0.0), (0.016, 0.0), (0.047, 0.014)):
        s.ball(V((0.118 + dx, -0.296 + dy, 0.038)), (0.024, 0.028, 0.030), seg=16)   # toes
    s.ball(V((0.115, 0.19, 0.41)), (0.10, 0.15, 0.15))                           # haunch
    s.limb(HP, KN, 0.10, 0.072)
    s.limb(KN, HK, 0.072, 0.048)
    s.limb(HK, BALL, 0.048, 0.044)
    s.ball(V((0.125, 0.145, 0.043)), (0.060, 0.085, 0.042))                      # hind paw
    for dx, dy in ((-0.037, 0.012), (-0.013, 0.0), (0.013, 0.0), (0.037, 0.012)):
        s.ball(V((0.125 + dx, 0.085 + dy, 0.034)), (0.020, 0.025, 0.026), seg=16)
    # Fur down the sides of his chest (it shows as the ragged edge of his body in the drawing).
    X = V((1, 0, 0))
    cc = V((0, -0.13, 0.51))
    for row, (off, spread) in enumerate(((0.02, (28, 52, 76)), (0.085, (20, 44, 68)), (0.15, (30, 56)))):
        for a in spread:
            ar = math.radians(a)
            n = SIT_F * math.cos(ar) + X * math.sin(ar)
            base = cc + n * 0.150 + SIT_D * (off - 0.03)
            s.tuft(base, base + SIT_D * (0.085 + 0.01 * row) + n * 0.040, 0.040, thin=n)
    s.mirror()
    # The middle: trunk, neck, tail, and the bib of fur on his chest.
    s.ball(V((0, 0.20, 0.50)), (0.168, 0.19, 0.175))
    s.ball(V((0, 0.04, 0.497)), (0.163, 0.20, 0.165))
    s.ball(cc, (0.168, 0.18, 0.18))
    s.limb(NECK - V((0, -0.03, 0.03)), HEADJ + V((0, -0.01, 0.02)), 0.118, 0.100)
    s.chain(TAIL, TAIL_R)
    tip, prev = TAIL[-1], TAIL[-2]
    td = (tip - prev).normalized()
    for side in (V((1, 0, 0)), V((-1, 0, 0)), V((0, 0.5, 0.9)), V((0, -0.6, -0.5))):
        s.tuft(tip - td * 0.01 + side * 0.012, tip + td * 0.075 + side * 0.022, 0.03)
    for off, xs in ((0.0, (-0.045, 0.045)), (0.055, (-0.08, 0.0, 0.08)), (0.12, (-0.04, 0.045)), (0.18, (0.0,))):
        for x in xs:
            n = (SIT_F + X * (x / 0.17)).normalized()
            base = cc + n * 0.152 + SIT_D * (off - 0.04)
            s.tuft(base, base + SIT_D * 0.105 + n * 0.055, 0.048, thin=n)
    body = add('body', s, mats['fur'], voxel=0.0095, smooth=4)
    return body


# ---------------------------------------------------------------- head ----
def build_head(mats):
    s = Shape()
    # One side, then mirrored: cheek fur, the tuft by each ear, the whisker pad.
    for z, ln, a, r in ((-0.045, 0.060, 4, 0.040), (-0.105, 0.072, -14, 0.044), (-0.165, 0.066, -34, 0.042), (-0.212, 0.048, -58, 0.034)):
        ar = math.radians(a)
        d = V((math.cos(ar), -0.12, math.sin(ar))).normalized()
        base = HC + V((0.250, -0.045, z))
        s.tuft(base, base + d * ln, r, thin=(0, -1, 0), flat=0.7)
    s.tuft(HC + V((0.19, -0.07, 0.15)), HC + V((0.272, -0.09, 0.205)), 0.04, thin=(0, -1, 0), flat=0.7)
    s.ball(HC + V((0.050, -0.205, -0.168)), (0.066, 0.050, 0.045))
    s.mirror()
    s.ball(HC + V((0, 0, 0.02)), (0.285, 0.25, 0.25), seg=48)                    # skull
    s.ball(HC + V((0, -0.03, -0.085)), (0.300, 0.215, 0.175), seg=48)            # cheeks and jaw: widest low down
    s.ball(HC + V((0, -0.105, -0.04)), (0.255, 0.150, 0.20), seg=48)             # a flatter face
    s.ball(HC + V((0, -0.185, -0.228)), (0.062, 0.050, 0.034))                   # chin
    s.ball(HC + V((0, -0.222, -0.128)), (0.040, 0.030, 0.036))                   # bridge under the nose
    # The tuft on top, swept to his left as in the drawing.
    for x, ln, lean, r in ((-0.085, 0.07, 0.25, 0.034), (-0.045, 0.085, 0.35, 0.038), (0.0, 0.10, 0.45, 0.040), (0.045, 0.09, 0.6, 0.038), (0.09, 0.07, 0.8, 0.032)):
        base = HC + V((x, -0.06, 0.02 + 0.225 * math.sqrt(max(0.0, 1 - (x / 0.285) ** 2))))
        s.tuft(base, base + V((lean, -0.25, 1)).normalized() * ln, r, thin=(0, -1, 0), flat=0.7)
    head = add('head', s, mats['fur'], bone='head', voxel=0.008, smooth=4)
    return head


def build_ears(mats):
    out = []
    for side, sx in (('L', 1), ('R', -1)):
        b = HC + V((sx * EAR_BASE.x, EAR_BASE.y, EAR_BASE.z))
        t = HC + V((sx * EAR_TIP.x, EAR_TIP.y, EAR_TIP.z))
        d = t - b
        front = V((sx * 0.30, -1, 0.05)).normalized()
        s = Shape()
        # A leaf: wide low down, a rounded outer edge, a soft point.
        prof = ((-0.18, 0.158), (0.12, 0.162), (0.38, 0.150), (0.62, 0.116), (0.82, 0.070), (0.95, 0.032), (1.0, 0.015))
        for (t0, r0), (t1, r1) in zip(prof, prof[1:]):
            s.cone(b + d * t0, b + d * t1, r0, r1, thin=front, flat=0.30, seg=40)
        s.ball(t, (0.014, 0.011, 0.014), seg=16)
        ear = add('ear_' + side, s, mats['fur'], bone='ear.' + side, voxel=0.005, smooth=8)
        s = Shape()
        fm = front - d.normalized() * front.dot(d.normalized())
        fm.normalize()
        prof = ((0.06, 0.124), (0.30, 0.120), (0.55, 0.092), (0.74, 0.054), (0.89, 0.017))
        for (t0, r0), (t1, r1) in zip(prof, prof[1:]):
            s.cone(b + d * t0 + fm * (0.026 - 0.016 * t0), b + d * t1 + fm * (0.026 - 0.016 * t1), r0, r1, thin=front, flat=0.24, seg=40)
        inner = add('ear_in_' + side, s, mats['ear_in'], bone='ear.' + side, voxel=0.005, smooth=10)
        out += [ear, inner]
    return out


def eye_centre(head, sx):
    """Sink the eye into his face until just the right width of it shows."""
    p = surface(head, sx * EYE.x, EYE.z)
    return V((p.x, p.y + math.sqrt(EYE_R ** 2 - EYE_SHOW ** 2), p.z))


def build_face(mats, head):
    objs = []
    r = EYE_R
    for side, sx in (('L', 1), ('R', -1)):
        c = eye_centre(head, sx)
        o = add('eye_' + side, sphere_part(c, r, lambda v: v.y < 0.55), mats['eye'], 'head')
        o['c'] = list(c)
        objs.append(o)
        objs.append(add('eye_rim_' + side, cap(c, r * 1.006, 0.995, 0.995, rings=6, seg=64, inner=0.705), mats['dark'], 'head'))
        # The pupil: a tall oval on the front of the eye, with the drawing's three glints.
        objs.append(add('pupil_' + side, cap(c, r * 1.012, 0.42, 0.51), mats['pupil'], 'eye.' + side))
        g = [cap(c, r * 1.024, gr, gr, gx, gz, rings=3, seg=24) for gx, gz, gr in ((-0.20, 0.29, 0.165), (0.03, 0.10, 0.060), (0.22, -0.30, 0.072))]
        objs.append(add('glint_' + side, join_meshes(g, 'glint'), mats['glint'], 'eye.' + side))
        # Lids: half a shell each, turned back out of sight when his eyes are open.
        up = Matrix.Rotation(math.radians(-OPEN), 4, 'X')
        lo = Matrix.Rotation(math.radians(OPEN), 4, 'X')
        objs.append(add('lid_up_' + side, sphere_part(c, r * 1.065, lambda v: v.z >= -0.01, up), mats['fur'], 'lid_up.' + side))
        objs.append(add('lash_up_' + side, sphere_part(c, r * 1.072, lambda v: -0.01 <= v.z <= 0.11 and v.y < 0.3, up, seg=64), mats['dark'], 'lid_up.' + side))
        objs.append(add('lid_lo_' + side, sphere_part(c, r * 1.060, lambda v: v.z <= 0.01, lo), mats['fur'], 'lid_lo.' + side))
        objs.append(add('lash_lo_' + side, sphere_part(c, r * 1.067, lambda v: -0.11 <= v.z <= 0.01 and v.y < 0.3, lo, seg=64), mats['dark'], 'lid_lo.' + side))
        # Brow: a short dark stroke above the eye.
        pts = []
        for k in range(7):
            u = k / 6
            x = sx * (0.070 + 0.115 * u)
            z = BROW_Z + 0.030 * math.sin(math.pi * (0.15 + 0.75 * u)) - 0.014 * u
            pts.append(surface(head, x, z, 0.004))
        objs.append(add('brow_' + side, tube(pts, [0.005, 0.008, 0.0095, 0.0095, 0.0085, 0.0065, 0.003]), mats['dark'], 'brow.' + side))
        # Whiskers and the dots they grow from.
        w = []
        for z0, z1, ln in ((-0.142, -0.128, 0.085), (-0.162, -0.170, 0.095)):
            p = [surface(head, sx * (0.118 + ln * u), z0 + (z1 - z0) * u, 0.003 + 0.004 * u) for u in (0, 0.33, 0.66, 1)]
            w.append(tube(p, [0.0026, 0.0023, 0.0018, 0.0008], seg=6))
        for x, z in ((0.066, -0.152), (0.086, -0.170), (0.064, -0.186)):
            p = surface(head, sx * x, z, 0.0)
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=6, radius=0.0045, matrix=Matrix.Translation(p))
            w.append(bm_mesh(bm))
        objs.append(add('whiskers_' + side, join_meshes(w, 'whiskers'), mats['dark'], 'head'))

    # Nose: a soft triangle.
    n = surface(head, 0, NOSE_Z, -0.006)
    s = Shape()
    s.ball(n + V((0.021, 0, 0.008)), (0.021, 0.014, 0.016), seg=16)
    s.ball(n + V((-0.021, 0, 0.008)), (0.021, 0.014, 0.016), seg=16)
    s.ball(n + V((0, 0, -0.011)), (0.018, 0.014, 0.019), seg=16)
    objs.append(add('nose', s, mats['nose'], 'head', voxel=0.003, smooth=2))
    # Mouth line: down from the nose, then out and up each side into his smile.
    z = NOSE_Z
    m = [tube([surface(head, 0, zz, 0.001) for zz in (z - 0.022, z - 0.036, z - 0.050, z - 0.062)], 0.0045, seg=8)]
    for sx in (1, -1):
        pts = [surface(head, sx * x, z + dz, 0.001) for x, dz in ((0, -0.062), (0.018, -0.074), (0.040, -0.079), (0.062, -0.075), (0.082, -0.064), (0.097, -0.048), (0.106, -0.032))]
        m.append(tube(pts, [0.0045, 0.0048, 0.0048, 0.0048, 0.0045, 0.0036, 0.0018], seg=8))
    objs.append(add('mouth_line', join_meshes(m, 'mouth_line'), mats['dark'], 'head'))
    # Under his chin: a short stroke, as in the drawing.
    pts = [surface(head, x, z - 0.106 - 0.006 * math.cos(x / 0.04 * 1.5), 0.001) for x in (-0.04, -0.02, 0, 0.02, 0.04)]
    objs.append(add('chin_line', tube(pts, [0.0015, 0.0035, 0.004, 0.0035, 0.0015], seg=8), mats['dark'], 'head'))
    return objs


def build_mouth(mats, head):
    """The inside of his mouth. It waits inside his head and comes forward as the mouth opens."""
    c = surface(head, 0, MOUTH_Z, 0.0) + V((0, MOUTH_IN, 0))
    objs = []
    s = Shape()
    s.ball(c, (0.060, 0.022, 0.050))
    s.ball(c + V((0, 0, 0.022)), (0.070, 0.022, 0.030))
    o = add('mouth_in', s, mats['mouth'], 'mouth', voxel=0.004, smooth=3)
    o['c'] = list(c)
    objs.append(o)
    s = Shape()
    s.ball(c + V((0, -0.016, -0.022)), (0.036, 0.012, 0.026))
    objs.append(add('tongue', s, mats['tongue'], 'mouth', voxel=0.004, smooth=2))
    s = Shape()
    for sx in (1, -1):
        s.cone(c + V((sx * 0.040, -0.018, 0.040)), c + V((sx * 0.038, -0.020, 0.012)), 0.009, 0.001, seg=10)
        s.cone(c + V((sx * 0.030, -0.018, -0.042)), c + V((sx * 0.031, -0.020, -0.024)), 0.006, 0.001, seg=10)
    objs.append(add('teeth', s, mats['tooth'], 'mouth'))
    return objs


# ----------------------------------------------------------- his things ----
def build_collar(mats):
    axis = (HEADJ - NECK).normalized()
    c = NECK + (HEADJ - NECK) * 0.16
    f = frame(axis).to_3x3()
    fwd = f @ V((0, 1, 0))      # frame() keeps Y near -Y: the front of his neck
    R, h, t = 0.136, 0.027, 0.014
    prof = [(R - t, -h * 0.6), (R - t * 0.2, -h), (R, -h * 0.75), (R, h * 0.75), (R - t * 0.2, h), (R - t, h * 0.6)]
    objs = [add('collar', bm_mesh(lathe(prof, c, axis, 48)), mats['collar'], 'neck')]
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1, matrix=M(c + fwd * (R + 0.002), frame(axis, fwd), (0.078, 0.026, 0.054)))
    bmesh.ops.bevel(bm, geom=bm.edges[:], offset=0.010, segments=3, affect='EDGES', profile=0.5)
    objs.append(add('buckle', bm_mesh(bm), mats['buckle'], 'neck'))
    # The tag hangs from the buckle: a ring, a disc and his name.
    top = c + fwd * (R + 0.016) - axis * 0.022
    down = V((0, -0.10, -1)).normalized()
    ring = [top + V((0.012 * math.cos(a), 0, 0)) + down * (0.020 - 0.022 * math.sin(a)) + V((0, -0.004, 0)) for a in (2 * math.pi * k / 16 for k in range(16))]
    ring_me = tube(ring + [ring[0]], 0.0048, seg=8, caps=False)
    disc_c = top + down * 0.082
    disc = lathe([(0.040, -0.008), (0.050, -0.006), (0.052, 0.0), (0.050, 0.006), (0.040, 0.008)], disc_c, V((0, -1, 0.10)), 40)
    objs.append(add('tag', join_meshes([ring_me, bm_mesh(disc)], 'tag'), mats['tag'], 'tag'))
    cu = bpy.data.curves.new('tagtext', 'FONT')
    cu.body = 'JIMMY'
    cu.align_x = 'CENTER'
    cu.align_y = 'CENTER'
    cu.size = 0.030
    cu.extrude = 0.002
    cu.offset = 0.0011
    cu.space_character = 1.05
    tmp = bpy.data.objects.new('tagtext', cu)
    collection().objects.link(tmp)
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    bpy.data.objects.remove(tmp, do_unlink=True)
    bpy.data.curves.remove(cu)
    # Text is made lying in XY facing +Z; stand it up facing the front, on the disc.
    face = frame(V((0, -1, 0.10)), thin=V((0, 0, 1)))     # Z out of the disc, Y up
    me.transform(M(disc_c + V((0, -0.0085, 0.0)), face))
    txt = add('tag_text', me, mats['tag_text'], 'tag')
    objs.append(txt)
    return objs


def build_headset(mats, head):
    objs = []
    for side, sx, big in (('R', -1, True), ('L', 1, False)):
        c = HC + V((sx * 0.303, -0.035, -0.030))
        axis = V((sx * 1, -0.55, 0.0)).normalized()
        r = 0.060 if big else 0.052
        cup = lathe([(r * 0.80, -0.022), (r, -0.014), (r, 0.012), (r * 0.86, 0.022)], c, axis, 32)
        bmesh.ops.scale(cup, vec=(1, 1, 1.18), verts=cup.verts[:], space=Matrix.Translation(-c))
        objs.append(add('cup_' + side, bm_mesh(cup), mats['set'], 'headset'))
        glass = lathe([(r * 0.66, 0.020), (r * 0.70, 0.027), (r * 0.5, 0.031)], c, axis, 32)
        bmesh.ops.scale(glass, vec=(1, 1, 1.18), verts=glass.verts[:], space=Matrix.Translation(-c))
        objs.append(add('cup_glow_' + side, bm_mesh(glass), mats['set_glow'], 'headset'))
    # The band goes round the back of his head, out of the drawing's view.
    band = [HC + V((0.312 * math.cos(a), 0.266 * math.sin(a) - 0.02, -0.030 + 0.05 * math.sin(a))) for a in (math.pi * k / 20 for k in range(21))]
    objs.append(add('band', tube(band, 0.011, seg=8), mats['set'], 'headset'))
    # The boom comes down from his right cup and round to his mouth.
    b0 = HC + MIC_FROM
    ctrl = [b0, HC + V((-0.318, -0.150, -0.190)), HC + V((-0.285, -0.250, -0.240)), HC + MIC_TO]
    pts = []
    for k in range(13):
        u = k / 12
        p = (1 - u) ** 3 * ctrl[0] + 3 * (1 - u) ** 2 * u * ctrl[1] + 3 * (1 - u) * u * u * ctrl[2] + u ** 3 * ctrl[3]
        pts.append(p)
    objs.append(add('boom', tube(pts, 0.0105, seg=10), mats['set'], 'mic'))
    d = (pts[-1] - pts[-2]).normalized()
    s = Shape()
    s.ball(pts[-1] + d * 0.030, (0.024, 0.024, 0.040), frame(d))
    objs.append(add('mic', s, mats['foam'], 'mic'))
    return objs


def build_hat(mats, head):
    base = HC + V((0.012, -0.015, 0.258))
    axis = V((0.03, -0.02, 1)).normalized()
    H, R, n = 0.34, 0.120, 24
    bm = lathe([(R * (1 - k / n) + 0.004, H * k / n) for k in range(n + 1)], base, axis, 64)
    # Stripes: slanted bands round the cone, pink, yellow, pink, cyan. The cone is cut along each edge so they are clean.
    slant = (axis + V((0.9, 0.0, 0.0))).normalized()
    wide = 0.068
    for k in range(-4, 12):
        bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-6, plane_co=base + slant * (k * wide), plane_no=slant)
    bm.faces.ensure_lookup_table()
    bm.faces.index_update()
    parts = {'hat_pink': [], 'hat_yellow': [], 'hat_cyan': []}
    names = ['hat_pink', 'hat_yellow', 'hat_pink', 'hat_cyan']
    for f in bm.faces:
        t = (f.calc_center_median() - base).dot(slant) / wide
        parts[names[int(math.floor(t)) % 4]].append(f)
    objs = []
    for name, faces in parts.items():
        b2 = bm.copy()
        b2.faces.ensure_lookup_table()
        keep = {f.index for f in faces}
        bmesh.ops.delete(b2, geom=[f for f in b2.faces if f.index not in keep], context='FACES')
        objs.append(add(name, bm_mesh(b2), mats[name], 'hat'))
    bm.free()
    s = Shape()
    tip = base + axis * H
    s.ball(tip, 0.040)
    for k in range(10):
        a = 2 * math.pi * k / 10
        d = V((math.cos(a), math.sin(a) * 0.8, 0.5 * math.sin(a * 2 + 1))).normalized()
        s.ball(tip + d * 0.030 + V((0, 0, 0.012)), 0.024, seg=12)
    objs.append(add('hat_pom', s, mats['pom'], 'hat', voxel=0.005, smooth=2))
    for o in objs:
        o['hat'] = True
    return objs


def build_pads(mats):
    objs = []
    # name, paw middle, where the toes end, where the big pad is, bone, how wide the toes spread, size
    for name, paw, toe_y, sole_y, bone, spread, k in (('pads_front', V((0.118, -0.225, 0.050)), -0.296, -0.212, 'paw', 0.047, 1.25),
                                                     ('pads_hind', V((0.125, 0.145, 0.043)), 0.085, 0.165, 'foot', 0.037, 1.0)):
        for side, sx in (('L', 1), ('R', -1)):
            s = Shape()
            s.ball(V((sx * paw.x, sole_y, 0.008)), (0.030 * k, 0.027 * k, 0.012))
            s.ball(V((sx * paw.x - 0.019 * k, sole_y + 0.013 * k, 0.008)), (0.018 * k, 0.020 * k, 0.011))
            s.ball(V((sx * paw.x + 0.019 * k, sole_y + 0.013 * k, 0.008)), (0.018 * k, 0.020 * k, 0.011))
            for dx, dy in ((-spread, 0.014), (-spread / 3, 0.0), (spread / 3, 0.0), (spread, 0.014)):
                s.ball(V((sx * paw.x + dx, toe_y + dy + 0.022 * k, 0.010)), (0.0125 * k, 0.016 * k, 0.008), seg=12)
            o = s.to_object(name + '_' + side, None, 0, mats['pad'])
            rigid(o, bone + '.' + side)
            objs.append(o)
    return objs


def build():
    mats = j3d_mats.make_all()
    c = HC + V((EYE.x, EYE.y, EYE.z))
    mats['eye'] = j3d_mats.gradient('eye', PAL['eye_bottom'], PAL['eye_top'], c.z - EYE_R * 0.75, c.z + EYE_R * 0.75, edge=PAL['eye_rim'])
    ez = HC.z + EAR_BASE.z
    mats['ear_in'] = j3d_mats.gradient('ear_in', '#dc866a', '#a9676d', ez + 0.02, ez + 0.36)
    mats['pupil'] = j3d_mats.gradient('pupil', PAL['pupil_bottom'], PAL['pupil_top'], c.z - EYE_R * 0.5, c.z + EYE_R * 0.25)
    body = build_body(mats)
    head = build_head(mats)
    bpy.context.view_layer.update()
    parts = [body, head]
    parts += build_ears(mats)
    parts += build_face(mats, head)
    parts += build_mouth(mats, head)
    parts += build_collar(mats)
    parts += build_headset(mats, head)
    parts += build_hat(mats, head)
    parts += build_pads(mats)
    thin = {'whiskers', 'brow', 'mouth_line', 'chin_line', 'lash', 'eye_rim', 'glint', 'pupil', 'eye_', 'tag_text', 'teeth', 'tongue', 'mouth_in', 'lid_'}
    for o in parts:
        if any(o.name.startswith(t) for t in thin):
            continue
        j3d_mats.outline(o, mats['line'], 0.006 if o.name.startswith(('pads', 'nose', 'cup_glow', 'hat_', 'boom', 'band', 'tag', 'buckle')) else 0.010)
    return {'objects': len(parts), 'verts': sum(len(o.data.vertices) for o in parts)}
