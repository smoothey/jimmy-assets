"""3D Jimmy: shared helpers (JM-491). Runs inside Blender.

Jimmy faces -Y (towards a front camera), +X is his left, Z is up.
Sizes are in Blender units; he is about 1.5 tall sitting, ears and all.
"""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

COLL = 'Jimmy'

# The drawing's colours, sampled from Ant's layers (Downloads/Jimmy/export).
PAL = {
    'fur_shadow': '#3a2a3c', 'fur_mid': '#513b43', 'fur_light': '#694c4c', 'fur_rim': '#906354',
    'line': '#35161a',
    'ear_in': '#cf7866', 'ear_in_dark': '#a8605f', 'ear_in_light': '#e38c68',
    'eye_top': '#e49d44', 'eye_bottom': '#f8da6b', 'eye_rim': '#b5613a',
    'pupil_top': '#3d1711', 'pupil_bottom': '#7f3f15', 'glint': '#fff3cf',
    'nose': '#a8696c', 'nose_dark': '#6f3f45', 'pad': '#9a625c', 'pad_light': '#b97a70',
    'collar': '#4366aa', 'collar_dark': '#39407c', 'collar_light': '#6b8fd0',
    'buckle': '#b5352f', 'buckle_dark': '#6e2122', 'buckle_light': '#d85a4c',
    'tag': '#f29d3e', 'tag_dark': '#b9682c', 'tag_light': '#fcd574', 'tag_text': '#4f1a0f',
    'set': '#545d68', 'set_dark': '#33363f', 'set_light': '#8b9aa1', 'set_glow': '#93c9c7',
    'foam': '#2c2f36', 'foam_light': '#4c5560',
    'mouth': '#6a2330', 'mouth_dark': '#3a171a', 'tongue': '#d67a7c', 'tongue_dark': '#a85560', 'tooth': '#f6ecd6',
    'hat_pink': '#fd5b72', 'hat_yellow': '#fddf46', 'hat_cyan': '#4fdbdc', 'hat_dark': '#c72e4a', 'pom': '#f7b54c',
}


def rgb(hex_, a=1.0):
    """#rrggbb to linear RGBA (Blender works in linear light)."""
    h = hex_.lstrip('#')
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return (*out, a)


def collection():
    c = bpy.data.collections.get(COLL)
    if not c:
        c = bpy.data.collections.new(COLL)
        bpy.context.scene.collection.children.link(c)
    return c


def wipe():
    """Remove everything this build made, and the start-up cube."""
    for o in list(bpy.data.objects):
        if o.name in ('Cube',) or o.get('jimmy'):
            bpy.data.objects.remove(o, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.armatures, bpy.data.curves):
        for d in list(block):
            if d.users == 0:
                block.remove(d)
    for m in list(bpy.data.materials):
        if m.name.startswith('J_'):
            bpy.data.materials.remove(m)
    for a in list(bpy.data.actions):
        bpy.data.actions.remove(a)


def frame(z, thin=None):
    """A rotation whose Z runs along z and whose Y is as near to thin as it can be."""
    z = Vector(z).normalized()
    ref = Vector(thin) if thin is not None else Vector((0, -1, 0))
    y = ref - z * ref.dot(z)
    if y.length < 1e-5:
        ref = Vector((1, 0, 0))
        y = ref - z * ref.dot(z)
    y.normalize()
    x = y.cross(z)
    return Matrix((x, y, z)).transposed().to_4x4()


def M(loc, rot=None, scale=(1, 1, 1)):
    m = Matrix.Translation(Vector(loc))
    if rot is not None:
        m = m @ rot
    return m @ Matrix.Diagonal((*scale, 1))


class Shape:
    """Solid pieces dropped into one mesh, then fused into one skin."""

    def __init__(self):
        self.bm = bmesh.new()

    def ball(self, c, r, rot=None, seg=32):
        r = (r, r, r) if isinstance(r, (int, float)) else r
        bmesh.ops.create_uvsphere(self.bm, u_segments=seg, v_segments=seg // 2, radius=1, matrix=M(c, rot, r))
        return self

    def cone(self, p0, p1, r0, r1, thin=None, flat=1.0, wide=1.0, seg=24):
        p0, p1 = Vector(p0), Vector(p1)
        d = p1 - p0
        bmesh.ops.create_cone(self.bm, cap_ends=True, cap_tris=False, segments=seg, radius1=r0, radius2=max(r1, 1e-4),
                              depth=d.length, matrix=M((p0 + p1) / 2, frame(d, thin), (wide, flat, 1)))
        return self

    def limb(self, p0, p1, r0, r1, thin=None, flat=1.0):
        """A tapered sausage: a cone with a ball at each end."""
        rot = frame(Vector(p1) - Vector(p0), thin)
        self.cone(p0, p1, r0, r1, thin, flat)
        self.ball(p0, (r0, r0 * flat, r0), rot)
        self.ball(p1, (r1, r1 * flat, r1), rot)
        return self

    def tuft(self, base, tip, r, thin=None, flat=0.6):
        """A lock of fur: a flattened cone with a rounded root."""
        rot = frame(Vector(tip) - Vector(base), thin)
        self.cone(base, tip, r, r * 0.06, thin, flat)
        self.ball(base, (r, r * flat, r), rot, seg=16)
        return self

    def chain(self, pts, radii, thin=None, flat=1.0):
        for i in range(len(pts) - 1):
            self.limb(pts[i], pts[i + 1], radii[i], radii[i + 1], thin, flat)
        return self

    def mirror(self):
        """Copy everything made so far across X (call before adding middle pieces)."""
        geom = self.bm.verts[:] + self.bm.edges[:] + self.bm.faces[:]
        dup = bmesh.ops.duplicate(self.bm, geom=geom)
        verts = [g for g in dup['geom'] if isinstance(g, bmesh.types.BMVert)]
        faces = [g for g in dup['geom'] if isinstance(g, bmesh.types.BMFace)]
        bmesh.ops.scale(self.bm, vec=(-1, 1, 1), verts=verts)
        bmesh.ops.reverse_faces(self.bm, faces=faces)
        return self

    def to_object(self, name, voxel=None, smooth=0, mat=None):
        me = bpy.data.meshes.new(name)
        self.bm.to_mesh(me)
        self.bm.free()
        obj = bpy.data.objects.new(name, me)
        obj['jimmy'] = True
        collection().objects.link(obj)
        if voxel:
            fuse(obj, voxel, smooth)
        obj.data.shade_smooth()
        if mat:
            obj.data.materials.append(mat)
        return obj


def fuse(obj, voxel, smooth=0):
    """Fuse overlapping pieces into one skin (voxel remesh), then soften."""
    m = obj.modifiers.new('fuse', 'REMESH')
    m.mode = 'VOXEL'
    m.voxel_size = voxel
    m.adaptivity = 0
    m.use_smooth_shade = True
    if smooth:
        s = obj.modifiers.new('soft', 'SMOOTH')
        s.factor = 0.5
        s.iterations = smooth
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    old = obj.data
    obj.modifiers.clear()
    obj.data = me
    me.name = obj.name
    bpy.data.meshes.remove(old)
    obj['islands'] = keep_biggest(me)


def keep_biggest(me):
    """Drop any loose bits inside the skin (bubbles left where pieces overlapped). Returns how many there were."""
    bm = bmesh.new()
    bm.from_mesh(me)
    seen = set()
    islands = []
    for v in bm.verts:
        if v.index in seen:
            continue
        stack, isle = [v], []
        seen.add(v.index)
        while stack:
            a = stack.pop()
            isle.append(a)
            for e in a.link_edges:
                b = e.other_vert(a)
                if b.index not in seen:
                    seen.add(b.index)
                    stack.append(b)
        islands.append(isle)
    if len(islands) > 1:
        islands.sort(key=len, reverse=True)
        bmesh.ops.delete(bm, geom=[v for isle in islands[1:] for v in isle], context='VERTS')
        bm.to_mesh(me)
    bm.free()
    return len(islands)


def rigid(obj, bone):
    """The whole object follows one bone."""
    for g in list(obj.vertex_groups):
        obj.vertex_groups.remove(g)
    g = obj.vertex_groups.new(name=bone)
    g.add(range(len(obj.data.vertices)), 1.0, 'REPLACE')
    obj['bone'] = bone


def deg(*a):
    return tuple(math.radians(x) for x in a)
