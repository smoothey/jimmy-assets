"""Jimmy from Meshy (JM-491): build him in Blender from Meshy's model, wearing Ant's drawing.

Meshy gives his shape (a real model all the way round). His look is the drawing: version 2's front view is put on
every part of him that faces us and can be seen from the front; Meshy's own paint, recoloured to the drawing's fur
by prep.py, covers the sides and back. Shown unlit, as the flat art is. Next to this file: model.npz and
meshy-fur.png (both from prep.py) and front-art.png (version 2's front view of all of him, 2048 x 3072:
rig.shot(path, size=3072, whole=True) in ../from-layers/rig.py).

In Blender:   import dress; dress.build(); dress.shot(path, yaw=30)
"""
import math
import os

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = 'Jimmy3'
SRC, TALL = 2048, 3072
S = 1.23 / SRC                    # Blender units per drawing pixel, as version 2
# Where the drawing's cat stands in the room (read off version 2's front view): his paws, his ear tip, his middle.
FEET, TOP, MIDDLE = -0.4006, 1.178, -0.0266
NECK = (0.37, 0.085)             # the height of his collar and how far either side of it counts as his neck


def picture(name):
    for folder in (HERE, os.path.join(os.path.dirname(HERE), 'meshy')):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            img = bpy.data.images.load(p, check_existing=True)
            img.reload()
            return img
    raise FileNotFoundError(name)


def material():
    mat = bpy.data.materials.get('J3_skin') or bpy.data.materials.new('J3_skin')
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    N, L = nt.nodes.new, nt.links.new
    out, em = N('ShaderNodeOutputMaterial'), N('ShaderNodeEmission')
    own = N('ShaderNodeTexImage')
    own.image = picture('meshy-fur.png')
    art = N('ShaderNodeTexImage')
    art.image = picture('front-art.png')
    art.extension = 'CLIP'
    um = N('ShaderNodeUVMap')
    um.uv_map = 'front'
    L(um.outputs[0], art.inputs['Vector'])
    w = N('ShaderNodeAttribute')
    w.attribute_name = 'wfront'
    mul = N('ShaderNodeMath')
    mul.operation = 'MULTIPLY'
    L(w.outputs['Fac'], mul.inputs[0])
    L(art.outputs['Alpha'], mul.inputs[1])
    mix = N('ShaderNodeMix')
    mix.data_type = 'RGBA'
    L(mul.outputs[0], mix.inputs['Factor'])
    # Meshy painted his collar teal; the drawing's is blue. Turn the teal blue, but only round his neck (the
    # headset is teal in the drawing too).
    sep, comb, add = N('ShaderNodeSeparateColor'), N('ShaderNodeCombineColor'), N('ShaderNodeMath')
    sep.mode = comb.mode = 'HSV'
    L(own.outputs['Color'], sep.inputs['Color'])
    add.operation = 'ADD'
    add.inputs[1].default_value = 0.11
    L(sep.outputs['Red'], add.inputs[0])
    L(add.outputs[0], comb.inputs['Red'])
    L(sep.outputs['Green'], comb.inputs['Green'])
    L(sep.outputs['Blue'], comb.inputs['Blue'])
    near = N('ShaderNodeMapRange')            # how teal: 1 at hue 0.5, 0 by 0.1 either side
    near.inputs['From Min'].default_value, near.inputs['From Max'].default_value = 0.10, 0.04
    dist, ab = N('ShaderNodeMath'), N('ShaderNodeMath')
    dist.operation, ab.operation = 'SUBTRACT', 'ABSOLUTE'
    dist.inputs[1].default_value = 0.5
    L(sep.outputs['Red'], dist.inputs[0])
    L(dist.outputs[0], ab.inputs[0])
    L(ab.outputs[0], near.inputs['Value'])
    vivid = N('ShaderNodeMapRange')
    vivid.inputs['From Min'].default_value, vivid.inputs['From Max'].default_value = 0.25, 0.45
    L(sep.outputs['Green'], vivid.inputs['Value'])
    neck = N('ShaderNodeAttribute')
    neck.attribute_name = 'neck'
    m1, m2 = N('ShaderNodeMath'), N('ShaderNodeMath')
    m1.operation = m2.operation = 'MULTIPLY'
    L(near.outputs['Result'], m1.inputs[0])
    L(vivid.outputs['Result'], m1.inputs[1])
    L(m1.outputs[0], m2.inputs[0])
    L(neck.outputs['Fac'], m2.inputs[1])
    blue = N('ShaderNodeMix')
    blue.data_type = 'RGBA'
    L(m2.outputs[0], blue.inputs['Factor'])
    L(own.outputs['Color'], blue.inputs['A'])
    L(comb.outputs['Color'], blue.inputs['B'])
    L(blue.outputs['Result'], mix.inputs['A'])
    L(art.outputs['Color'], mix.inputs['B'])
    L(mix.outputs['Result'], em.inputs['Color'])
    L(em.outputs[0], out.inputs['Surface'])
    return mat


def front_weights(me):
    """For each point: where it falls on the front view, and how much of the drawing it takes (it must face us and
    nothing may stand in front of it, or his tail would wear his face)."""
    n = len(me.vertices)
    co = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    nor = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('normal', nor)
    nor = nor.reshape(-1, 3)
    uv = np.stack([(co[:, 0] / S + SRC / 2) / SRC, (co[:, 2] / S + (TALL - SRC)) / TALL], 1).astype('f4')
    loops = np.zeros(len(me.loops), 'i4')
    me.loops.foreach_get('vertex_index', loops)
    layer = me.uv_layers.get('front') or me.uv_layers.new(name='front')
    layer.data.foreach_set('uv', uv[loops].ravel())
    bvh = BVHTree.FromPolygons([Vector(v) for v in co.tolist()], [tuple(p.vertices) for p in me.polygons])
    facing = -nor[:, 1]
    seen = np.zeros(n, 'f4')
    to_us = Vector((0, -1, 0))
    for i in np.nonzero(facing > 0.05)[0].tolist():
        start = Vector(co[i].tolist()) + Vector(nor[i].tolist()) * 0.002 + to_us * 0.002
        if bvh.ray_cast(start, to_us)[0] is None:
            seen[i] = 1.0
    t = np.clip((facing - 0.15) / 0.35, 0, 1)
    w = seen * t * t * (3 - 2 * t)
    ed = np.zeros(len(me.edges) * 2, 'i4')
    me.edges.foreach_get('vertices', ed)
    ed = ed.reshape(-1, 2)
    deg = np.zeros(n)
    np.add.at(deg, ed[:, 0], 1)
    np.add.at(deg, ed[:, 1], 1)
    for _ in range(4):                        # soften the edge between drawing and paint
        acc = np.zeros(n)
        np.add.at(acc, ed[:, 0], w[ed[:, 1]])
        np.add.at(acc, ed[:, 1], w[ed[:, 0]])
        w = 0.5 * w + 0.5 * acc / np.maximum(deg, 1)
    neck = np.clip(1 - np.abs(co[:, 2] - NECK[0]) / NECK[1], 0, 1) > 0
    nk = me.attributes.get('neck') or me.attributes.new('neck', 'FLOAT', 'POINT')
    nk.data.foreach_set('value', neck.astype('f4'))
    at = me.attributes.get('wfront') or me.attributes.new('wfront', 'FLOAT', 'POINT')
    at.data.foreach_set('value', w.astype('f4'))
    return int(seen.sum())


def build():
    old = bpy.data.collections.get(NAME)
    if old:
        for o in list(old.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(old)
    coll = bpy.data.collections.new(NAME)
    bpy.context.scene.collection.children.link(coll)
    d = np.load(os.path.join(HERE, 'model.npz'))
    co, uv, tri = d['co'].astype('f8'), d['uv'], d['tri']
    co *= (TOP - FEET) / (co[:, 2].max() - co[:, 2].min())
    co[:, 2] += FEET - co[:, 2].min()
    co[:, 0] += MIDDLE - (co[:, 0].max() + co[:, 0].min()) / 2
    me = bpy.data.meshes.new('jimmy3')
    me.from_pydata(co.tolist(), [], tri.tolist())
    loops = np.zeros(len(me.loops), 'i4')
    me.loops.foreach_get('vertex_index', loops)
    me.uv_layers.new(name='UVMap').data.foreach_set('uv', uv[loops].ravel())
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)       # Meshy splits points along the seams of its paint
    bmesh.ops.join_triangles(bm, faces=bm.faces, angle_face_threshold=math.radians(40), angle_shape_threshold=math.radians(40), cmp_uvs=True)
    bm.to_mesh(me)
    bm.free()
    me.shade_smooth()
    seen = front_weights(me)
    me.materials.append(material())
    o = bpy.data.objects.new('j3_skin', me)
    o['jimmy3'] = True
    coll.objects.link(o)
    return {'points': len(me.vertices), 'faces': len(me.polygons), 'four_sided': sum(1 for p in me.polygons if len(p.vertices) == 4), 'seen_from_front': seen}


def camera(yaw=0.0, pitch=0.0, size=1200, whole=True):
    """Look at him from yaw degrees round (0 = straight on, as the drawing). whole: all of him; else the badge's frame."""
    sc = bpy.context.scene
    for c in bpy.data.collections:
        if c.name in ('Jimmy', 'Jimmy2', 'JimmyMeshy', NAME):
            c.hide_render = c.name != NAME
    cam = sc.camera
    if cam is None:
        cam = bpy.data.objects.new('JimmyCam', bpy.data.cameras.new('JimmyCam'))
        sc.collection.objects.link(cam)
        sc.camera = cam
    frame = SRC * S
    tall = TALL * S if whole else frame
    sc.render.engine = 'BLENDER_EEVEE'
    sc.render.resolution_x, sc.render.resolution_y = int(round(size * frame / tall)), size
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.view_settings.view_transform = 'Standard'
    a, p = math.radians(yaw), math.radians(pitch)
    look = Vector((math.sin(a) * math.cos(p), math.cos(a) * math.cos(p), -math.sin(p)))
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = tall if yaw == 0 and pitch == 0 else tall * 1.15
    cam.data.clip_end = 100
    cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()
    cam.location = Vector((0, 0, frame - tall / 2)) - look * 30


def shot(path, yaw=0.0, pitch=0.0, size=1200, whole=True):
    camera(yaw, pitch, size, whole)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path
