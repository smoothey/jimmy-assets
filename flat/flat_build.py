"""Prototype: Jimmy built from the drawing's own layers, each puffed into a soft solid (JM-491). Runs inside Blender."""
import math
import os

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = 'JimmyFlat'


def lin(c):
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]


def build():
    old = bpy.data.collections.get(NAME)
    if old:
        for o in list(old.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(old)
    coll = bpy.data.collections.new(NAME)
    bpy.context.scene.collection.children.link(coll)
    d = np.load(os.path.join(HERE, 'parts.npz'))
    for name in [str(n) for n in d['names']]:
        vf, vb, uv, f = d[name + '_vf'], d[name + '_vb'], d[name + '_uv'], d[name + '_f']
        for side, v, faces in (('front', vf, f), ('back', vb, f[:, ::-1])):
            me = bpy.data.meshes.new(name + '_' + side)
            me.from_pydata(v.tolist(), [], faces.tolist())
            me.shade_smooth()
            layer = me.uv_layers.new(name='UVMap')
            loops = np.array([l.vertex_index for l in me.loops])
            layer.data.foreach_set('uv', uv[loops].ravel())
            m = bpy.data.materials.new('F_' + name + '_' + side)
            m.use_nodes = True
            nt = m.node_tree
            nt.nodes.clear()
            out = nt.nodes.new('ShaderNodeOutputMaterial')
            em = nt.nodes.new('ShaderNodeEmission')
            if side == 'front':
                img = bpy.data.images.load(os.path.join(HERE, name + '.png'), check_existing=True)
                tex = nt.nodes.new('ShaderNodeTexImage')
                tex.image = img
                tex.extension = 'CLIP'
                tr = nt.nodes.new('ShaderNodeBsdfTransparent')
                mix = nt.nodes.new('ShaderNodeMixShader')
                nt.links.new(tex.outputs['Color'], em.inputs['Color'])
                nt.links.new(tex.outputs['Alpha'], mix.inputs['Fac'])
                nt.links.new(tr.outputs[0], mix.inputs[1])
                nt.links.new(em.outputs[0], mix.inputs[2])
                nt.links.new(mix.outputs[0], out.inputs['Surface'])
                m.use_backface_culling = True
            else:
                # The back: the drawing's line colour at the outline, his own colour further round.
                if name + '_g' in d:
                    attr = me.color_attributes.new('shade', 'FLOAT_COLOR', 'POINT')
                    gg = np.clip(d[name + '_g'] * 1.6, 0, 1)
                    attr.data.foreach_set('color', np.repeat(gg[:, None], 4, 1).ravel())
                    an = nt.nodes.new('ShaderNodeAttribute')
                    an.attribute_name = 'shade'
                    mx = nt.nodes.new('ShaderNodeMix')
                    mx.data_type = 'RGBA'
                    nt.links.new(an.outputs['Fac'], mx.inputs['Factor'])
                    mx.inputs['A'].default_value = (*lin((0.208, 0.086, 0.102)), 1)
                    mx.inputs['B'].default_value = (*lin(d[name + '_col']), 1)
                    nt.links.new(mx.outputs['Result'], em.inputs['Color'])
                else:
                    em.inputs['Color'].default_value = (*lin(d[name + '_col']), 1)
                nt.links.new(em.outputs[0], out.inputs['Surface'])
                m.use_backface_culling = True
            me.materials.append(m)
            o = bpy.data.objects.new('flat_' + name + '_' + side, me)
            o['jimmy_flat'] = True
            coll.objects.link(o)
    return len(coll.objects)


def shots(folder, yaws=(0, 20, 40), size=1024, prefix='flat_'):
    sc = bpy.context.scene
    j = bpy.data.collections.get('Jimmy')
    if j:
        j.hide_render = True
    cam = sc.camera
    sc.render.resolution_x = sc.render.resolution_y = size
    sc.render.film_transparent = True
    sc.view_settings.view_transform = 'Standard'
    try:
        sc.eevee.taa_render_samples = 64
    except Exception:
        pass
    out = []
    c = Vector((0, -0.03, 0.615))
    for yaw in yaws:
        a = math.radians(yaw)
        dvec = Vector((math.sin(a), math.cos(a), 0))
        cam.data.type = 'ORTHO'
        cam.data.ortho_scale = 1.23 if yaw == 0 else 1.45
        cam.rotation_euler = dvec.to_track_quat('-Z', 'Y').to_euler()
        cam.location = (Vector((0, 0, 0.615)) if yaw == 0 else c) - dvec * 30
        p = os.path.join(folder, '%syaw%02d.png' % (prefix, yaw))
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        out.append(p)
    if j:
        j.hide_render = False
    return out
