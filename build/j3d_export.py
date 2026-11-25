"""3D Jimmy: save the Blender file, export him for the web, render clips (JM-491). Runs inside Blender."""
import os

import bpy

from j3d_common import rgb


def save(path):
    bpy.ops.wm.save_as_mainfile(filepath=path)
    return os.path.getsize(path)


def glb(path):
    """Export him as one .glb: skeleton, skin, every action. Plain colours: the page draws the toon look and the line itself."""
    plain = {}
    for m in list(bpy.data.materials):
        if not m.name.startswith('J_'):
            continue
        p = bpy.data.materials.new('P_' + m.name[2:])
        p.use_nodes = True
        bsdf = p.node_tree.nodes.get('Principled BSDF')
        bsdf.inputs['Base Color'].default_value = rgb(m.get('base', '#808080'))
        bsdf.inputs['Roughness'].default_value = 1.0
        plain[m.name] = p
    slots = []
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.get('jimmy'):
            for s in o.material_slots:
                if s.material and s.material.name in plain:
                    slots.append((s, s.material))
                    s.material = plain[s.material.name]
    arm = bpy.data.objects['Jimmy']
    was = arm.animation_data.action
    try:
        bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', export_animations=True, export_animation_mode='ACTIONS',
                                  export_apply=False, export_skins=True, export_extras=True, export_yup=True,
                                  export_force_sampling=True, export_optimize_animation_size=True)
    finally:
        for s, m in slots:
            s.material = m
        for p in plain.values():
            bpy.data.materials.remove(p)
        arm.animation_data.action = was
    return os.path.getsize(path)


def clip(folder, action, view='badge', size=512, step=1, hold=0, pad=1.12):
    """Render one action as PNG frames with a clear background: <folder>/<action>/0001.png ..."""
    import j3d_moves
    import j3d_scene
    j3d_scene.setup(size)
    sc = bpy.context.scene
    act = bpy.data.actions[action]
    j3d_moves.show(action, 1)
    j3d_scene.badge_view() if view == 'badge' else j3d_scene.view(view, None, pad)
    out = os.path.join(folder, action)
    os.makedirs(out, exist_ok=True)
    n = 0
    last = int(round(act.frame_end)) + hold
    for f in range(1, last + 1, step):
        sc.frame_set(min(f, int(round(act.frame_end))))
        n += 1
        sc.render.filepath = os.path.join(out, '%04d.png' % n)
        bpy.ops.render.render(write_still=True)
    return n
