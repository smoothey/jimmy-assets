"""Jimmy on Meshy's bones (JM-491): Ant's first Meshy cat, rigged by Meshy's Smart Rig, set up and moved in Blender.

Next to this file: jimmy-rigged.glb (Meshy's rigged model, made smaller by slim.py). Meshy's own look is kept: its
paint, lit softly. Its bones have numbers for names; NAMES says which is which (read off where each one sits).

In Blender:   import rigged; rigged.build(); rigged.moves(); rigged.shot(path, action='demo', at=0.5)
"""
import math
import os

import bpy
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = 'Jimmy4'
FPS = 30
# Meshy's bone: ours. Left and right are as we see him (he faces us).
NAMES = {
    'Bone_000': 'root', 'Bone_001': 'hips',
    'Bone_005': 'tail-1', 'Bone_004': 'tail-2', 'Bone_003': 'tail-3', 'Bone_002': 'tail-4',
    'Bone_008': 'spine-1', 'Bone_007': 'spine-2', 'Bone_006': 'chest',
    'Bone_019': 'neck', 'Bone_018': 'head', 'Bone_017': 'crown',
    'Bone_028': 'ear-left-1', 'Bone_027': 'ear-left-2', 'Bone_030': 'ear-right-1', 'Bone_029': 'ear-right-2',
    'Bone_022': 'arm-left-1', 'Bone_021': 'arm-left-2', 'Bone_020': 'arm-left-3',
    'Bone_026': 'arm-right-1', 'Bone_025': 'arm-right-2', 'Bone_024': 'arm-right-3', 'Bone_023': 'arm-right-4',
    'Bone_012': 'leg-right-1', 'Bone_011': 'leg-right-2', 'Bone_010': 'leg-right-3', 'Bone_009': 'leg-right-4',
    'Bone_016': 'leg-left-1', 'Bone_015': 'leg-left-2', 'Bone_014': 'leg-left-3', 'Bone_013': 'leg-left-4',
}


def build():
    old = bpy.data.collections.get(NAME)
    if old:
        for o in list(old.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(old)
    for act in list(bpy.data.actions):
        if act.get('j4'):
            bpy.data.actions.remove(act)
    coll = bpy.data.collections.new(NAME)
    bpy.context.scene.collection.children.link(coll)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(HERE, 'jimmy-rigged.glb'))
    new = [o for o in bpy.data.objects if o not in before]
    arm = next(o for o in new if o.type == 'ARMATURE')
    skin = max((o for o in new if o.type == 'MESH'), key=lambda o: len(o.data.vertices))
    for o in new:
        if o not in (arm, skin):                  # Meshy's little marker ball for its bones
            bpy.data.objects.remove(o, do_unlink=True)
            continue
        for c in list(o.users_collection):
            c.objects.unlink(o)
        coll.objects.link(o)
    arm.name, skin.name = NAME, 'j4_skin'
    for b in arm.data.bones:
        b.name = NAMES.get(b.name, b.name)        # the skin's weights follow the new names
    for pb in arm.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.custom_shape = None
    # Meshy's paint, lit softly: no shine, no metal (the lighting pictures were left behind by slim.py).
    for mat in skin.data.materials:
        nt = mat.node_tree
        bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
        for sock in ('Metallic', 'Roughness', 'Normal'):
            for l in list(bsdf.inputs[sock].links):
                nt.links.remove(l)
        bsdf.inputs['Metallic'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 0.75
    stage()
    return {'bones': len(arm.data.bones), 'points': len(skin.data.vertices), 'weights': len(skin.vertex_groups), 'height': round(skin.dimensions[2], 3)}


def stage():
    """Soft, even light: a bright sky and one gentle sun from the front left."""
    sc = bpy.context.scene
    world = sc.world or bpy.data.worlds.new('JimmyWorld')
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    bg.inputs['Color'].default_value = (1, 1, 1, 1)
    bg.inputs['Strength'].default_value = 0.9
    sun = bpy.data.objects.get('JimmySun')
    if sun is None:
        sun = bpy.data.objects.new('JimmySun', bpy.data.lights.new('JimmySun', 'SUN'))
        sc.collection.objects.link(sun)
    sun.data.energy = 2.2
    sun.data.angle = math.radians(25)
    sun.rotation_euler = (math.radians(55), 0, math.radians(-35))


def turn(arm, bone, x=0.0, y=0.0, z=0.0):
    """Turn a bone by degrees about the room's axes (x: nod, y: roll, z: turn), from its rest."""
    pb = arm.pose.bones[bone]
    R = arm.data.bones[bone].matrix_local.to_3x3()
    q = Matrix.Rotation(math.radians(z), 3, 'Z') @ Matrix.Rotation(math.radians(x), 3, 'X') @ Matrix.Rotation(math.radians(y), 3, 'Y')
    pb.rotation_quaternion = (R.inverted() @ q @ R).to_quaternion()
    return pb


def write(name, seconds, tracks):
    """tracks: {bone: [(t from 0 to 1, {'x':, 'y':, 'z':}), ...]}"""
    arm = bpy.data.objects[NAME]
    old = bpy.data.actions.get(name)
    if old:
        bpy.data.actions.remove(old)
    act = bpy.data.actions.new(name)
    act['j4'] = True
    act.use_fake_user = True
    if not arm.animation_data:
        arm.animation_data_create()
    arm.animation_data.action = act
    frames = seconds * FPS
    for bone, keys in tracks.items():
        for t, v in keys:
            turn(arm, bone, **v).keyframe_insert('rotation_quaternion', frame=1 + t * frames, group=bone)
    act.use_frame_range = True
    act.frame_start, act.frame_end = 1, 1 + frames
    rest()
    return act


def rest():
    arm = bpy.data.objects[NAME]
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)


def moves():
    """A first set, to see how Meshy's bones and weights hold up: a look round, an ear twitch, a tail swish, a wave."""
    Z = {}
    sway = lambda a: [(0, Z), (0.25, {'y': a}), (0.75, {'y': -a}), (1, Z)]
    write('look-round', 3.0, {
        'neck': [(0, Z), (0.2, {'z': 26}), (0.45, {'z': 26}), (0.7, {'z': -26}), (0.9, {'z': -26}), (1, Z)],
        'chest': [(0, Z), (0.2, {'z': 6}), (0.45, {'z': 6}), (0.7, {'z': -6}), (0.9, {'z': -6}), (1, Z)],
    })
    write('ear-twitch', 0.8, {
        'ear-right-1': [(0, Z), (0.25, {'y': 16, 'x': 8}), (0.5, {'y': -6}), (0.75, {'y': 8}), (1, Z)],
        'ear-left-1': [(0, Z), (0.35, {'y': -10}), (0.7, {'y': 4}), (1, Z)],
    })
    write('tail-swish', 2.0, {'tail-1': [(0, Z), (0.25, {'z': 10}), (0.75, {'z': -10}), (1, Z)], 'tail-2': sway(12), 'tail-3': sway(16), 'tail-4': sway(20)})
    up = {'x': -70, 'y': -22}
    write('wave', 2.2, {
        'arm-right-1': [(0, Z), (0.2, up), (0.8, up), (1, Z)],
        'arm-right-2': [(0, Z), (0.2, {'x': -75}), (0.35, {'x': -75, 'y': -20}), (0.5, {'x': -75, 'y': 18}), (0.65, {'x': -75, 'y': -20}), (0.8, {'x': -75, 'y': 12}), (1, Z)],
        'arm-right-3': [(0, Z), (0.2, {'x': -25}), (0.8, {'x': -25}), (1, Z)],
        'neck': [(0, Z), (0.25, {'y': 6}), (0.8, {'y': 6}), (1, Z)],
        'ear-right-1': [(0, Z), (0.3, {'y': 10}), (0.8, {'y': 6}), (1, Z)],
    })
    bpy.data.objects[NAME].animation_data.action = None
    rest()
    return sorted(a.name for a in bpy.data.actions if a.get('j4'))


def show(action=None, at=0.5):
    arm = bpy.data.objects[NAME]
    rest()
    if action:
        act = bpy.data.actions[action]
        arm.animation_data.action = act
        f = act.frame_start + (act.frame_end - act.frame_start) * at
        bpy.context.scene.frame_set(int(f), subframe=f - int(f))
    else:
        arm.animation_data.action = None
        bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()


def camera(yaw=0.0, pitch=0.0, size=900):
    sc = bpy.context.scene
    for c in bpy.data.collections:
        if c.name in ('Jimmy', 'Jimmy2', 'Jimmy3', 'JimmyMeshy', NAME):
            c.hide_render = c.name != NAME
    cam = sc.camera
    if cam is None:
        cam = bpy.data.objects.new('JimmyCam', bpy.data.cameras.new('JimmyCam'))
        sc.collection.objects.link(cam)
        sc.camera = cam
    sc.render.engine = 'BLENDER_EEVEE'
    sc.render.resolution_x, sc.render.resolution_y = int(size * 0.8), size
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.view_settings.view_transform = 'Standard'
    a, p = math.radians(yaw), math.radians(pitch)
    look = Vector((math.sin(a) * math.cos(p), math.cos(a) * math.cos(p), -math.sin(p)))
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 2.05
    cam.data.clip_end = 100
    cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()
    cam.location = Vector((0, 0, 0.86)) - look * 30


def shot(path, yaw=0.0, pitch=0.0, action=None, at=0.5, size=900):
    show(action, at)
    camera(yaw, pitch, size)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path
