"""Jimmy, version 5 (JM-491): the open-mouth standing cat from Meshy, on Meshy's Quadruped Dog rig, in Blender.

Meshy made the model (from its own picture of Ant's first Meshy cat, standing, mouth open), rebuilt it to about
100,000 four-sided faces, rigged it and added five of its ready-made motions. Next to this file: jimmy-open.glb,
that download as it came. Meshy's look is kept: its paint, lit softly.

Its rig has 52 bones, among them a jaw (jaw_01), three in each ear, six in his tail and eight small ones in his face.
His mouth was modelled open so that it has an inside; turning the jaw by SHUT degrees closes it.

The jaw alone leaves a gap with teeth and red showing, and a jaw made for a big open smile is too big for a shut
face. So seal() adds things that all follow the jaw bone: a 'shut' shape that makes his lower jaw smaller (SMALL),
lifts his bottom lip to his top lip and packs his teeth and tongue away deep in his head (worked out in mouth5.py);
a short dark mouth line in the middle; and fur colour over the rest of the join, where Meshy painted a dark rim
round his open mouth. Any motion that turns jaw_01 gets them. The target was Meshy's own closed-mouth cat.

build() also: deletes a few loose specks Meshy's model came with; makes his name tag move as one piece (tag());
holds his jaw shut all through Meshy's five motions (shut_in()); and paints him eyelids (eyes(), blink()).

His everyday pose is sitting (sit()), a still frame of Meshy's sitting motion. moves() writes his own motions from
it: j5-wave, j5-look-round, j5-nod, j5-ear-twitch, j5-tail-swish, j5-talk, j5-blink, j5-sit. His paw leaves the
ground only in the wave.

In Blender:   import jimmy5; jimmy5.build(); jimmy5.moves(); jimmy5.sit(); jimmy5.shot(path, face=True)
              jimmy5.clip(folder, name='j5-wave')      # a motion as numbered frames
"""
import math
import os

import bpy
import numpy as np
from mathutils import Matrix, Vector

import mouth5

HERE = os.path.dirname(os.path.abspath(__file__))
ALL = ('Jimmy', 'Jimmy2', 'Jimmy3', 'Jimmy4', 'JimmyMeshy', 'Jimmy5', 'Jimmy5ref', 'Jimmy6')

# What differs between the models this script can build. use('5') or use('6') picks one; both can sit in one Blender file.
#   5: Meshy's cat from a three-quarter picture. His head is modelled turned 52 degrees from his body.
#   6: the same cat from a straight-on picture: head forward, standing square, the same left and right.
MODELS = {
    '5': dict(
        NAME='Jimmy5', SKIN='j5_skin', FLAG='j5', PRE='', OWN='j5-', GLB='jimmy-open.glb',
        SHUT=-42.0,                         # degrees on jaw_01 that bring his lower jaw up to his top lip
        HEAD=(-0.52, -0.45, 0.30),          # the middle of his face in the room, for close-ups
        MOUTH=(-0.51, 0.26, -0.38, -0.10, 0.30),    # the box his mouth is in: middle across, half width, front of, from, to (height)
        FACE=(0.15, -0.30, 0.20),           # the front of his head is above this and in front of this; his eyes are above this
        TAG=0.12,                           # his name tag is below this height
        EYEHUE=(0.09, 0.19),                # the colour of his eyes, as a range of hue
        TEETH=(0.25, 0.6),                  # his teeth: paint with less colour than this, and brighter than this
        FANGS=False,                        # whether some teeth are cut off from the rest of the inside of his mouth
        EDGE=(1.0, 0.3),                    # how much of the dark mouth line sits on the edge of his top lip, and of his bottom lip
        WIDE=(0.40, 0.70),                  # the line is full to this far across his mouth, and gone by this far (1 = the corners)
        THICK=0.0,                          # if set, the line is painted this far on to each lip instead (Meshy gave model 5 a dark rim)
        SMALL=(0.73, 0.81, 0.40),           # his lower jaw when shut: width, how far forward, height from lip to chin (1 = as modelled)
        SMILE=0.008,                        # how far the corners of his shut mouth lift
        SIT=('dog_beifumo', 66),            # the frame of Meshy's sitting motion where he sits upright and still: his everyday pose
        WAVE=((0.75, -0.45, 0.35), (0.25, -0.15, 1.0), (0.10, -0.30, 1.0), (0.0, -0.50, 1.0)),   # where each part of his raised leg points
    ),
    '6': dict(
        NAME='Jimmy6', SKIN='j6_skin', FLAG='j6', PRE='v6_', OWN='j6-', GLB=os.path.join('..', 'v6', 'jimmy-open.glb'),
        SHUT=-45.0,
        HEAD=(0.004, -0.70, 0.33),
        MOUTH=(0.004, 0.26, -0.59, -0.09, 0.31),
        FACE=(0.15, -0.50, 0.20),
        TAG=0.12,
        EYEHUE=(0.06, 0.19),                # his eyes came out more amber
        TEETH=(0.30, 0.55),                 # his fangs are shaded greyer (his whiskers are greyer still, and must not count)
        FANGS=True,
        EDGE=(1.0, 1.0),
        WIDE=(0.55, 0.90),
        THICK=0.016,
        SMALL=(1.0, 1.0, 1.0),               # his jaw is already small for his head
        SMILE=0.008,
        SIT=('dog_beifumo', 66),
        WAVE=((0.75, -0.45, 0.35), (0.25, -0.15, 1.0), (0.10, -0.30, 1.0), (0.0, -0.50, 1.0)),
    ),
}


def use(model):
    """Pick which model the rest of this script works on."""
    globals().update(MODELS[str(model)])
    return NAME


def motion(name):
    """The name one of Meshy's motions goes by for this model."""
    return PRE + name


LINE = (0.03, 0.02, 0.03, 1.0)    # the dark line of his shut mouth
# how sealed his mouth is, 0 to 1, from the jaw bone's own turn (w): nothing until 60% shut, all by fully shut
SEALED = 'clamp((2*acos(min(1,abs(w)))/%.4f-0.6)/0.4,0,1)'      # the gap is filled with SHUT, in radians
EYE = (1.0, 0.22, 0.90, 0.07)     # his eyelids: (spare), how much the lid's edge curves, how far down a full blink
                                  # brings it, and how thick the lash line is
MOTIONS = ('cat_idle', 'cat_sajiao', 'cat_walk', 'dog_beifumo', 'dog_yaoweiba')   # Meshy's names: sajiao is act cute, beifumo is being petted, yaoweiba is wag tail


def build():
    old = bpy.data.collections.get(NAME)
    if old:
        for o in list(old.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.collections.remove(old)
    for act in list(bpy.data.actions):
        if act.get(FLAG):
            bpy.data.actions.remove(act)
    had = set(bpy.data.actions)
    coll = bpy.data.collections.new(NAME)
    bpy.context.scene.collection.children.link(coll)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.normpath(os.path.join(HERE, GLB)))
    new = [o for o in bpy.data.objects if o not in before]
    arm = next(o for o in new if o.type == 'ARMATURE')
    skin = max((o for o in new if o.type == 'MESH'), key=lambda o: len(o.data.vertices))
    for o in new:
        if o not in (arm, skin):                  # Meshy's marker ball for its bones
            bpy.data.objects.remove(o, do_unlink=True)
            continue
        for c in list(o.users_collection):
            c.objects.unlink(o)
        coll.objects.link(o)
    arm.name, skin.name = NAME, SKIN
    for pb in arm.pose.bones:
        pb.custom_shape = None
        pb.rotation_mode = 'QUATERNION'
    for act in [act for act in bpy.data.actions if act not in had]:
        act[FLAG] = True                          # Meshy's motions: cat_idle, cat_walk, cat_sajiao, dog_beifumo, dog_yaoweiba
        act.use_fake_user = True
        act.name = motion(act.name.split('.')[0])
    for mat in skin.data.materials:               # Meshy's paint, lit softly: no shine, no metal
        nt = mat.node_tree
        bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
        for sock in ('Metallic', 'Roughness', 'Normal'):
            for l in list(bsdf.inputs[sock].links):
                nt.links.remove(l)
        bsdf.inputs['Metallic'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 0.75
    stage()
    rest()
    specks()
    tag()
    seal()
    for name in MOTIONS:
        shut_in(bpy.data.actions[motion(name)])
    rest()
    return {'bones': len(arm.data.bones), 'points': len(skin.data.vertices), 'motions': sorted(a.name for a in bpy.data.actions if a.get(FLAG))}


def stage():
    sc = bpy.context.scene
    world = sc.world or bpy.data.worlds.new('JimmyWorld')
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    bg.inputs['Color'].default_value = (1, 1, 1, 1)
    bg.inputs['Strength'].default_value = 0.55
    sun = bpy.data.objects.get('JimmySun')
    if sun is None:
        sun = bpy.data.objects.new('JimmySun', bpy.data.lights.new('JimmySun', 'SUN'))
        sc.collection.objects.link(sun)
    sun.data.energy = 2.6
    sun.data.angle = math.radians(25)
    sun.rotation_euler = (math.radians(55), 0, math.radians(-35))


def arm():
    return bpy.data.objects[NAME]


def skin():
    return bpy.data.objects[SKIN]


def turn(bone, x=0.0, y=0.0, z=0.0):
    """Turn a bone by degrees about the room's axes (x: nod, y: roll, z: turn), from its rest."""
    a = arm()
    pb = a.pose.bones[bone]
    R = a.data.bones[bone].matrix_local.to_3x3()
    q = Matrix.Rotation(math.radians(z), 3, 'Z') @ Matrix.Rotation(math.radians(x), 3, 'X') @ Matrix.Rotation(math.radians(y), 3, 'Y')
    pb.rotation_quaternion = (R.inverted() @ q @ R).to_quaternion()
    return pb


def slide(bone, x=0.0, y=0.0, z=0.0):
    """Move a bone by room units along the room's axes, from its rest."""
    a = arm()
    pb = a.pose.bones[bone]
    pb.location = a.data.bones[bone].matrix_local.to_3x3().inverted() @ Vector((x, y, z))
    return pb


def rest():
    a = arm()
    if a.animation_data:
        a.animation_data.action = None
    for pb in a.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    if 'blink' in a:
        blink(0.0)
    bpy.context.scene.frame_set(1)


def jaw(shut=1.0):
    """0 = mouth wide open as modelled, 1 = shut. Only the jaw bone is turned: the seal and the mouth line follow it."""
    turn('jaw_01', x=SHUT * shut)


def curves(action, bone, what):
    """One bone's curves in a motion, by channel number."""
    path = 'pose.bones["%s"].%s' % (bone, what)
    return {fc.array_index: fc for layer in action.layers for strip in layer.strips
            for bag in strip.channelbags for fc in bag.fcurves if fc.data_path == path}


def shut_in(action, shut=1.0):
    """Meshy's motions hold his jaw as modelled, wide open (one pants it wider still). Hold it shut all through instead."""
    q = turn('jaw_01', x=SHUT * shut).rotation_quaternion.copy()
    for i, fc in curves(action, 'jaw_01', 'rotation_quaternion').items():
        for key in fc.keyframe_points:
            key.co.y = key.handle_left.y = key.handle_right.y = q[i]
        fc.update()


def play(name, frame=None):
    """Put him in a motion, at a frame if one is given."""
    a = arm()
    a.animation_data_create()
    act = bpy.data.actions[name]
    if 'blink' in a:
        blink(0.0)                                         # eyes open, unless the motion says otherwise
    a.animation_data.action = act
    if act.slots:
        a.animation_data.action_slot = act.slots[0]
    if frame is not None:
        bpy.context.scene.frame_set(int(frame))
    return act


def hold(name, frame):
    """Freeze him in one frame of a motion and let go of the motion, leaving a pose to build on."""
    play(name, frame)
    bpy.context.view_layer.update()
    a = arm()
    kept = {pb.name: pb.matrix_basis.copy() for pb in a.pose.bones}
    a.animation_data.action = None
    for pb in a.pose.bones:
        pb.matrix_basis = kept[pb.name]


def sit():
    """His everyday pose: sitting upright, mouth shut, paws down."""
    rest()
    hold(motion(SIT[0]), SIT[1])


def snap():
    """The pose he is in now, to keep."""
    bpy.context.view_layer.update()
    return {pb.name: pb.matrix_basis.copy() for pb in arm().pose.bones}


def put(pose):
    for pb in arm().pose.bones:
        pb.matrix_basis = pose[pb.name]


def facing():
    """The camera turn, in degrees, that looks him straight in the face as he is posed now."""
    bpy.context.view_layer.update()
    a = arm()
    was = a.data.bones['head'].matrix_local.to_3x3()
    now = a.pose.bones['head'].matrix.to_3x3()
    f = a.matrix_world.to_3x3() @ now @ was.inverted() @ Vector((0.0, -1.0, 0.0))
    return math.degrees(math.atan2(-f.x, -f.y))


def nudge(bone, x=0.0, y=0.0, z=0.0, face=0.0):
    """Turn a bone a bit further from where it is now, by degrees: x nods, y rolls, z turns.

    The axes are the room's, turned by `face` degrees so that they can be lined up with him (see facing())."""
    bpy.context.view_layer.update()
    pb = arm().pose.bones[bone]
    M = pb.matrix.copy()
    h = Matrix.Translation(M.to_translation())
    F = Matrix.Rotation(math.radians(-face), 4, 'Z')
    D = Matrix.Rotation(math.radians(z), 4, 'Z') @ Matrix.Rotation(math.radians(x), 4, 'X') @ Matrix.Rotation(math.radians(y), 4, 'Y')
    pb.matrix = h @ F @ D @ F.inverted() @ h.inverted() @ M
    bpy.context.view_layer.update()


def aim(bone, child, way):
    """Swing a bone about its joint so that it points from its own joint toward `way` (a direction in the room).

    `child` is the next bone down the limb: the two joints give the line that is pointed."""
    bpy.context.view_layer.update()
    a = arm()
    pb = a.pose.bones[bone]
    M = pb.matrix.copy()
    h = M.to_translation()
    now = (a.pose.bones[child].matrix.to_translation() - h).normalized()
    q = now.rotation_difference(Vector(way).normalized())
    pb.matrix = Matrix.Translation(h) @ q.to_matrix().to_4x4() @ Matrix.Translation(-h) @ M
    bpy.context.view_layer.update()


def blend(a, b, t):
    """A pose part-way from pose a to pose b."""
    out = {}
    for name in a:
        la, qa, _ = a[name].decompose()
        lb, qb, _ = b[name].decompose()
        out[name] = Matrix.Translation(la.lerp(lb, t)) @ qa.slerp(qb, t).to_matrix().to_4x4()
    return out


def ease(t):
    """Slow out, slow in."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def write(name, poses, blinks=None):
    """Save a list of poses, one a frame, as a motion of his own. Every bone is keyed on every frame, as Meshy does.

    `blinks`, if given, is how shut his eyes are on each frame."""
    a = arm()
    old = bpy.data.actions.get(name)
    if old:
        bpy.data.actions.remove(old)
    act = bpy.data.actions.new(name)
    act[FLAG + 'own'] = True
    act.use_fake_user = True
    a.animation_data_create()
    a.animation_data.action = act
    last = {}
    for i, pose in enumerate(poses):
        for pb in a.pose.bones:
            loc, rot, _ = pose[pb.name].decompose()
            if pb.name in last and last[pb.name].dot(rot) < 0:
                rot.negate()                          # the same turn written the other way round would spin him between frames
            last[pb.name] = rot.copy()
            pb.location, pb.rotation_quaternion = loc, rot
            pb.keyframe_insert('location', frame=i)
            pb.keyframe_insert('rotation_quaternion', frame=i)
        if blinks:
            a['blink'] = float(blinks[i])
            a.keyframe_insert('["blink"]', frame=i)
    a.animation_data.action = None
    if blinks:
        a['blink'] = 0.0
    return act


def tail_bones():
    """His tail's bones, from his rump to the tip."""
    bone, out = arm().data.bones['tail_01'], []
    while bone:
        out.append(bone.name)
        bone = bone.children[0] if bone.children else None
    return out


TALK = (1, 1, 0.75, 0.45, 0.3, 0.55, 0.85, 0.6, 0.35, 0.5, 0.9, 1, 0.7, 0.4, 0.25, 0.45, 0.8, 1, 1, 0.8, 0.5, 0.35, 0.6, 0.9, 1, 1)


def moves():
    """Write his own motions, each starting and ending in his sitting pose (the tail swish loops instead).

    30 frames a second, like Meshy's. His paw leaves the ground only in j5-wave. Returns each motion's length in frames."""
    sit()
    base, face = snap(), facing()

    def waving(swing):
        put(base)
        upper, lower, wrist, paw = WAVE
        aim('frontUpr.L', 'frontLwr.L', upper)
        aim('frontLwr.L', 'frontHorselink.L', (lower[0] + 0.30 * swing, lower[1], lower[2]))
        aim('frontHorselink.L', 'frontFoot.L', (wrist[0] + 0.45 * swing, wrist[1], wrist[2]))
        aim('frontFoot.L', 'frontToe.L', (paw[0] + 0.50 * swing, paw[1], paw[2]))
        nudge('head', y=5.0, face=face)
        return snap()

    def looking(turn_, tilt=0.0, nod=0.0):
        put(base)
        nudge('neck1', z=0.35 * turn_, face=face)
        nudge('head', z=0.65 * turn_, y=tilt, x=nod, face=face)
        return snap()

    def ears(left=0.0, right=0.0):
        put(base)
        for side, a in (('L', left), ('R', -right)):
            if a:
                nudge('ear.%s_01' % side, y=a * 0.5, face=face)
                nudge('ear.%s_02' % side, y=a * 0.5, face=face)
        return snap()

    def tail(phase, size=1.0):
        put(base)
        for k, bone in enumerate(tail_bones()):
            nudge(bone, z=size * (8 + 3.5 * k) * math.sin(phase - 0.7 * k), face=face)
        return snap()

    def talking(shut):
        put(base)
        turn('jaw_01', x=SHUT * shut)
        return snap()

    def to(a, b, frames):
        return [blend(a, b, ease((i + 1) / frames)) for i in range(frames)]

    made = {}
    # wave: paw up beside his head, three swings, paw down again
    out, mid, back = waving(1.0), waving(0.25), waving(-0.5)
    swings = [blend(back, out, 0.5 + 0.5 * math.sin(2 * math.pi * (i + 1) / 14)) for i in range(42)]
    made[OWN + 'wave'] = [base] * 4 + to(base, mid, 12) + swings + to(mid, base, 12) + [base] * 4
    # look round: to one side, to the other, back
    one, other = looking(30, 4), looking(-30, -4)
    made[OWN + 'look-round'] = [base] * 3 + to(base, one, 10) + [one] * 8 + to(one, other, 16) + [other] * 8 + to(other, base, 10) + [base] * 3
    # nod: yes
    down = looking(0, nod=14)
    made[OWN + 'nod'] = [base] * 2 + to(base, down, 6) + to(down, base, 6) + to(base, down, 6) + to(down, base, 7) + [base] * 3
    # ear twitch: one ear flicks out and back, twice
    flick = ears(left=30)
    made[OWN + 'ear-twitch'] = [base] * 3 + to(base, flick, 3) + to(flick, base, 4) + [base] * 3 + to(base, flick, 3) + to(flick, base, 4) + [base] * 6
    # tail swish: a slow wave down his tail, round once, so it loops
    made[OWN + 'tail-swish'] = [tail(2 * math.pi * i / 48) for i in range(48)]
    # talk: jaw only
    made[OWN + 'talk'] = [talking(v) for v in TALK]
    # sit: just sitting, for holding still
    made[OWN + 'sit'] = [base, base]
    # blink: lids down and up, eyes shut for two frames
    shut = (0, 0, 0.35, 0.8, 1, 1, 0.7, 0.3, 0, 0)
    made[OWN + 'blink'] = [base] * len(shut)
    for name, poses in made.items():
        write(name, poses, shut if name == OWN + 'blink' else None)
    rest()
    return {name: len(poses) for name, poses in made.items()}


def follow_jaw(owner, path):
    """Tie a value to how far his jaw bone is shut, so any motion that turns the jaw gets the seal for free."""
    owner.driver_remove(path)
    d = owner.driver_add(path).driver
    d.type = 'SCRIPTED'
    for v in list(d.variables):
        d.variables.remove(v)
    v = d.variables.new()
    v.name, v.type = 'w', 'TRANSFORMS'
    t = v.targets[0]
    t.id, t.bone_target = arm(), 'jaw_01'
    t.transform_type, t.rotation_mode, t.transform_space = 'ROT_W', 'QUATERNION', 'LOCAL_SPACE'
    d.expression = SEALED % abs(math.radians(SHUT))


def mouth_line(line, cover, fur):
    """As his mouth seals, paint a short dark mouth line in the middle and fur colour over the rest of the join.

    Meshy painted a dark rim round his open mouth. Shut, that rim would read as a mouth as wide as his open smile."""
    me = skin().data
    for name, values in (('j5_inside', line), ('j5_cover', cover)):
        att = me.attributes.get(name) or me.attributes.new(name, 'FLOAT', 'POINT')
        att.data.foreach_set('value', values.astype('f4'))
    for mat in me.materials:
        nt = mat.node_tree
        bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
        first = nt.nodes.get('j5_mix')
        paint = (first.inputs[6] if first else bsdf.inputs['Base Color']).links[0].from_socket
        for n in [n for n in nt.nodes if n.name.startswith('j5_')]:
            nt.nodes.remove(n)
        much = nt.nodes.new('ShaderNodeValue')
        much.name = 'j5_shut'
        for name, where_name, colour in (('j5_mix', 'j5_inside', LINE), ('j5_mix2', 'j5_cover', fur)):
            mix = nt.nodes.new('ShaderNodeMix')
            mix.name, mix.data_type = name, 'RGBA'
            where = nt.nodes.new('ShaderNodeAttribute')
            where.name, where.attribute_name = where_name, where_name
            times = nt.nodes.new('ShaderNodeMath')
            times.name, times.operation = name + '_times', 'MULTIPLY'
            nt.links.new(where.outputs['Fac'], times.inputs[0])
            nt.links.new(much.outputs[0], times.inputs[1])
            nt.links.new(times.outputs[0], mix.inputs[0])
            nt.links.new(paint, mix.inputs[6])
            mix.inputs[7].default_value = colour
            paint = mix.outputs[2]
        nt.links.new(paint, bsdf.inputs['Base Color'])
        follow_jaw(much.outputs[0], 'default_value')


def specks():
    """Delete the few loose bits Meshy's model came with (they float by his nose). Returns how many points went."""
    import bmesh
    me = skin().data
    bm = bmesh.new()
    bm.from_mesh(me)
    spot = {}
    for v in bm.verts:                                     # seams in the paint split points that sit on one spot
        spot.setdefault((round(v.co.x, 4), round(v.co.y, 4), round(v.co.z, 4)), []).append(v)
    twin = {v: vs for vs in spot.values() for v in vs}
    left, loose = set(bm.verts), []
    while left:
        v = left.pop()
        piece, todo = [v], [v]
        while todo:
            u = todo.pop()
            for w in [e.other_vert(u) for e in u.link_edges] + twin[u]:
                if w in left:
                    left.discard(w)
                    piece.append(w)
                    todo.append(w)
        if len(piece) < 200:
            loose += piece
    for v in loose:
        bm.verts.remove(v)
    bm.to_mesh(me)
    bm.free()
    return len(loose)


def tag():
    """Make his name tag move as one piece with the fur it rests on.

    Meshy's model has the tag joined on to his chest, and its rig shared the tag out between his neck and his front
    legs, each point differently. So a lifted paw or a bent neck stretched it like toffee. Now every point of the tag
    gets the same pull: the average of the fur just round it."""
    from mathutils import kdtree
    ob, me = skin(), skin().data
    n = len(me.vertices)
    co = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    paint = paint_at_points().astype('f8')
    hi, lo = paint.max(1), paint.min(1)
    d = np.maximum(hi - lo, 1e-6)
    r, g, b = paint[:, 0], paint[:, 1], paint[:, 2]
    hue = np.where(hi == r, ((g - b) / d) % 6, np.where(hi == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    sat = (hi - lo) / np.maximum(hi, 1e-6)
    below = co[:, 2] < TAG                                 # his eyes are gold too, but higher up
    sure = (hue > 0.07) & (hue < 0.16) & (sat > 0.4) & (hi > 0.55) & below
    middle = np.median(co[sure], axis=0)
    # gold, and the brown of its edge and ring: anything orange to yellow near the tag. His fur is plum, his collar red and blue.
    maybe = (hue > 0.03) & (hue < 0.20) & (sat > 0.25) & (hi > 0.15)
    gold = np.where(maybe & (np.linalg.norm(co - middle, axis=1) < 0.13))[0]
    near = (np.linalg.norm(co - middle, axis=1) < 0.15).nonzero()[0]
    tree = kdtree.KDTree(len(gold))
    for k, i in enumerate(gold):
        tree.insert(co[i].tolist(), k)
    tree.balance()
    ids = [int(i) for i in near if tree.find(co[i].tolist())[2] < 0.008]
    inside = np.zeros(n, bool)
    inside[ids] = True
    # Meshy's rig lets his front legs and jaw pull on the fur under his collar. Near the tag, take that pull away.
    off = np.linalg.norm(co - middle, axis=1)
    wrong = {g.index for g in ob.vertex_groups if g.name.startswith(('front', 'clavicle', 'jaw'))}
    for i in np.where(off < 0.22)[0]:
        keep = ease((off[i] - 0.10) / 0.12)
        v = me.vertices[int(i)]
        if not any(w.group in wrong for w in v.groups):
            continue
        new = {w.group: w.weight * (keep if w.group in wrong else 1.0) for w in v.groups}
        total = sum(new.values())
        if total < 1e-4:
            continue
        for g, w in new.items():
            ob.vertex_groups[g].add([int(i)], w / total, 'REPLACE')
    # the fur it rests on: points just off the tag that share a face with it, leaving out his bright collar
    under = set()
    for f in me.polygons:
        on = [inside[v] for v in f.vertices]
        if any(on) and not all(on):
            under.update(v for v, o in zip(f.vertices, on) if not o and sat[v] < 0.35)
    pull = {}
    for i in under:
        for w in me.vertices[i].groups:
            pull[w.group] = pull.get(w.group, 0.0) + w.weight
    total = sum(pull.values())
    for group in ob.vertex_groups:
        group.remove(ids)
    for g, w in pull.items():
        if w / total > 0.02:
            ob.vertex_groups[g].add(ids, w / total, 'REPLACE')
    return {'points': len(ids), 'fur_points_under_it': len(under),
            'moves_with': {ob.vertex_groups[g].name: round(w / total, 2) for g, w in pull.items() if w / total > 0.02}}


def linear(colour):
    """Paint as stored, to the numbers Blender mixes with."""
    return np.where(colour <= 0.04045, colour / 12.92, ((colour + 0.055) / 1.055) ** 2.4)


def eyes():
    """Give him eyelids, as paint: blink(0..1) brings a fur-coloured lid down over each eye, with a dark lash line.

    His eyes are painted on his head, so there is nothing to shut. Each point of an eye is given its height in the
    eye (0 at the bottom, 1 at the top, a little higher toward the corners so the lid's edge curves like a closed
    eye), and the paint is swapped for fur colour above the lid's edge. The amount is a number on his bones called
    'blink', so a motion can key it."""
    a, me = arm(), skin().data
    n = len(me.vertices)
    co = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3).astype('f8')
    paint = paint_at_points().astype('f8')
    hi, lo = paint.max(1), paint.min(1)
    d = np.maximum(hi - lo, 1e-6)
    r, g, b = paint[:, 0], paint[:, 1], paint[:, 2]
    hue = np.where(hi == r, ((g - b) / d) % 6, np.where(hi == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    sat = (hi - lo) / np.maximum(hi, 1e-6)
    head = np.where((co[:, 2] > FACE[0]) & (co[:, 1] < FACE[1]))[0]    # the front of his head
    hp, hue, sat, hi = co[head], hue[head], sat[head], hi[head]
    gold = (hue > EYEHUE[0]) & (hue < EYEHUE[1]) & (sat > 0.3) & (hi > 0.6) & (hp[:, 2] > FACE[2])  # the yellow of his eyes
    fur = (hue > 0.62) & (hue < 0.98) & (sat < 0.5) & (hi > 0.12)       # plum; an eye's dark rim and pupil are not
    nb = mouth5.neighbours(hp)
    middle = (a.matrix_world @ a.data.bones['head'].head_local).x       # his eyes sit either side of his head bone
    height = np.full(n, -1.0)
    found, lid = {}, []
    for side, mine in (('his right', hp[:, 0] < middle), ('his left', hp[:, 0] >= middle)):
        seeds = np.where(gold & mine)[0]
        seeds = seeds[np.linalg.norm(hp[seeds] - np.median(hp[seeds], axis=0), axis=1) < 0.14]   # leave out stray yellow
        eye = np.zeros(len(head), bool)
        eye[seeds] = True
        todo = list(seeds)
        while todo:                                         # everything joined to the yellow that is not fur
            i = todo.pop()
            for j in nb[i]:
                if not eye[j] and not fur[j]:
                    eye[j] = True
                    todo.append(j)
        for _ in range(4):                                  # and odd fur-coloured points caught inside
            for i in np.where(~eye)[0]:
                if len(nb[i]) and eye[nb[i]].mean() > 0.6:
                    eye[i] = True
        e = hp[eye]
        c = (e.min(0) + e.max(0)) / 2
        rx, rz = (e[:, 0].max() - e[:, 0].min()) / 2, (e[:, 2].max() - e[:, 2].min()) / 2
        across, up = (hp[:, 0] - c[0]) / rx, (hp[:, 2] - c[2]) / rz
        eye |= (across ** 2 + up ** 2 < 0.8) & (np.abs(hp[:, 1] - c[1]) < 0.08)      # and any odd point missed in the middle
        height[head[eye]] = ((up[eye] + 1) / 2 + EYE[1] * across[eye] ** 2) / (1 + EYE[1])
        near = np.array([len(nb[i]) > 0 and eye[nb[i]].any() for i in range(len(head))]) & ~eye
        ring = near.copy()
        for _ in range(3):                                  # a band of fur a few points wide round the eye, for the lid's colour
            ring |= np.array([len(nb[i]) > 0 and ring[nb[i]].any() for i in range(len(head))]) & ~eye
        ring &= fur & ~near
        lid.append(linear(paint[head[ring]]).mean(0))
        found[side] = {'points': int(eye.sum()), 'middle': [round(float(v), 3) for v in c], 'half_size': [round(float(rx), 3), round(float(rz), 3)]}
    att = me.attributes.get('j5_eye') or me.attributes.new('j5_eye', 'FLOAT', 'POINT')
    att.data.foreach_set('value', height.astype('f4'))
    fur = np.mean(lid, axis=0)
    if 'blink' not in a:
        a['blink'] = 0.0
    a.id_properties_ui('blink').update(min=0.0, max=1.0)
    for mat in me.materials:
        nt = mat.node_tree
        bsdf = next(k for k in nt.nodes if k.type == 'BSDF_PRINCIPLED')
        paint_in = bsdf.inputs['Base Color'].links[0].from_socket
        if paint_in.node.name.startswith('j5e_'):
            paint_in = nt.nodes['j5e_lid'].inputs[6].links[0].from_socket
        for k in [k for k in nt.nodes if k.name.startswith('j5e_')]:
            nt.nodes.remove(k)

        def node(kind, name, **settings):
            k = nt.nodes.new(kind)
            k.name = 'j5e_' + name
            for key, value in settings.items():
                setattr(k, key, value)
            return k

        def maths(name, op, x, y=None, z=None):
            k = node('ShaderNodeMath', name, operation=op)
            for socket, value in zip(k.inputs, (x, y, z)):
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    socket.default_value = value
                else:
                    nt.links.new(value, socket)
            return k.outputs[0]

        def ramp(name, x, lo_, hi_, a_, b_):
            k = node('ShaderNodeMapRange', name)
            nt.links.new(x, k.inputs[0])
            for i, value in zip((1, 2, 3, 4), (lo_, hi_, a_, b_)):
                k.inputs[i].default_value = value
            return k.outputs[0]

        amount = node('ShaderNodeValue', 'blink')
        drv = amount.outputs[0].driver_add('default_value').driver
        drv.type = 'SUM'
        var = drv.variables.new()
        var.type = 'SINGLE_PROP'
        var.targets[0].id = a
        var.targets[0].data_path = '["blink"]'
        where = node('ShaderNodeAttribute', 'where', attribute_name='j5_eye')
        edge = maths('edge', 'MULTIPLY_ADD', amount.outputs[0], -EYE[2], 1.0)       # the lid's edge: 1 (top) down to nearly 0
        off = maths('off', 'SUBTRACT', where.outputs['Fac'], edge)
        on = ramp('on', amount.outputs[0], 0.0, 0.08, 0.0, 1.0)
        lidded = maths('lidded', 'MULTIPLY', ramp('above', off, -0.03, 0.03, 0.0, 1.0), on)
        lash = maths('lash', 'MULTIPLY', ramp('near', maths('gap', 'ABSOLUTE', off), 0.0, EYE[3], 1.0, 0.0), on)
        for name, amount_, colour in (('lid', lidded, (float(fur[0]), float(fur[1]), float(fur[2]), 1.0)), ('line', lash, LINE)):
            mix = node('ShaderNodeMix', name, data_type='RGBA')
            nt.links.new(amount_, mix.inputs[0])
            nt.links.new(paint_in, mix.inputs[6])
            mix.inputs[7].default_value = colour
            paint_in = mix.outputs[2]
        nt.links.new(paint_in, bsdf.inputs['Base Color'])
    return found


def blink(amount=1.0):
    """0 = eyes open, 1 = shut."""
    a = arm()
    a['blink'] = float(amount)
    a.update_tag()


def seal(smile=None):
    """Give his skin a 'shut' shape that seals his mouth, blended in by the jaw bone over the last part of closing."""
    ob, a, me = skin(), arm(), skin().data
    if me.shape_keys and 'shut' in me.shape_keys.key_blocks:
        me.shape_keys.key_blocks['shut'].driver_remove('value')
        ob.shape_key_remove(me.shape_keys.key_blocks['shut'])
    n = len(me.vertices)
    co = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3).astype('f8')
    mx, half, front, low, high = MOUTH
    pick = np.where((np.abs(co[:, 0] - mx) < half) & (co[:, 1] < front) & (co[:, 2] > low) & (co[:, 2] < high))[0]
    rest()
    jaw(1.0)
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = ob.evaluated_get(dg)
    posed = ev.to_mesh()
    shut = np.zeros(n * 3, 'f4')
    posed.vertices.foreach_get('co', shut)
    ev.to_mesh_clear()
    shut = shut.reshape(-1, 3).astype('f8')
    # how the bones carry each point from rest to the shut pose, so a move in the shut pose can be put back into rest
    A = a.matrix_world
    carry = {g.index: np.array(A @ a.pose.bones[g.name].matrix @ a.data.bones[g.name].matrix_local.inverted() @ A.inverted())
             for g in ob.vertex_groups if g.name in a.pose.bones}
    jaws = {g.index for g in ob.vertex_groups if g.name in ('jaw_01', 'jaw')}
    M = np.zeros((len(pick), 4, 4))
    on_jaw = np.zeros(len(pick))
    for k, i in enumerate(pick):
        total = 0.0
        for g in me.vertices[i].groups:
            if g.group in carry:
                M[k] += g.weight * carry[g.group]
                total += g.weight
            if g.group in jaws:
                on_jaw[k] += g.weight
        M[k] = M[k] / total if total else np.eye(4)
    guess = np.einsum('kij,kj->ki', M[:, :3, :3], co[pick]) + M[:, :3, 3]
    off = float(np.abs(guess - shut[pick]).max())
    paint = paint_at_points()[pick].astype('f8')
    mouth5.WHITE, mouth5.FANGS = TEETH, FANGS
    moves, found = mouth5.seal(co[pick], shut[pick], paint, on_jaw, SMILE if smile is None else smile, SMALL)
    back = np.linalg.solve(M[:, :3, :3], moves[:, :, None])[:, :, 0]
    if not me.shape_keys:
        ob.shape_key_add(name='Basis')
    key = ob.shape_key_add(name='shut', from_mix=False)
    full = co.copy()
    full[pick] += back
    key.data.foreach_set('co', full.astype('f4').ravel())
    follow_jaw(key, 'value')
    middle = np.clip((WIDE[1] - found['across']) / (WIDE[1] - WIDE[0]), 0.0, 1.0)
    middle = middle * middle * (3 - 2 * middle)
    on_lips = np.clip(1.0 - found['lipdist'] / 0.014, 0.0, 1.0)
    line, cover = np.zeros(n), np.zeros(n)
    if THICK:
        on_edge = np.clip(1.0 - found['lipdist'] / THICK, 0.0, 1.0)
    else:
        on_edge = np.where(found['top'], EDGE[0], np.where(found['bottom'], EDGE[1], 0.0))
    line[pick] = middle * np.where(found['inside'], 1.0, on_edge)
    cover[pick] = (1.0 - middle) * np.where(found['inside'], 1.0, on_lips)
    ring = found['fur'] & (found['lipdist'] > 0.03) & (found['lipdist'] < 0.06)      # the fur just outside his lips
    fur = np.where(paint[ring] <= 0.04045, paint[ring] / 12.92, ((paint[ring] + 0.055) / 1.055) ** 2.4).mean(0)
    mouth_line(line, cover, (float(fur[0]), float(fur[1]), float(fur[2]), 1.0))
    eyes()                                                 # his eyelids are painted after his mouth, so redo them
    rest()
    return {'points': int(len(pick)), 'inside': int(found['inside'].sum()), 'top': int(found['top'].sum()),
            'bottom': int(found['bottom'].sum()), 'off': round(off, 5), 'most': [round(float(v), 3) for v in np.abs(moves).max(0)]}


def camera(center, scale, yaw=0.0, pitch=0.0, size=800, aspect=1.0):
    sc = bpy.context.scene
    for c in bpy.data.collections:
        if c.name in ALL:
            c.hide_render = c.name != NAME
    cam = sc.camera
    if cam is None:
        cam = bpy.data.objects.new('JimmyCam', bpy.data.cameras.new('JimmyCam'))
        sc.collection.objects.link(cam)
        sc.camera = cam
    sc.render.engine = 'BLENDER_EEVEE'
    sc.render.resolution_x, sc.render.resolution_y = int(size * aspect), size
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.view_settings.view_transform = 'Standard'
    a, p = math.radians(yaw), math.radians(pitch)
    look = Vector((math.sin(a) * math.cos(p), math.cos(a) * math.cos(p), -math.sin(p)))
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = scale
    cam.data.clip_end = 100
    cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()
    cam.location = Vector(center) - look * 30


def face_centre():
    """The middle of his face, wherever his head is now."""
    bpy.context.view_layer.update()
    a = arm()
    return a.matrix_world @ a.pose.bones['head'].matrix @ a.data.bones['head'].matrix_local.inverted() @ Vector(HEAD)


def body_centre():
    """The middle of all of him as he is posed now."""
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = skin().evaluated_get(dg)
    me = ev.to_mesh()
    co = np.zeros(len(me.vertices) * 3, 'f4')
    me.vertices.foreach_get('co', co)
    ev.to_mesh_clear()
    co = co.reshape(-1, 3)
    return Vector(((co.min(0) + co.max(0)) / 2).tolist())


def shot(path, face=False, yaw=0.0, pitch=0.0, size=600, front=False, centre=None):
    """Render him as he is posed now: face=True for a close-up of his face, else all of him.

    front=True measures yaw from straight in front of his face instead of from the room. Returns (centre, yaw) used,
    so that the frames of a clip can all be taken from one place."""
    bpy.context.view_layer.update()
    if front:
        yaw += facing()
    if centre is None:
        centre = face_centre() if face else body_centre()
    if face:
        camera(centre, 0.95, yaw, pitch, size)
    else:
        camera(centre, 2.3, yaw, pitch, size, aspect=1.1)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return centre, yaw


def clip(folder, name=None, poses=None, frames=None, face=False, yaw=0.0, size=380, front=True, start=None):
    """Render a motion (by name) or a list of poses as numbered frames in `folder`, all from one camera place.

    The camera is set up on `start` (a pose; default his sitting pose) so that it does not follow him about."""
    os.makedirs(folder, exist_ok=True)
    sit() if start is None else put(start)
    centre, turn_ = (face_centre() if face else body_centre()), yaw + (facing() if front else 0.0)
    if name:
        act = play(name)
        frames = frames or range(int(act.frame_range[0]), int(act.frame_range[1]) + 1)
    out = []
    for i, f in enumerate(frames if name else range(len(poses))):
        if name:
            bpy.context.scene.frame_set(int(f))
        else:
            put(poses[f])
        path = os.path.join(folder, '%03d.png' % i)
        shot(path, face=face, yaw=turn_, size=size, centre=centre)
        out.append(path)
    rest()
    return out


def paint_at_points():
    """The colour of Meshy's paint at each point of his skin (0 to 1, as stored), for finding teeth and tongue."""
    me = skin().data
    img = next(n.image for m in me.materials for n in m.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image and n.outputs['Color'].is_linked)
    small = img.copy()
    small.scale(1024, 1024)
    px = np.zeros(1024 * 1024 * 4, 'f4')
    small.pixels.foreach_get(px)
    bpy.data.images.remove(small)
    px = px.reshape(1024, 1024, 4)
    uv = np.zeros(len(me.loops) * 2, 'f4')
    me.uv_layers.active.data.foreach_get('uv', uv)
    uv = uv.reshape(-1, 2)
    lv = np.zeros(len(me.loops), 'i4')
    me.loops.foreach_get('vertex_index', lv)
    first = np.zeros(len(me.vertices), 'i4')
    first[lv[::-1]] = np.arange(len(lv))[::-1]
    u = np.clip((uv[first, 0] % 1.0) * 1024, 0, 1023).astype(int)
    v = np.clip((uv[first, 1] % 1.0) * 1024, 0, 1023).astype(int)
    return px[v, u, :3]


use('5')
