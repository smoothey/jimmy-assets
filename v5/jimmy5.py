"""Jimmy from Meshy, versions 5, 6 and 7 (JM-491), on Meshy's Quadruped Dog rig, in Blender.

One script builds each model; MODELS holds what differs and use('7') picks one. Version 7 is the one to use: the
straight-on cat made with his mouth closed, so his shut face is Meshy's own. He has no jaw to open, so mouth() gives
him a mouth that opens as a hollow with paint. The notes below were written for version 5 and still hold for 5 and 6.

flat_moves() carries every move and pose of the flat rig over from moves.json. eyes() paints his eyelids: 'blink'
shuts them to a sleepy dip, 'joy' turns that to a happy arch. raise_paw() lifts his paw from the shoulder, by way
of a tucked pose, to beside his face.

Version 5: the open-mouth standing cat from Meshy.

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
from mathutils import Matrix, Quaternion, Vector

import mouth5

HERE = os.path.dirname(os.path.abspath(__file__))
ALL = ('Jimmy', 'Jimmy2', 'Jimmy3', 'Jimmy4', 'JimmyMeshy', 'Jimmy5', 'Jimmy5ref', 'Jimmy6', 'Jimmy7')

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
        LIKE=None,
        POINT=None,                         # if set, where each part of his leg points when he holds a paw out to show you something (see raise_paw()), and
        POINT_PALM=0.0,                     # how far that paw is turned about its own length
        SAUCER=None,                        # if set, he has a small flying saucer to sit in (saucer()): its sizes and colours
        TAPER=None,                         # if set, his tail is slimmed and tapered (taper()): how much of its thickness is kept where it leaves his rump, and at its tip
        TUCK=None,                          # if set, where each part of his leg points half-way up: paw lifted in front of his chest, elbow bent
        LICK_TUCK=None,                     # if set, the tuck on the way to a lick: which way his forearm's end and his paw point, folded down at the wrist
        LICK=None,                          # if set, his paw held up in front of his mouth to lick (lick_paw()): where his wrist goes, in his own space, which way his forearm's end points, which way his paw points, and how far it is turned over (degrees)
        SHOULDER=None,                      # if set, the pull of each front leg on the skin round its shoulder is tidied (shoulders()): how far from the upper leg's bone its pull is left whole and where it ends, how far round the shoulder joint to work, and how many rounds of smoothing
        LID=None,                           # if set, a shut eye is a lid over an eyeball, not a patch (lids()): see MODELS['7']
        LIGHT=None,                         # if set, he is lit evenly about the way his face points (lit()); if not, by one lamp fixed in the room
        FEET=None,                          # if set, when he sits his paws are set flat on one level floor (ground()): the most a paw may be turned to flatten it, in degrees
        ALIVE=None,                         # if set, every move of his own is brought to life (alive()): how much of a head turn, tilt and nod his body takes up; how strongly and over how many frames a move is given wind-up and overshoot; how far his head drifts in a held pose, in degrees; how far his chest fills when he breathes, and how many frames a breath takes; and how long a move must be before he blinks in it
        SLEEVE=None,                        # if set, each front leg is cut free of his body from elbow to shoulder (free_legs()): how far from the leg's bone the leg ends, how far above the shoulder joint the cut stops, how flat the patch left on his body is (1 = the leg's own shape), how far each side of the elbow its bend is spread, how deep the leg's new back is beside half its width, and how much of the fur round the shoulder joint follows his collar bone
        CHEEKS=None,                        # if set, the fur of his head is held by his head, not his neck (head_fur()): how far out from the line of his neck the head's fur starts and is all head, and how far above his collar it starts and is all head
        BROW=None,                          # if set, his painted brows can move (browed()): how far a brow lifts, and how many degrees it tilts
        BULGE=None,                         # if set, how far his eyeballs dome out from his face (eyeballs()); Meshy models them nearly flat
        GRUMP=None,                         # if set, a frown also turns his mouth down: how far out from the middle its corners are, how far they drop, and that drop scaled for the corner on his right and on his left
        FLAT=None,                          # if set, where his left ear's two lower parts point when his ears go flat out sideways ('airplane ears'); his right ear mirrors it
        EARS=None,                          # if set, each ear's first bone pulls on the whole ear and nothing else (ears()): how thin his skin must be to count as ear, how wide the join where ear blends into head is, how far from the ear's line to look, how far in front of and behind the ear's top the ear may reach, how far below its first joint, how far each side of its second joint the ear's bend is spread, and how far up the ear's line it stands clear of his head
        TAIL=None,                          # if set, his tail is handled as a cat's (see tail_up(), lay_tail()): how far round him it lies as he sits (degrees from straight behind, toward his left), where along it the tip starts to lift, how high it lifts, the tail's radius, and how far the lifted tip leans away from him
        WRAP=None,                          # if set, his tail is laid round him as he sits: the way it leaves him and the way its tip points (degrees round from straight back, toward his left), and how far its root tips up
        JAW=True,                           # whether he has a jaw bone and an inside to his mouth
        OPEN=None,
        CLOSED=None,                        # if set, a closed-mouth model of the same cat beside his own, to match his shut face to
        FLOOF=None,                         # if set, a front leg only moves fur this near its bone, fading out by this far
        BEANS=None,                         # if set, the pink of his toe beans
        PALM=0.0,                           # degrees his raised paw is turned about its own length, to show its pads
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
        WIDE=(0.30, 0.55),
        THICK=0.022,
        LIKE=(0.11, 0.21, 0.035),           # matching his shut face: full pull within this far of his mouth, none beyond this, and how far above it the pull stops
        FLOOF=(0.065, 0.11),
        BEANS=(0.90, 0.36, 0.42, 1.0),
        PALM=90.0,
        SMALL=(0.55, 0.75, 0.25),            # a small chin tucked under his muzzle, like Ant's first Meshy cat
        SMILE=0.008,
        SIT=('dog_beifumo', 66),
        WAVE=((0.8, -0.2, -0.5), (0.9, -0.2, 0.3), (0.15, -0.15, 1.0), (0.05, -0.25, 1.0), (0.0, -0.45, 1.0)),   # shoulder first: his arm hinges at the top of his shoulder, elbow out
        CLOSED=None,                        # 'jimmy-closed.glb' is Meshy's model of the same cat with its mouth closed; matching his shut face to it warped his nose, so it is off (version 7 uses that model as its base instead)
    ),
    '7': dict(
        NAME='Jimmy7', SKIN='j7_skin', FLAG='j7', PRE='v7_', OWN='j7-', GLB=os.path.join('..', 'v7', 'jimmy.glb'),
        TAIL=(150.0, 0.5, 0.46, 0.065, 0.14),
        TAPER=(0.85, 0.45),                 # Meshy's tail is a thick sausage, as wide at the end as at his rump (Ant, 24 Nov: slimmer, tapered)
        SHOULDER=(0.16, 0.30, 0.50, 30),
        SLEEVE=(0.135, 0.07, 0.25, 0.07, 1.05, 0.8),
        CHEEKS=(0.18, 0.27, 0.03, 0.08),
        LID=dict(whole=0.92, out=1.15,      # how far out from the eye (as j5_eyem measures) his skin is laid afresh to lose the eye's rim, and where that has faded away
                 above=(1.15, 1.6),         # the same above the eye, where it reaches up to his brow: a shut lid runs smoothly down from the brow, and the hollow of the socket goes
                 swell=0.05,                # how far the lid stands out over the eyeball, at its middle
                 paint=(1.0, 1.15),         # where the lid's paint fades out
                 dip=0.2,                   # how far his brow comes down with the lid, as a share of its whole travel (not for a happy arch)
                 fold=(0.10, 0.007),        # the crease where the lids meet: its width (in eye heights) and depth
                 cheek=(0.022, 0.012),      # a full cheek lift: up, and out from his face
                 oh=0.012,                  # how far the corners of his mouth are drawn in for an 'o'
                 smirk=(0.006, 0.003),      # how far a wink lifts that corner of his mouth, and draws it out
                 lower=(-0.08, 0.14)),      # the lower lid before it rises: how far below the eye's lowest point its edge waits, and how curved it is (the curve of the eye's own lower rim)
        LIGHT=((55, 22, 1.9), (68, -40, 0.9)),   # a main lamp and a softer one from the other side, each: how far it leans from overhead, how far round from straight in front of his face, how bright
        FEET=50.0,
        ALIVE=dict(body=(0.18, 0.25, 0.20), sharp=(0.8, 3.0), sway=0.6, breath=(0.020, 96), blink=54),
        BROW=(0.04, 16.0),
        BULGE=0.045,
        GRUMP=(0.09, 0.03, 0.75, 1.35),
        FLAT=((0.93, 0.22, 0.28), (1.0, 0.30, 0.02)),
        EARS=(0.20, 0.05, 0.40, 0.06, 0.16, 0.10, 0.14),      # his ears are thick, cartoon ears: about 0.09 through
        JAW=False,                          # made with his mouth closed, so he has Meshy's own neat shut face and no jaw bone
        SHUT=0.0,
        HEAD=(0.003, -0.62, 0.31),
        MOUTH=(0.003, 0.26, -0.50, -0.11, 0.29),
        FACE=(0.13, -0.42, 0.18),
        TAG=0.12,
        EYEHUE=(0.06, 0.19),
        TEETH=(0.30, 0.55), FANGS=False, EDGE=(1.0, 1.0), WIDE=(0.3, 0.55), THICK=0.0, LIKE=None, CLOSED=None,
        FLOOF=(0.065, 0.11),
        BEANS=(0.90, 0.36, 0.42, 1.0),
        PALM=90.0,
        SMALL=(1.0, 1.0, 1.0), SMILE=0.0,
        SIT=('dog_beifumo', 66),
        # like a beckoning cat: shoulder out, upper arm out and forward, forearm straight up, paw curled forward
        WAVE=((0.8, -0.2, -0.5), (0.60, -0.40, 0.70), (0.25, -0.15, 1.0), (0.0, -0.35, 1.0), 0.0),
        TUCK=((0.30, -0.80, -0.50), (0.05, -0.45, 0.90), (0.0, -0.85, 0.5), 0.0),
        # showing you something: his paw held out to his left, a little forward and up. His upper arm stays low and
        # forward, as in the tuck, and it is his forearm that points: with his whole arm straight out sideways the
        # skin under it opened up, and a stiff arm is not a cat's
        POINT=((0.8, -0.2, -0.3), (0.35, -0.85, -0.25), (0.85, -0.45, 0.35), (0.92, -0.35, 0.20), 0.0),
        POINT_PALM=20.0,
        # grooming (JM-513; Ant, 25 Nov: "the paw needs to go in front of his mouth to lick it, Jimmy may need to move
        # his head to meet it"): his wrist just under and in front of his chin, a touch to his left, forearm up, and
        # his paw bent back at the wrist so its tip points at his mouth, and turned over 90 degrees about its length so
        # its pads face his mouth and we see the back of it (Ant, 12:46 and 12:58: "his toe beans are still facing
        # towards us"; of four turns tried, -90 hid them from the front and the side)
        LICK=((0.03, -0.50, -0.26), (0.0, -0.3, 0.95), (0.0, 0.95, 0.3), -90.0),
        # on the way up, his paw folds down at the wrist, as a cat's does, so its pads face his chest, not us (13:02:
        # "turn the paw over first"; of three folds tried, this one hid them from the front and the side)
        LICK_TUCK=((0.0, -0.5, 0.86), (0.0, 0.7, -0.7)),
        # his flying saucer, for the Wall's night sky (Ant, 24 Nov: saucer A of the mock-up, smaller, in colouring 2).
        # out: how far its edge is from its middle. room: how much wider than him its well is (a little less than
        # nothing: his tail tip and toes tuck inside its wall). rim: how high its lip stands above the floor he sits on
        SAUCER=dict(out=0.98, room=-0.02, rim=0.34, hull=(0.36, 0.26, 0.72), stripe=(0.16, 0.74, 0.70), trim=(0.95, 0.70, 0.22),
                    lights=((0.30, 0.92, 0.86), (1.0, 0.70, 0.25)), under=(0.45, 0.95, 0.90), well=(0.10, 0.30, 0.34)),
        OPEN=(0.095, 0.058, 0.045, 0.030, 0.030),  # his talking mouth: half its width, its height fully open, how deep it goes, how far his chin drops, and how far under his nose it starts
    ),
}


def use(model):
    """Pick which model the rest of this script works on."""
    globals().update(MODELS['5'])                           # model 5's settings are the defaults
    globals().update(MODELS[str(model)])
    return NAME


def motion(name):
    """The name one of Meshy's motions goes by for this model."""
    return PRE + name


LINE = (0.03, 0.02, 0.03, 1.0)    # the dark line of his shut mouth
# how sealed his mouth is, 0 to 1, from the jaw bone's own turn (w): nothing until 60% shut, all by fully shut
SEALED = 'clamp((2*acos(min(1,abs(w)))/%.4f-0.6)/0.4,0,1)'      # the gap is filled with SHUT, in radians
# his eyelids: how far past the eye the lid's patch reaches; how much the lid's edge arches as it comes down; where a
# shut eye's line lies (0 = the bottom of the eye, 1 = the top) and how much it dips in the middle; half the line's
# thickness; where a happy eye's line lies and how much it arches; and how far a frown slants the lid's edge
# down toward his nose
EYE = (1.24, 0.22, 0.32, 0.28, 0.065, 0.60, 0.40, 0.30)
LIDS = ('blink', 'blink_l', 'blink_r')   # numbers on his bones that shut both eyes, his left eye, his right eye
TALK = 'talk'                            # one that opens his mouth, on a model with no jaw
JOY = 'joy'                              # one that turns a shut eye from a sleepy dip to a happy arch
FROWN = 'frown'                          # one that slants his lids down toward his nose: narrowed, cross eyes
BROWS = ('brow_l', 'brow_r', 'tilt_l', 'tilt_r')   # and four for his brows: his left and right brow up (1) or down (-1), and each tilted, inner end down (1) or up (-1)
# Brow moves the flat rig does not have, added to its own: for each move, (how far through, (his left brow up, his
# right brow up, his left brow's tilt, his right brow's tilt)). A pose has one entry. Moves with nothing here keep
# still brows: a settle or an idle tail has nothing to say with them.
_UP = lambda v: (v, v, 0.0, 0.0)
_O = (0.0, 0.0, 0.0, 0.0)
BROW_MOVES = {
    'blink': [(0, _O), (0.45, _UP(-0.15)), (0.6, _UP(-0.15)), (1, _O)],                   # they dip a touch as his lids shut
    'sleepy': [(0, _O), (0.3, (-0.25, -0.25, -0.3, -0.3)), (0.67, (-0.3, -0.3, -0.3, -0.3)), (0.72, _UP(0.35)), (0.86, (-0.1, -0.1, -0.15, -0.15)), (1, _O)],   # heavy, outer ends drooping; up as he starts awake
    'wave': [(0, _O), (0.15, _UP(0.4)), (0.85, _UP(0.4)), (1, _O)],                       # a friendly lift while his paw is up
    'point': [(0, _O), (0.18, (0.35, 0.05, 0.0, 0.0)), (0.46, (0.35, 0.05, 0.0, 0.0)), (0.58, _UP(0.45)), (0.8, _UP(0.45)), (1, _O)],   # the brow on the side he looks to, then both as he looks back at you: "see?"
    'talk': [(0, _O), (0.06, _UP(0.25)), (0.2, _UP(0.05)), (0.38, _UP(0.35)), (0.48, _UP(0.1)), (0.56, _UP(0.3)), (0.75, _UP(0.1)), (1, _O)],      # lifting on the words he leans on
    'chatting': [(0, _O), (0.05, _UP(0.22)), (0.29, _UP(0.05)), (0.34, _UP(0.35)), (0.45, _UP(0.25)), (0.6, _UP(0.05)), (0.65, _UP(0.35)), (0.76, _UP(0.2)), (0.95, _UP(0.05)), (1, _O)],
    'thinking': [(0, _O), (0.2, (-0.2, 0.45, 0.6, 0.0)), (0.8, (-0.2, 0.45, 0.6, 0.0)), (1, _O)],   # one up, one down and slanted, as in the 'think' pose
    'nod': [(0, _O), (0.25, _UP(0.2)), (0.75, _UP(0.2)), (1, _O)],
    'found': [(0, _O), (0.15, _UP(0.5)), (0.35, _UP(0.5)), (0.55, _UP(0.2)), (0.75, _UP(0.2)), (1, _O)],   # up at the find, easing as he settles on it
    'glance-left': [(0, _O), (0.25, (-0.05, 0.35, 0.0, 0.0)), (0.75, (-0.05, 0.35, 0.0, 0.0)), (1, _O)],   # the brow on the side he looks to (the picture's left is his right)
    'glance-right': [(0, _O), (0.25, (0.35, -0.05, 0.0, 0.0)), (0.75, (0.35, -0.05, 0.0, 0.0)), (1, _O)],
    'turn-left': [(0, _O), (0.22, (0.0, 0.2, 0.0, 0.0)), (0.78, (0.0, 0.2, 0.0, 0.0)), (1, _O)],
    'turn-right': [(0, _O), (0.22, (0.2, 0.0, 0.0, 0.0)), (0.78, (0.2, 0.0, 0.0, 0.0)), (1, _O)],
    'look-round': [(0, _O), (0.2, _UP(0.3)), (0.8, _UP(0.3)), (1, _O)],
    'ear-twitch-left': [(0, _O), (0.3, (0.0, 0.3, 0.0, 0.0)), (0.6, (0.0, 0.15, 0.0, 0.0)), (1, _O)],      # a flick of the brow under the ear that twitched
    'ear-twitch-right': [(0, _O), (0.3, (0.3, 0.0, 0.0, 0.0)), (0.6, (0.15, 0.0, 0.0, 0.0)), (1, _O)],
    'ear-twitch': [(0, _O), (0.12, _O), (0.24, (0.3, 0.0, 0.0, 0.0)), (0.42, _O), (0.52, _O), (0.64, (0.3, 0.0, 0.0, 0.0)), (0.82, _O), (1, _O)],
    'tail-droop': [(0, _O), (0.3, (0.3, 0.3, -1.2, -1.2)), (0.8, (0.3, 0.3, -1.2, -1.2)), (1, _O)],        # sad: inner ends well up
    'on-it': [(0, _O), (0.3, (-0.3, -0.3, 0.8, 0.8)), (0.6, (-0.3, -0.3, 0.8, 0.8)), (1, _O)],             # set, for a moment: down, inner ends lowest, but short of cross
    'milestone': [(0, _O), (0.12, _UP(0.7)), (0.85, _UP(0.7)), (1, _O)],
    'pose-hover': [(0, _UP(0.25))],
    'pose-party': [(0, _UP(0.5))],
    'pose-dozy': [(0, (-0.3, -0.3, -0.3, -0.3))],
    'pose-shut': [(0, (-0.3, -0.3, -0.15, -0.15))],   # at rest, like his lids
    'pose-ready': [(0, _UP(0.35))],
    'pose-point': [(0, (0.35, 0.1, 0.0, 0.0))],
    'pose-saucer': [(0, _UP(0.2))],
    'pose-saucer-travel': [(0, _UP(0.3))],
    'point-left': [(0, _O), (0.18, (0.05, 0.35, 0.0, 0.0)), (0.46, (0.05, 0.35, 0.0, 0.0)), (0.58, _UP(0.45)), (0.8, _UP(0.45)), (1, _O)],
    'pose-point-left': [(0, (0.1, 0.35, 0.0, 0.0))],
    'pose-travel': [(0, _UP(0.2))],
    'pose-sunk': [(0, (0.3, 0.3, -1.0, -1.0))],                                           # ducked down, looking up: inner ends well up
    'slow-blink': [(0, _O), (0.4, (-0.12, -0.12, -0.12, -0.12)), (0.64, (-0.12, -0.12, -0.12, -0.12)), (1, _O)],   # soft, inner ends a touch up
    'yawn-stretch': [(0, _O), (0.3, _UP(0.4)), (0.66, _UP(0.4)), (0.8, _O), (1, _O)],   # up with the yawn
    'groom': [(0, _O), (0.16, _UP(-0.15)), (0.5, _UP(-0.15)), (0.6, (-0.1, -0.3, 0.0, 0.0)), (0.86, (-0.1, -0.3, 0.0, 0.0)), (1, _O)],   # intent; the one under the paw pushed down
}
BREATH = 'breath'                        # and one that fills his chest as he breathes in
SAD, OH = 'sad', 'oh'                    # and two for his mouth: its corners turned down (with no cross lids), and a small round 'o'
MOUTHS = (SAD, OH)
CHEEK = 'cheek'                          # and one that says how much his cheeks lift when his eyes shut: a little for a blink, hardly at all when sleepy
CHEEK_AS = 0.5                           # what that is unless a move says otherwise
CHEEK_MOVES = {'sleepy': 0.05, 'pose-dozy': 0.05, 'tail-droop': 0.1, 'pose-shut': 0.2, 'slow-blink': 0.3, 'yawn-stretch': 0.9, 'groom': 0.6}   # heavy lids, slack face; a content one lifts a little, a yawn squeezes
LOOPS = ('tail-swish', 'idle')           # moves that go round and round: the last frame leads into the first
BLINKS = (0.6, 1.0, 1.0, 0.85, 0.6, 0.35, 0.15, 0.04)  # a blink, frame by frame: shut in two frames, held, open in five. A real blink opens more slowly than it shuts
HOLD = 96                                # how many frames a pose to hold is written as: one breath, going round and round, so that he is never frozen
# Ear moves the flat rig does not have: for each move, a kind of ear (EAR_SHAPES) and how much of it at each point
# through the move. His ears are the biggest thing on him and say a lot.
EAR_MOVES = {
    'found': ('perk', [(0, 0), (0.15, 1), (0.45, 1), (0.7, 0.4), (1, 0)]),        # up and forward at the find
    'on-it': ('perk', [(0, 0), (0.3, 1), (0.7, 1), (1, 0)]),
    'point': ('perk', [(0, 0), (0.18, 1), (0.8, 0.8), (1, 0)]),
    'pose-point': ('perk', [(0, 0.8)]),
    'pose-saucer': ('perk', [(0, 0.4)]),
    'pose-saucer-travel': ('back', [(0, 0.35)]),           # ears a little back, as into a wind
    'point-left': ('perk', [(0, 0), (0.18, 1), (0.8, 0.8), (1, 0)]),
    'pose-point-left': ('perk', [(0, 0.8)]),
    'milestone': ('perk', [(0, 0), (0.12, 1), (0.85, 0.7), (1, 0)]),
    'sleepy': ('droop', [(0, 0), (0.3, 0.5), (0.66, 1), (0.74, 0.2), (0.86, 0.4), (1, 0)]),   # sinking with his lids, up as he starts awake
    'worried': ('back', [(0, 0), (0.16, 0.85), (0.8, 0.85), (1, 0)]),              # back and low, not out sideways as when cross
    'tail-droop': ('droop', [(0, 0), (0.3, 0.8), (0.8, 0.8), (1, 0)]),
    'oops': ('back', [(0, 0), (0.12, 1), (0.55, 1), (0.8, 0.4), (1, 0)]),
    'pose-dozy': ('droop', [(0, 0.6)]),
    'pose-hover': ('perk', [(0, 1.0)]),
    'pose-ready': ('perk', [(0, 1.0)]),
    'pose-travel': ('perk', [(0, 0.6)]),
    'pose-sunk': ('back', [(0, 0.8)]),
    'slow-blink': ('droop', [(0, 0), (0.4, 0.3), (0.7, 0.3), (1, 0)]),   # ears relax
    'yawn-stretch': ('back', [(0, 0), (0.35, 0.7), (0.66, 0.7), (0.8, 0), (1, 0)]),
}
# each kind, as degrees for his left ear: tipped forward, rolled out sideways, turned (his right ear mirrors it)
EAR_SHAPES = {'perk': (12.0, -7.0, 0.0), 'droop': (-4.0, 24.0, 0.0), 'back': (-22.0, 10.0, 0.0)}
# His mouth, move by move: (corners down, 'o') at moments through the move, as BROW_MOVES does his brows
MOUTH_MOVES = {
    'worried': [(0, (0, 0)), (0.18, (0.8, 0)), (0.8, (0.8, 0)), (1, (0, 0))],
    'tail-droop': [(0, (0, 0)), (0.3, (0.6, 0)), (0.8, (0.6, 0)), (1, (0, 0))],                  # sad
    'oops': [(0, (0, 0)), (0.07, (0, 1)), (0.22, (0.6, 0)), (0.5, (0.6, 0)), (0.68, (0.3, 0)), (0.86, (0.3, 0)), (1, (0, 0))],   # 'o', a wince, then sheepish
    'found': [(0, (0, 0)), (0.08, (0, 0.8)), (0.3, (0, 0.8)), (0.42, (0, 0)), (1, (0, 0))],     # 'o' at the find
    'milestone': [(0, (0, 0)), (0.12, (0, 0.6)), (0.3, (0, 0.6)), (0.42, (0, 0)), (1, (0, 0))],
    'talk': [(0, (0, 0)), (0.3, (0, 0)), (0.38, (0, 0.9)), (0.44, (0, 0)), (1, (0, 0))],        # one of his words is an 'o'
    'chatting': [(0, (0, 0)), (0.29, (0, 0)), (0.34, (0, 0.8)), (0.39, (0, 0)), (0.6, (0, 0)), (0.65, (0, 0.9)), (0.69, (0, 0)), (1, (0, 0))],
    'pose-held': [(0, (0, 0.5))],                                                                # picked up: a small 'o'
}
# Moves written afresh for him, in the flat rig's own words (see flat_moves()), where the flat rig's version did not
# stand up in the round (review of 24 Nov, suggestions 8 to 13): three happy moves that were one picture, a worried
# face that smiled, an oops that read as a blink, a found that read smug, a sleepy that was over in a second, and an
# on-it that was a nod. Times are parts of the whole move; 'ms' is how long it lasts.
def _both(part, keys):
    return {part + '-left': keys, part + '-right': keys}


REWORK = {
    # happy: a quick squint with his head tipped, and his tail tip up
    'happy': {'ms': 1300, 'tracks': dict(
        _both('closed', [[0, {'o': 0}], [0.12, {'o': 1}], [0.62, {'o': 1}], [0.8, {'o': 0}], [1, {'o': 0}]]),
        **_both('brow', [[0, {}], [0.14, {'dyh': -0.35}], [0.62, {'dyh': -0.35}], [0.82, {}], [1, {}]]),
        **_both('ear', [[0, {}], [0.16, {'sy': 0.05}], [0.62, {'sy': 0.05}], [0.84, {}], [1, {}]]),
        **{'head': [[0, {}], [0.2, {'r': 7, 'dy': -1}], [0.62, {'r': 7, 'dy': -1}], [0.86, {}], [1, {}]],
           'tail-mid': [[0, {}], [0.22, {'r': 5}], [0.7, {'r': 5}], [1, {}]],
           'tail-tip': [[0, {}], [0.24, {'r': 18}], [0.7, {'r': 18}], [1, {}]]})},
    # done: one firm nod, then a slow, satisfied shutting of his eyes, tail tip up
    'done': {'ms': 1500, 'tracks': dict(
        _both('closed', [[0, {'o': 0}], [0.3, {'o': 0}], [0.46, {'o': 1}], [0.72, {'o': 1}], [0.9, {'o': 0}], [1, {'o': 0}]]),
        **_both('brow', [[0, {}], [0.3, {}], [0.46, {'dyh': -0.25}], [0.72, {'dyh': -0.25}], [0.9, {}], [1, {}]]),
        **{'head': [[0, {}], [0.1, {'dy': -0.8}], [0.24, {'dy': 3.4}], [0.4, {'dy': 0.4}], [0.55, {}], [1, {}]],
           'tail-mid': [[0, {}], [0.35, {'r': 6}], [0.8, {'r': 6}], [1, {}]],
           'tail-tip': [[0, {}], [0.4, {'r': 16}], [0.8, {'r': 16}], [1, {}]]})},
    # milestone: the big one. A crouch, a hop, a landing and a smaller hop, eyes wide, then a happy squint; ears and tail up
    'milestone': {'ms': 2000, 'tracks': dict(
        _both('closed', [[0, {'o': 0}], [0.36, {'o': 0}], [0.48, {'o': 1}], [0.8, {'o': 1}], [0.93, {'o': 0}], [1, {'o': 0}]]),
        **{'body': [[0, {}], [0.07, {'sy': -0.03}], [0.17, {'dy': -9}], [0.27, {'sy': -0.03}], [0.37, {'dy': -5}], [0.46, {'sy': -0.02}], [0.56, {}], [1, {}]],
           'head': [[0, {}], [0.07, {'dy': 1.5}], [0.17, {'dy': -2.5}], [0.3, {'dy': 1}], [0.45, {'r': 5}], [0.8, {'r': 5}], [1, {}]],
           'tail': [[0, {}], [0.15, {'r': -12}], [0.85, {'r': -12}], [1, {}]],
           'tail-mid': [[0, {}], [0.18, {'r': 10}], [0.85, {'r': 10}], [1, {}]],
           'tail-tip': [[0, {}], [0.2, {'r': 24}], [0.85, {'r': 24}], [1, {}]]})},
    # worried: head ducked and looking from side to side, ears back (EAR_MOVES), mouth down (MOUTH_MOVES), tail drawn in.
    # His brows say most of it: lifted, and tilted hard, inner ends up (22 degrees here; BROW_FLIP turns the flat
    # rig's way over), with the brow on the side he is looking to a little higher. At 10 degrees, as first written,
    # the arch of his painted brows hid the tilt and he looked no more than mild
    'worried': {'ms': 2200, 'tracks': {
        'head': [[0, {}], [0.16, {'dy': 3.2, 'r': -3}], [0.36, {'dy': 3.2, 'r': -3, 'dx': 1.2}], [0.58, {'dy': 3.2, 'r': -3, 'dx': -1.2}], [0.8, {'dy': 3.2, 'r': -3}], [1, {}]],
        'brow-left': [[0, {}], [0.15, {'dyh': -0.4, 'r': 22}], [0.3, {'dyh': -0.3, 'r': 22}], [0.44, {'dyh': -0.3, 'r': 22}], [0.52, {'dyh': -0.52, 'r': 22}], [0.7, {'dyh': -0.52, 'r': 22}], [0.8, {'dyh': -0.4, 'r': 22}], [1, {}]],
        'brow-right': [[0, {}], [0.15, {'dyh': -0.4, 'r': -22}], [0.3, {'dyh': -0.52, 'r': -22}], [0.44, {'dyh': -0.52, 'r': -22}], [0.52, {'dyh': -0.3, 'r': -22}], [0.7, {'dyh': -0.3, 'r': -22}], [0.8, {'dyh': -0.4, 'r': -22}], [1, {}]],
        'tail': [[0, {}], [0.25, {'r': 8}], [0.8, {'r': 8}], [1, {}]],
        'tail-tip': [[0, {}], [0.3, {'r': -10}], [0.8, {'r': -10}], [1, {}]]}},
    # oops: a start, a wince (head ducked and tipped, one eye squeezed shut, the other half), then a sheepish look
    'oops': {'ms': 1800, 'tracks': {
        'body': [[0, {}], [0.07, {'dy': -1.5}], [0.2, {}], [1, {}]],
        'head': [[0, {}], [0.07, {'dy': -2.5}], [0.24, {'dy': 3.2, 'r': 5}], [0.5, {'dy': 3.2, 'r': 5}], [0.66, {'dy': 1.2, 'r': 3}], [0.86, {'dy': 1.2, 'r': 3}], [1, {}]],
        'brow-left': [[0, {}], [0.06, {'dyh': -0.5}], [0.2, {'dyh': 0.25, 'r': 9}], [0.5, {'dyh': 0.25, 'r': 9}], [0.66, {'dyh': -0.25, 'r': -16}], [0.86, {'dyh': -0.25, 'r': -16}], [1, {}]],     # up in a start; down and pinched in the wince; then sheepish, inner ends up
        'brow-right': [[0, {}], [0.06, {'dyh': -0.5}], [0.2, {'dyh': 0.25, 'r': -9}], [0.5, {'dyh': 0.25, 'r': -9}], [0.66, {'dyh': -0.25, 'r': 16}], [0.86, {'dyh': -0.25, 'r': 16}], [1, {}]],
        'lid-left': [[0, {'dyh': 0}], [0.12, {'dyh': 0}], [0.24, {'dyh': 1.05}], [0.5, {'dyh': 1.05}], [0.64, {'dyh': 0}], [1, {'dyh': 0}]],
        'lid-right': [[0, {'dyh': 0}], [0.12, {'dyh': 0}], [0.24, {'dyh': 0.5}], [0.5, {'dyh': 0.5}], [0.64, {'dyh': 0}], [1, {'dyh': 0}]]}},
    # found: a pop (head up, ears up, eyes wide, a small 'o'), then a happy squint with his head tipped
    'found': {'ms': 1500, 'tracks': dict(
        _both('closed', [[0, {'o': 0}], [0.46, {'o': 0}], [0.58, {'o': 1}], [0.8, {'o': 1}], [0.93, {'o': 0}], [1, {'o': 0}]]),
        **{'head': [[0, {}], [0.1, {'dy': -2}], [0.26, {'dy': 0.6}], [0.4, {}], [0.5, {'r': 4}], [0.8, {'r': 4}], [1, {}]],
           'body': [[0, {}], [0.1, {'dy': -1.2}], [0.26, {}], [1, {}]]})},
    # sleepy: his lids sink in steps, his head nods lower and lower, then jerks back up
    'sleepy': {'ms': 3600, 'tracks': dict(
        _both('lid', [[0, {'dyh': 0}], [0.12, {'dyh': 0.38}], [0.2, {'dyh': 0.3}], [0.34, {'dyh': 0.62}], [0.42, {'dyh': 0.52}], [0.56, {'dyh': 0.92}],
                      [0.62, {'dyh': 1.05}], [0.68, {'dyh': 1.05}], [0.72, {'dyh': 0.12}], [0.84, {'dyh': 0.4}], [1, {'dyh': 0}]]),
        **{'head': [[0, {}], [0.2, {'dy': 1}], [0.42, {'dy': 2.4}], [0.56, {'dy': 3.6}], [0.67, {'dy': 5.2, 'r': 3}], [0.72, {'dy': -1}], [0.84, {'dy': 0.8}], [1, {}]]})},
    # on-it: a sharp, set nod (a small lift first), ears forward and brows set
    'on-it': {'ms': 800, 'tracks': {
        'head': [[0, {}], [0.14, {'dy': -0.9}], [0.36, {'dy': 3.6}], [0.56, {'dy': 0.5}], [0.75, {}], [1, {}]]}},
    # --- review of 24 Nov, suggestions 14 to 17 ---
    # wave: paw up, three wide beats, head tipped toward the paw and chest leaning a little away, happy eyes as it comes down
    'wave': {'ms': 2300, 'tracks': dict(
        _both('closed', [[0, {'o': 0}], [0.64, {'o': 0}], [0.74, {'o': 1}], [0.9, {'o': 1}], [0.97, {'o': 0}], [1, {'o': 0}]]),
        **_both('ear', [[0, {}], [0.15, {'sy': 0.05}], [0.85, {'sy': 0.05}], [1, {}]]),
        **{'paw': [[0, {}], [0.2, {'r': -2}], [0.31, {'r': 17}], [0.42, {'r': -4}], [0.53, {'r': 16}], [0.64, {'r': -3}], [0.74, {'r': 13}], [0.83, {'r': 3}], [1, {}]],
           'head': [[0, {}], [0.2, {'r': 4}], [0.8, {'r': 4}], [1, {}]],
           'tail-tip': [[0, {}], [0.25, {'r': 12}], [0.8, {'r': 12}], [1, {}]]})},
    # ear twitches: three times the flat rig's, so they show at badge size
    'ear-twitch-left': {'ms': 600, 'tracks': {'ear-left': [[0, {}], [0.16, {'r': -26}], [0.36, {}], [0.54, {'r': -17}], [0.78, {}], [1, {}]]}},
    'ear-twitch-right': {'ms': 600, 'tracks': {'ear-right': [[0, {}], [0.16, {'r': 26}], [0.36, {}], [0.54, {'r': 17}], [0.78, {}], [1, {}]]}},
    # settle: he sinks on to his haunches and shifts his weight from one shoulder to the other
    'settle': {'ms': 1300, 'tracks': {
        'body': [[0, {}], [0.2, {'sy': -0.055, 'r': 3}], [0.42, {'sy': -0.05, 'r': 8}], [0.64, {'sy': -0.025, 'r': -6.5}], [0.82, {'sy': 0.01, 'r': 2.5}], [1, {}]],
        'head': [[0, {}], [0.2, {'dy': 2.2}], [0.42, {'dy': 1.2, 'r': -5}], [0.64, {'dy': 0.4, 'r': 4}], [0.85, {'r': -1}], [1, {}]],
        'tail-tip': [[0, {}], [0.5, {'r': 9}], [0.8, {'r': 3}], [1, {}]]}},
    # thinking: head tipped and lifted, looking up and away; one ear turned; the tip of his tail ticking
    'thinking': {'ms': 2200, 'tracks': {
        'head': [[0, {}], [0.2, {'r': -9, 'dy': -2.2, 'dx': 1.2}], [0.5, {'r': -10, 'dy': -2.4, 'dx': 1.4}], [0.8, {'r': -9, 'dy': -2.2, 'dx': 1.2}], [1, {}]],
        'ear-left': [[0, {}], [0.25, {'r': -12}], [0.8, {'r': -12}], [1, {}]],
        'tail-tip': [[0, {}], [0.2, {'r': 10}], [0.35, {'r': -4}], [0.5, {'r': 10}], [0.65, {'r': -4}], [0.8, {'r': 8}], [1, {}]]}},
    # puzzled: a tilt one way, a beat, then the other way (tilt is the one-way one)
    'puzzled': {'ms': 2300, 'tracks': {
        'head': [[0, {}], [0.15, {'r': 11, 'dy': -0.6}], [0.42, {'r': 11, 'dy': -0.6}], [0.58, {'r': -9, 'dy': 0.6}], [0.84, {'r': -9, 'dy': 0.6}], [1, {}]],
        'brow-left': [[0, {}], [0.15, {'dyh': -0.6}], [0.42, {'dyh': -0.6}], [0.58, {'dyh': 0.12, 'r': 11}], [0.84, {'dyh': 0.12, 'r': 11}], [1, {}]],      # one brow up, the other down a touch and slanted, inner end down: a quizzical look
        'brow-right': [[0, {}], [0.15, {'dyh': 0.12, 'r': -11}], [0.42, {'dyh': 0.12, 'r': -11}], [0.58, {'dyh': -0.6}], [0.84, {'dyh': -0.6}], [1, {}]],
        'ear-right': [[0, {}], [0.16, {'r': 16}], [0.42, {'r': 16}], [0.58, {}], [1, {}]],
        'ear-left': [[0, {}], [0.42, {}], [0.58, {'r': -16}], [0.84, {'r': -16}], [1, {}]],
        'tail-tip': [[0, {}], [0.2, {'r': 10}], [0.84, {'r': 10}], [1, {}]]}},
    # the tail moves, each with a rhythm of its own. flick: one sharp snap of the tip
    'tail-flick': {'ms': 700, 'tracks': {
        'tail-mid': [[0, {}], [0.14, {'r': -2}], [0.34, {'r': 9}], [0.56, {'r': -3}], [0.78, {'r': 1}], [1, {}]],
        'tail-tip': [[0, {}], [0.1, {'r': -6}], [0.3, {'r': 24}], [0.5, {'r': -9}], [0.72, {'r': 5}], [1, {}]]}},
    # lash: cross. Four hard thumps of the whole end of his tail, quick and even
    'tail-lash': {'ms': 1700, 'tracks': {
        'tail': [[0, {}], [0.1, {'r': 3}], [0.9, {'r': 3}], [1, {}]],
        'tail-mid': [[0, {}], [0.1, {'r': 10}], [0.2, {'r': -8}], [0.3, {'r': 10}], [0.4, {'r': -8}], [0.5, {'r': 10}], [0.6, {'r': -8}], [0.7, {'r': 10}], [0.8, {'r': -8}], [1, {}]],
        'tail-tip': [[0, {}], [0.12, {'r': 22}], [0.22, {'r': -18}], [0.32, {'r': 22}], [0.42, {'r': -18}], [0.52, {'r': 22}], [0.62, {'r': -18}], [0.72, {'r': 22}], [0.82, {'r': -16}], [1, {}]],
        'brow-left': [[0, {}], [0.1, {'dyh': 0.3, 'r': 8}], [0.85, {'dyh': 0.3, 'r': 8}], [1, {}]],
        'brow-right': [[0, {}], [0.1, {'dyh': 0.3, 'r': -8}], [0.85, {'dyh': 0.3, 'r': -8}], [1, {}]]}},
    # sway: content. One slow wave that runs down his tail and back, the tip last
    'tail-sway': {'ms': 5200, 'tracks': {
        'tail': [[0, {}], [0.28, {'r': 4}], [0.66, {'r': -2}], [1, {}]],
        'tail-mid': [[0, {}], [0.36, {'r': 9}], [0.74, {'r': -6}], [1, {}]],
        'tail-tip': [[0, {}], [0.44, {'r': 16}], [0.82, {'r': -12}], [1, {}]]}},
    # talk: the words differ (wide, small, an 'o' in MOUTH_MOVES), his head marks the ones he leans on, and he blinks
    'talk': {'ms': 1800, 'tracks': dict(
        _both('lid', [[0, {'dyh': 0}], [0.56, {'dyh': 0}], [0.61, {'dyh': 1.05}], [0.65, {'dyh': 1.05}], [0.72, {'dyh': 0}], [1, {'dyh': 0}]]),
        **{'mouth-talk': [[0, {'o': 0}], [0.06, {'o': 1}], [0.14, {'o': 0.25}], [0.22, {'o': 0.8}], [0.3, {'o': 0}], [0.46, {'o': 0}], [0.54, {'o': 1}],
                          [0.62, {'o': 0.35}], [0.7, {'o': 0.9}], [0.78, {'o': 0.2}], [0.86, {'o': 0.6}], [0.94, {'o': 0}], [1, {'o': 0}]],
           'head': [[0, {}], [0.06, {'dy': 1.3, 'r': 1.5}], [0.16, {'dy': 0.2, 'r': 2}], [0.24, {'dy': 1}], [0.38, {'dy': -0.5, 'r': -2.5}], [0.54, {'dy': 1.5, 'r': -2}],
                    [0.64, {'dy': 0.3}], [0.72, {'dy': 1.1, 'r': 1.5}], [0.86, {'dy': 0.3, 'r': 1}], [1, {}]]})},
    # chatting: a longer run of talk with a pause in it, his head tipping from one side to the other
    'chatting': {'ms': 2800, 'tracks': {
        'mouth-talk': [[0, {'o': 0}], [0.04, {'o': 0.9}], [0.09, {'o': 0.2}], [0.14, {'o': 1}], [0.19, {'o': 0.3}], [0.24, {'o': 0.7}], [0.29, {'o': 0}], [0.4, {'o': 0}],
                       [0.45, {'o': 1}], [0.5, {'o': 0.25}], [0.55, {'o': 0.8}], [0.6, {'o': 0}], [0.7, {'o': 0}], [0.75, {'o': 0.9}], [0.8, {'o': 0.3}], [0.85, {'o': 1}],
                       [0.9, {'o': 0.4}], [0.95, {'o': 0}], [1, {'o': 0}]],
        'head': [[0, {}], [0.04, {'dy': 1.2, 'r': 3}], [0.14, {'dy': 1.4, 'r': 4}], [0.29, {'dy': 0.2, 'r': 3}], [0.4, {'dy': -0.4, 'r': -1}], [0.45, {'dy': 1.3, 'r': -4}],
                 [0.6, {'dy': 0.3, 'r': -4}], [0.7, {'dy': -0.3, 'r': 1}], [0.75, {'dy': 1.4, 'r': 3}], [0.85, {'dy': 1.2, 'r': 2}], [0.95, {'dy': 0.2}], [1, {}]],
        'tail-tip': [[0, {}], [0.3, {'r': 8}], [0.6, {'r': -3}], [0.85, {'r': 8}], [1, {}]]}},
}
# Poses to hold, written afresh for him (review of 24 Nov, suggestions 19 to 21). In the flat rig four of them differ
# from sitting by an ear or a tail tip, which is lost on a body; 'sunk' slides him out of his frame; 'ready' lifts a
# paw, and he lifts a paw only to wave or point. 'body p' pitches his chest forward, 'body hang' lets his front paws
# hang from the wrist, and the party hat is a thing of its own (party_hat()), shown by the number HAT on his bones.
POSE_REWORK = {
    # hover: the pointer is over him. Alert: head up, ears forward (EAR_MOVES), brows up, tail tip up
    'hover': {'head': {'dy': -2.4}, 'body': {'p': 4}, 'tail-tip': {'r': 14}},
    # travel: on his way to something on the page. Tail carried up, ears forward, leaning in to it
    'travel': {'head': {'dy': -0.8}, 'body': {'p': 5}, 'tail': {'r': 6}, 'tail-mid': {'r': 14}, 'tail-tip': {'r': 22}},
    # held: picked up by the scruff. Off the floor, head tucked, ears out, brows up, front paws and tail hanging
    'held': {'body': {'dy': -22, 'hang': 1}, 'head': {'dy': 1.5}, 'ear-left': {'r': -14}, 'ear-right': {'r': 14},
             'brow-left': {'dyh': -0.4}, 'brow-right': {'dyh': -0.4}, 'tail': {'r': -5}, 'tail-mid': {'r': -9}, 'tail-tip': {'r': -14}},
    # sunk: ducked right down behind his paws, ears back, looking up
    'sunk': {'body': {'sy': -0.15, 'p': 22}, 'head': {'dy': -4.5}, 'tail-tip': {'r': -8}},
    # party: his hat, brows up, head tipped, tail tip up
    'party': {'hat': {'o': 1}, 'head': {'r': 4}, 'tail-mid': {'r': 6}, 'tail-tip': {'r': 16}},
    # ready: all four paws down. Leaning in, head up, ears forward, tail tip up
    'ready': {'body': {'p': 7}, 'head': {'dy': -2.2}, 'tail-tip': {'r': 14}},
}
# His pupils (review of 24 Nov, suggestion 7). His eyes are paint, so they could not look anywhere. gaze() lifts
# each eye's paint into a picture of its own that can slide inside the eye; these two numbers on his bones slide it.
GAZE = ('look_x', 'look_y')              # toward his left (1) or his right (-1), and up (1) or down (-1)
GAZE_REACH = ((0.10, 0.25), (0.22, 0.18))  # how far each pupil goes at 1, in half-widths of its eye: across (toward his nose, away from it), then up and down.
#                                          His pupils are big and sit low and a little toward his nose. Toward his nose the rim is nearly straight
#                                          and the pupil almost touches it, so it goes only as far as it stays whole. Away, it stops a little short
#                                          of the far rim: jammed against it (0.30) it read as a half moon, not a disc. Up, it tucks a little under
#                                          the curved rim, as an eye does; down, it goes under the lower rim
GAZE_BALL = 0.7                          # the pupil is made narrower the way it has gone, as a disc is on a ball of this radius (in half-widths of his eye). 1 is true to his eye's size and too slight to see
GAZE_SPAN = 1.25                         # how much of the face round each eye its picture holds, in half-widths of the eye
# Where he looks, move by move: (how far through, (to his left, up)), eased between, as MOUTH_MOVES does his mouth.
# His eyes go first and his head follows. The flat rig's left is the picture's left, which is his right.
GAZE_MOVES = {
    'glance-left': [(0, (0, 0)), (0.12, (-1, 0)), (0.72, (-1, 0)), (0.88, (0, 0)), (1, (0, 0))],
    'glance-right': [(0, (0, 0)), (0.12, (1, 0)), (0.72, (1, 0)), (0.88, (0, 0)), (1, (0, 0))],
    'turn-left': [(0, (0, 0)), (0.1, (-1, 0)), (0.3, (-0.3, 0)), (0.74, (-0.3, 0)), (0.84, (0.6, 0)), (1, (0, 0))],
    'turn-right': [(0, (0, 0)), (0.1, (1, 0)), (0.3, (0.3, 0)), (0.74, (0.3, 0)), (0.84, (-0.6, 0)), (1, (0, 0))],
    'look-round': [(0, (0, 0)), (0.1, (1, 0)), (0.3, (0.4, 0)), (0.4, (-1, 0)), (0.62, (-0.4, 0)), (0.75, (-0.4, 0)), (0.84, (0.5, 0)), (1, (0, 0))],
    'thinking': [(0, (0, 0)), (0.15, (0.7, 0.8)), (0.8, (0.7, 0.8)), (1, (0, 0))],                # up and away
    'puzzled': [(0, (0, 0)), (0.15, (0.5, 0.3)), (0.42, (0.5, 0.3)), (0.58, (-0.5, 0.3)), (0.84, (-0.5, 0.3)), (1, (0, 0))],
    'worried': [(0, (0, 0)), (0.16, (0, -0.3)), (0.3, (1, -0.2)), (0.44, (1, -0.2)), (0.52, (-1, -0.2)), (0.7, (-1, -0.2)), (0.8, (0, -0.3)), (1, (0, 0))],
    'oops': [(0, (0, 0)), (0.5, (0, 0)), (0.66, (-0.7, -0.5)), (0.86, (-0.7, -0.5)), (1, (0, 0))],   # sheepish: down and away
    'sleepy': [(0, (0, 0)), (0.56, (0, -0.6)), (0.72, (0, 0)), (1, (0, 0))],
    'idle': [(0, (0, 0)), (0.6, (0, 0)), (0.63, (0.8, 0.1)), (0.72, (0.8, 0.1)), (0.75, (0, 0)), (1, (0, 0))],   # a glance now and then
    'point': [(0, (0, 0)), (0.07, (1, 0.1)), (0.24, (0.5, 0.1)), (0.46, (0.5, 0.1)), (0.54, (0, 0)), (1, (0, 0))],   # his eyes go to it first, his head follows, then back to you
    'pose-think': [(0, (0.7, 0.8))],
    'pose-point': [(0, (0.6, 0.1))],
    'point-left': [(0, (0, 0)), (0.07, (-1, 0.1)), (0.24, (-0.5, 0.1)), (0.46, (-0.5, 0.1)), (0.54, (0, 0)), (1, (0, 0))],
    'pose-point-left': [(0, (-0.6, 0.1))],
    'pose-sunk': [(0, (0, 0.7))],
    'pose-hover': [(0, (0, 0.4))],
    'slow-blink': [(0, (0, 0)), (1, (0, 0))],   # straight at you
    'groom': [(0, (0, 0)), (0.14, (0.5, -0.8)), (0.5, (0.5, -0.8)), (0.58, (0, 0)), (1, (0, 0))],   # down at his paw for the licks
}
# Moves the flat rig does not have, in its words, made for the Wall's idles (JM-511 to JM-517; Ant, 25 Nov 2026).
# flat_moves() writes them after the flat rig's own; a name the flat rig gains later is left to the flat rig's.
MORE = {
    # slow blink (JM-512): a cat's 'I like you'. Lids three-quarters down, slowly; held; open. Head dips a touch.
    # Ant, 25 Nov 12:05: "its good".
    'slow-blink': {'ms': 2800, 'tracks': {
        'lid-left': [[0, {'dyh': 0}], [0.12, {'dyh': 0}], [0.4, {'dyh': 0.82}], [0.62, {'dyh': 0.82}], [0.9, {'dyh': 0.06}], [1, {'dyh': 0}]],
        'lid-right': [[0, {'dyh': 0}], [0.12, {'dyh': 0}], [0.4, {'dyh': 0.82}], [0.62, {'dyh': 0.82}], [0.9, {'dyh': 0.06}], [1, {'dyh': 0}]],
        'head': [[0, {}], [0.38, {'dy': 1.1, 'r': 2.5}], [0.64, {'dy': 1.1, 'r': 2.5}], [0.94, {}], [1, {}]],
        'tail-tip': [[0, {}], [0.42, {'r': 9}], [0.8, {'r': 9}], [1, {}]]}},
    # yawn and stretch (JM-514): eyes squeezed shut, mouth wide, head back, a stretch up through his chest, then a
    # little shake of his head and he settles. Ears go back with it.
    'yawn-stretch': {'ms': 3400, 'tracks': {
        'lid-left': [[0, {'dyh': 0}], [0.14, {'dyh': 0.3}], [0.3, {'dyh': 1.05}], [0.7, {'dyh': 1.05}], [0.84, {'dyh': 0.2}], [1, {'dyh': 0}]],
        'lid-right': [[0, {'dyh': 0}], [0.14, {'dyh': 0.3}], [0.3, {'dyh': 1.05}], [0.7, {'dyh': 1.05}], [0.84, {'dyh': 0.2}], [1, {'dyh': 0}]],
        'mouth-talk': [[0, {'o': 0}], [0.16, {'o': 0.3}], [0.4, {'o': 1.55}], [0.62, {'o': 1.55}], [0.78, {'o': 0}], [1, {'o': 0}]],   # past talking: a yawn opens his jaw almost all the way (1.67 is wide open)
        'head': [[0, {}], [0.38, {'dy': -3.2}], [0.62, {'dy': -3.2}], [0.76, {'dy': 0.3}], [0.82, {'r': 4}], [0.88, {'r': -4}], [0.94, {'r': 2}], [1, {}]],
        'body': [[0, {}], [0.38, {'sy': 0.03, 'p': -4}], [0.62, {'sy': 0.03, 'p': -4}], [0.8, {}], [1, {}]],
        'tail-mid': [[0, {}], [0.4, {'r': 6}], [0.7, {'r': 6}], [1, {}]],
        'tail-tip': [[0, {}], [0.4, {'r': 14}], [0.7, {'r': 14}], [1, {}]]}},
    # grooming (JM-513): left paw up in front of his mouth (LICK, by way of the tuck); his head comes down and round to
    # meet it for three licks, eyes on it; paw on up beside his head for two wipes, eyes shut, head leaning in; down.
    # Ant, 25 Nov 12:40: "the paw needs to go in front of his mouth to lick it, Jimmy may need to move his head to meet it".
    'groom': {'ms': 4600, 'tracks': {
        'paw': [[0, {'up': 0, 'r': 0, 'lick': 1}], [0.14, {'up': 1, 'r': 0, 'lick': 1}], [0.5, {'up': 1, 'r': 0, 'lick': 1}], [0.6, {'up': 1, 'r': 4, 'lick': 0}],
                [0.66, {'up': 1, 'r': -6, 'lick': 0}], [0.72, {'up': 1, 'r': 6, 'lick': 0}], [0.78, {'up': 1, 'r': -6, 'lick': 0}], [0.84, {'up': 1, 'r': 2, 'lick': 0}],
                [1, {'up': 0, 'r': 0, 'lick': 0}]],
        'head': [[0, {}], [0.16, {'dy': 3.0, 'dx': 1.7, 'r': 6}], [0.22, {'dy': 4.4, 'dx': 1.7, 'r': 6}], [0.28, {'dy': 3.0, 'dx': 1.7, 'r': 6}],
                 [0.34, {'dy': 4.4, 'dx': 1.7, 'r': 6}], [0.4, {'dy': 3.0, 'dx': 1.7, 'r': 6}], [0.46, {'dy': 4.4, 'dx': 1.7, 'r': 6}],
                 [0.56, {'dy': 2.0, 'r': 18}], [0.84, {'dy': 2.4, 'r': 22}], [1, {}]],
        'mouth-talk': [[0, {'o': 0}], [0.18, {'o': 0}], [0.21, {'o': 0.35}], [0.25, {'o': 0}], [0.33, {'o': 0.35}], [0.37, {'o': 0}],
                       [0.45, {'o': 0.35}], [0.49, {'o': 0}], [1, {'o': 0}]],
        'lid-left': [[0, {'dyh': 0}], [0.16, {'dyh': 0.55}], [0.5, {'dyh': 0.55}], [0.6, {'dyh': 1.05}], [0.86, {'dyh': 1.05}], [0.96, {'dyh': 0}], [1, {'dyh': 0}]],
        'lid-right': [[0, {'dyh': 0}], [0.16, {'dyh': 0.55}], [0.5, {'dyh': 0.55}], [0.6, {'dyh': 1.05}], [0.86, {'dyh': 1.05}], [0.96, {'dyh': 0}], [1, {'dyh': 0}]],
        'ear-right': [[0, {}], [0.58, {}], [0.66, {'r': 16}], [0.72, {'r': 6}], [0.78, {'r': 16}], [0.86, {}], [1, {}]],
        'tail-tip': [[0, {}], [0.3, {'r': 8}], [0.8, {'r': -6}], [1, {}]]}},
}
FLOP = 0.8                               # how far his wrist and paw swing in a wave, after his forearm: a loose paw, not a stiff one
HIS_OWN = ('annoyed',)                   # moves and poses written for him in moves() that the flat rig may come to have too: his own are kept (flat_moves())
FREE = ('pose-held',)                    # poses in which his front paws are off the floor, so are not kept planted (alive())
ABOARD = 'saucer'                        # the number on his bones that shows his flying saucer (0 none, 1 there); the 'pose-saucer' poses set it
RIDING = [False]                         # whether other motions are played with him in it (aboard())
HAT = 'hat'                              # the number on his bones that shows his party hat (0 none, 1 on)
BROW_FLIP = ('worried',)                 # the flat rig tilts these brows inner end down; on a face with depth that reads stern, so here the inner ends go up
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
    if ALIVE and not JAW:
        mend()                                             # the tears in his whisker pads, before anything is added to his skin
        unwhisker()
    GROUND.pop(NAME, None)                                 # his feet are set on the floor afresh, and before his legs are cut free: the cut is fitted to how he sits
    specks()
    if TAPER:
        taper()
    if FLOOF:
        floof()
    if SHOULDER:
        shoulders()
    if SLEEVE:
        free_legs()
        bridges()
    tag()                                                  # after the shoulders, whose smoothing would spread a leg's pull on to it
    if EARS:
        ears()
    if CHEEKS:
        head_fur()
    if BEANS:
        beans()
    if JAW:
        seal()
        for name in MOTIONS:
            shut_in(bpy.data.actions[motion(name)])
    else:
        mouth()
        eyes()
        if GRUMP:
            grump()
        if BROW:
            browed()
        if BULGE:
            eyeballs()
        if LID:
            lids()
        if ALIVE:
            breathing()
            gaze()
        wake('face')
    if ALIVE:
        party_hat()
        wake('hat')
        if not JAW:
            rest()
            whiskers()
            wake('whiskers')
    if TAIL:
        LAID.pop(NAME, None)
        # the last is 'wag tail'. Meshy's is a dog's quick wag; a quiver at the tip alone (tried) read as a twitch, not a
        # wag (Ant, 24 Nov). So: held up, the whole tail swinging wide from side to side twice, its tip following
        for name, sway, beats, quiver in zip(MOTIONS, (2.0, 3.0, 3.0, 2.0, 7.0 if ALIVE else 6.0), (2, 3, 1, 2, 2 if ALIVE else 6),
                                             (None, None, None, None, None)):
            tail_up(bpy.data.actions[motion(name)], sway, beats, quiver)
        wake('tails')
    if ALIVE and not JAW:                                  # Meshy's long motions stare: three blinks in each
        for name in MOTIONS:
            act = bpy.data.actions[motion(name)]
            length = int(act.frame_range[1] - act.frame_range[0])
            if length >= 100:
                blink_in(act, [int(act.frame_range[0] + length * k) for k in (0.2, 0.5, 0.82)])
    wake('build')
    if ALIVE and TAIL and not JAW:
        cat_ways()
        wake('cat_ways')
    if SAUCER:
        saucer()
        wake('saucer')
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
        brows()
    if TALK in a:
        a[TALK] = 0.0
    if BREATH in a:
        a[BREATH] = 0.0
    if CHEEK in a:
        a[CHEEK] = CHEEK_AS
    if HAT in a:
        a[HAT] = 0.0
    if ABOARD in a:
        a[ABOARD] = 1.0 if RIDING[0] else 0.0             # the saucer poses key it themselves
    for prop in MOUTHS + GAZE:
        if prop in a:
            a[prop] = 0.0
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
        brows()
    if TALK in a:
        a[TALK] = 0.0
    if BREATH in a:
        a[BREATH] = 0.0
    if CHEEK in a:
        a[CHEEK] = CHEEK_AS
    if HAT in a:
        a[HAT] = 0.0
    if ABOARD in a:
        a[ABOARD] = 1.0 if RIDING[0] else 0.0             # the saucer poses key it themselves
    for prop in MOUTHS + GAZE:
        if prop in a:
            a[prop] = 0.0
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
    if FEET:
        ground()
    if TAIL:
        lay_tail()
    elif WRAP:                                             # this rig's sitting motion leaves his tail hanging through the floor
        bones = tail_bones()
        for i, (bone, child) in enumerate(zip(bones, bones[1:])):
            angle = math.radians(WRAP[0] + (WRAP[1] - WRAP[0]) * i / (len(bones) - 2))
            aim(bone, child, (math.sin(angle), math.cos(angle), WRAP[2] if i == 0 else 0.0))


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


def twist(bone, degrees):
    """Turn a bone about its own length."""
    bpy.context.view_layer.update()
    pb = arm().pose.bones[bone]
    pb.matrix = pb.matrix @ Matrix.Rotation(math.radians(degrees), 4, 'Y')
    bpy.context.view_layer.update()


def lift(z):
    """Move all of him up or down from where he is."""
    a = arm()
    pb = a.pose.bones['root']
    pb.location = pb.location + a.data.bones['root'].matrix_local.to_3x3().inverted() @ Vector((0.0, 0.0, z))
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
            lids = blinks[i] if isinstance(blinks[i], (tuple, list)) else (blinks[i], 0.0, 0.0)
            for prop, value in zip(LIDS + (TALK, JOY, FROWN) + BROWS + (BREATH,) + MOUTHS + GAZE, lids):
                if prop in a:
                    a[prop] = float(value)
                    a.keyframe_insert('["%s"]' % prop, frame=i)
    if CHEEK in a:                                         # how much his cheeks lift when his eyes shut in this move
        a[CHEEK] = CHEEK_MOVES.get(name[len(OWN):] if name.startswith(OWN) else name, CHEEK_AS)
        a.keyframe_insert('["%s"]' % CHEEK, frame=0)
        a[CHEEK] = CHEEK_AS
    if HAT in a:                                           # his party hat is on in the party pose only
        a[HAT] = 1.0 if name == OWN + 'pose-party' else 0.0
        a.keyframe_insert('["%s"]' % HAT, frame=0)
        a[HAT] = 0.0
    if ABOARD in a and name.startswith(OWN + 'pose-saucer'):   # in his saucer. Other motions do not say, so any of them can be played in it (aboard())
        a[ABOARD] = 1.0
        a.keyframe_insert('["%s"]' % ABOARD, frame=0)
        a[ABOARD] = 0.0
    a.animation_data.action = None
    if blinks:
        blink(0.0)
        say(0.0)
    return act


def chain(first):
    """A limb's bones in order, from the one named down to its tip. Meshy's rigs differ in how many they give."""
    bone, out = arm().data.bones[first], []
    while bone:
        out.append(bone.name)
        bone = bone.children[0] if bone.children else None
    return out


def neck():
    """The bone his head sits on."""
    return arm().data.bones['head'].parent.name


def tail_up(action, sway=3.0, beats=1, quiver=None):
    """Carry his tail up, as modelled, all through one of Meshy's motions, with a slow sway down its length.

    Meshy's dog rig puts the first tail bone deep in his rump and its motions hang the whole tail from it, as a dog
    carries one. On him that swings the tail down from a joint between his back legs. So the first bone is held as
    modelled (it is rump, and the tail then leaves from the top of his rump), and the rest get a sway of `sway`
    degrees a bone, `beats` times through the motion, each bone a little behind the one before. `quiver`, if given, is
    (degrees, times through the motion) for a fast shiver that grows toward the tip: a happy cat's tail, held up."""
    a, bones = arm(), tail_bones()
    f0, f1 = action.frame_range
    span = max(f1 - f0, 1.0)
    for k, bone in enumerate(bones):
        R = a.data.bones[bone].matrix_local.to_3x3()
        for i, fc in curves(action, bone, 'rotation_quaternion').items():
            for key in fc.keyframe_points:
                t = (key.co.x - f0) / span
                side = sway * (0.4 + 0.25 * k) * math.sin(2 * math.pi * beats * t - 0.55 * k) if k else 0.0
                if quiver and k:
                    side += quiver[0] * (k / (len(bones) - 1.0)) ** 2 * math.sin(2 * math.pi * quiver[1] * t - 0.9 * k)
                bob = sway * 0.25 * math.sin(4 * math.pi * beats * t - 0.5 * k) if k else 0.0
                q = (R.inverted() @ Matrix.Rotation(math.radians(side), 3, 'Y') @ Matrix.Rotation(math.radians(bob), 3, 'X') @ R).to_quaternion()
                key.co.y = key.handle_left.y = key.handle_right.y = q[i]
            fc.update()


LAID = {}                              # each model's laid tail, kept: working it out means measuring all of him
GROUND = {}                            # and each model's feet as set on the floor, and how high that floor is


def ground():
    """Set his paws flat on one level floor as he sits.

    Meshy's sitting pose leaves each paw tipped toes-up by a different amount (up to 39 degrees), so his toe beans
    show from the front, and each foot at its own height. Each paw is turned about its own joint until its sole is
    level. Then each foot (the paw and the bone above it) is moved up or down, its leg bending to let it, so that
    all four rest at one height: the floor. His tail is laid on that same floor (lay_tail())."""
    a, ob, me = arm(), skin(), skin().data
    if NAME in GROUND:
        for bone, basis in GROUND[NAME]['pose'].items():
            a.pose.bones[bone].matrix_basis = basis.copy()
        bpy.context.view_layer.update()
        return
    limbs = [chain(first + side) for first in ('clavicle.', 'rearUpr.') for side in ('L', 'R')]
    n = len(me.vertices)
    sets = {}                                              # for each group, the feet and paws it is part of
    for k, limb in enumerate(limbs):
        for bone in limb[-4:]:
            if bone in ob.vertex_groups:
                sets.setdefault(ob.vertex_groups[bone].index, []).append((k, bone in limb[-3:]))
    foot_w, paw_w = np.zeros((len(limbs), n)), np.zeros((len(limbs), n))
    for v in me.vertices:
        for g in v.groups:
            for k, in_paw in sets.get(g.group, ()):
                foot_w[k, v.index] += g.weight
                if in_paw:
                    paw_w[k, v.index] += g.weight
    rest_co = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('co', rest_co)
    rest_co = rest_co.reshape(-1, 3)
    rest_no = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('normal', rest_no)
    rest_no = rest_no.reshape(-1, 3)
    soles = []                                             # the underside of each paw, found as he stands
    for k in range(len(limbs)):
        paw = paw_w[k] > 0.5
        soles.append(paw & (rest_no[:, 2] < -0.5) & (rest_co[:, 2] < rest_co[paw, 2].min() + 0.06))

    def shape():
        bpy.context.view_layer.update()
        made = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
        co = np.zeros(n * 3, 'f4')
        made.vertices.foreach_get('co', co)
        W = np.array(ob.matrix_world)
        return co.reshape(-1, 3).astype('f8') @ W[:3, :3].T + W[:3, 3]

    to_arm = a.matrix_world.to_3x3().inverted()
    turned_by = []
    for limb, sole in zip(limbs, soles):
        total = 0.0
        for _ in range(3):
            pts = shape()[sole]
            cx, cy, _c = np.linalg.lstsq(np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))], pts[:, 2], rcond=None)[0]
            q = (to_arm @ Vector((cx, cy, -1.0)).normalized()).rotation_difference(to_arm @ Vector((0.0, 0.0, -1.0)))
            if math.degrees(q.angle) < 0.3:
                break
            if total + math.degrees(q.angle) > FEET:
                break
            total += math.degrees(q.angle)
            pb = a.pose.bones[limb[-3]]
            M = pb.matrix.copy()
            here = Matrix.Translation(M.to_translation())
            pb.matrix = here @ q.to_matrix().to_4x4() @ here.inverted() @ M
        turned_by.append(round(total, 1))
    co = shape()
    lows = [float(np.percentile(co[foot_w[k] > 0.5, 2], 2)) for k in range(len(limbs))]
    floor = float(np.mean(lows))
    pose, bones = snap(), skeleton()
    for limb, low in zip(limbs, lows):
        M = solve(pose, bones)
        shift = Matrix.Translation(to_arm @ Vector((0.0, 0.0, floor - low)))
        want = {bone: shift @ M[bone] for bone in limb[-4:]}
        reach(pose, bones, limb[-6], limb[-5], limb[-4], want[limb[-4]].to_translation())
        now = solve(pose, bones)
        for bone in limb[-4:]:
            held(pose, now, bones, bone, want[bone])
            now[bone] = want[bone]
    put(pose)
    co = shape()
    GROUND[NAME] = {'pose': {bone: a.pose.bones[bone].matrix_basis.copy() for limb in limbs for bone in limb}, 'floor': floor,
                    'turned': turned_by, 'was': [round(v, 3) for v in lows],
                    'now': [round(float(np.percentile(co[foot_w[k] > 0.5, 2], 2)), 3) for k in range(len(limbs))]}


def plant(was):
    """After all of him has been lowered a little: put his feet, and his tail where it lies on the floor, back where
    they were in the pose `was`. His legs bend to let his body down, as a sitting cat's do."""
    bones = skeleton()
    pose, there = snap(), solve(was, bones)
    for limb in [chain(first + side) for first in ('clavicle.', 'rearUpr.') for side in ('L', 'R')]:
        reach(pose, bones, limb[-6], limb[-5], limb[-4], there[limb[-4]].to_translation())
        now = solve(pose, bones)
        for bone in limb[-4:]:
            held(pose, now, bones, bone, there[bone])
            now[bone] = there[bone]
    now = solve(pose, bones)
    for bone in tail_bones()[2:]:                          # his rump comes down; the tail on the floor stays on it
        held(pose, now, bones, bone, there[bone])
        now[bone] = there[bone]
    put(pose)
    bpy.context.view_layer.update()


def bridges():
    """Take out any scrap of skin still joining a front leg to his body after the cut (free_legs()).

    A face with one corner wholly leg and another wholly body is such a scrap: where leg really meets body, at
    the shoulder, the pull is shared and no corner is wholly either. The cut left two on his right side, each a
    tiny triangle, and with that paw held out they drew out into a dark thread from his arm to his chest.
    Returns how many were taken out, each side."""
    import bmesh
    ob = skin()
    me = ob.data
    found = {}
    for side in ('L', 'R'):
        groups = {ob.vertex_groups[b].index for b in chain('clavicle.' + side) if b in ob.vertex_groups}
        pull = np.zeros(len(me.vertices))
        for v in me.vertices:
            for g in v.groups:
                if g.group in groups:
                    pull[v.index] += g.weight
        found[side] = [p.index for p in me.polygons if pull[list(p.vertices)].max() >= 0.99 and pull[list(p.vertices)].min() <= 0.01]
    gone = sorted(set(found['L']) | set(found['R']))
    if gone:
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.faces[i] for i in gone], context='FACES_ONLY')
        bm.to_mesh(me)
        bm.free()
        me.update()
    return {side: len(faces) for side, faces in found.items()}


def taper():
    """Slim his tail and taper it toward its tip. Each point of it is drawn in toward the line down its middle:
    TAPER[0] of its thickness is kept where it leaves his rump, TAPER[1] at its tip, evenly between.

    Done to the model as Meshy made it, before anything else is built on it, so every motion gets the same tail.
    The line is the line of his tail bones; Meshy folds the last two back inside the tip, so it stops where they
    turn back. Points past the end are drawn in toward the end, which rounds the tip off smaller as well."""
    a, ob, bones = arm(), skin(), tail_bones()
    me = ob.data
    n = len(me.vertices)
    W, A = np.array(ob.matrix_world), np.array(a.matrix_world)
    co = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3).astype('f8') @ W[:3, :3].T + W[:3, 3]
    ends = [np.array(a.data.bones[b].head_local) for b in bones[1:]] + [np.array(a.data.bones[bones[-1]].tail_local)]
    ends = [p @ A[:3, :3].T + A[:3, 3] for p in ends]
    line = ends[:2]
    for p in ends[2:]:
        if (p - line[-1]) @ (line[-1] - line[-2]) <= 0:
            break
        line.append(p)
    groups = {ob.vertex_groups[b].index for b in bones[1:] if b in ob.vertex_groups}
    pull = np.array([sum(g.weight for g in v.groups if g.group in groups) for v in me.vertices])
    sel = np.where(pull > 0.0)[0]
    P = co[sel]
    seg = [float(np.linalg.norm(q - p)) for p, q in zip(line, line[1:])]
    along = np.concatenate([[0.0], np.cumsum(seg)])
    near, where, best = np.zeros_like(P), np.zeros(len(P)), np.full(len(P), 1e9)
    for k, (p, q) in enumerate(zip(line, line[1:])):
        d = q - p
        t = np.clip(((P - p) @ d) / (d @ d), 0.0, 1.0)
        on = p + t[:, None] * d
        r = np.linalg.norm(P - on, axis=1)
        m = r < best
        best[m], near[m], where[m] = r[m], on[m], (along[k] + t[m] * seg[k]) / along[-1]
    keep = TAPER[0] + (TAPER[1] - TAPER[0]) * where
    keep = 1.0 - np.clip(pull[sel], 0.0, 1.0) * (1.0 - keep)     # by how much it is tail: nothing changes where it joins his rump
    co[sel] = near + (P - near) * keep[:, None]
    back = (co - W[:3, 3]) @ np.linalg.inv(W[:3, :3]).T
    me.vertices.foreach_set('co', back.astype('f4').ravel())
    me.update()
    return {'points': int(len(sel)), 'widest': float(best.max()), 'now': float((best * keep).max())}


WEB_NAMES = {'cat_idle': 'stand-idle', 'cat_sajiao': 'act-cute', 'cat_walk': 'walk', 'dog_beifumo': 'petted', 'dog_yaoweiba': 'wag'}   # what Meshy's motions are called in the web copy


WEB_DIALS = ('blink', 'blink_l', 'blink_r', 'joy', 'frown', 'talk', 'oh', 'look_x', 'look_y', HAT)   # the numbers his face paint reads, carried to the page
HOLE = 60                                              # a hole on his head with fewer sides than this is closed in the web copy
SHADE = 'j5_ao'                                        # how open to the light each point of the web copy is: 1 in the open, less where he is tucked in
WEB_DIAL = 'dial_'                                     # each as a bone of this name that does nothing but rise by its number


def paint_code():
    """His skin's colour, as the page needs it: the material's own sums written out as shader code (GLSL).

    His lids' paint, lash line, the inside of his mouth and his pupils' moving are not in any picture: Blender
    works them out for every dot of him from masks on his points and a few numbers. A page has to do the same
    sums. Rather than write them twice and keep two copies in step, this walks the material back from its colour
    and writes down each step. Returns {'glsl': the lines, 'out': the name holding the colour, 'attributes': the
    masks it reads (as a_<name>), 'numbers': the numbers (as u_<name>), 'pictures': the pictures other than
    Meshy's paint (as png, base64)}. It knows only the kinds of step the material uses, and says so if it meets
    another."""
    import base64
    import tempfile
    nt = skin().data.materials[0].node_tree
    drv = {}
    for d in (nt.animation_data.drivers if nt.animation_data else []):
        drv[d.data_path.split('"')[1]] = d.driver.variables[0].targets[0].data_path.split('"')[1]
    lines, done, attrs, images, numbers = [], {}, [], {}, set()
    ids = {n.name: 'n%d' % i for i, n in enumerate(nt.nodes)}

    def f(v):
        return '%.6f' % v

    def const(sock, kind):
        v = sock.default_value
        if kind == 'float':
            return f(v if isinstance(v, (int, float)) else sum(v[:3]) / 3.0)
        if isinstance(v, (int, float)):
            return 'vec3(%s)' % f(v)
        return 'vec3(%s, %s, %s)' % (f(v[0]), f(v[1]), f(v[2]))

    def get(sock, kind):
        if not sock.links:
            return const(sock, kind)
        fs = sock.links[0].from_socket
        name, k = emit(fs.node, fs)
        if k == kind:
            return name
        return ('vec3(%s)' % name) if kind == 'vec3' else ('dot(%s, vec3(0.333333))' % name)

    def emit(node, sock):
        key = (node.name, sock.identifier)
        if key in done:
            return done[key]
        i, t = ids[node.name], node.bl_idname
        var, k = None, 'float'
        if t == 'ShaderNodeAttribute':
            if node.attribute_name not in attrs:
                attrs.append(node.attribute_name)
            var, k = 'a_' + node.attribute_name, 'vec3'
            if sock.name == 'Fac':
                var, k = var + '.x', 'float'
        elif t == 'ShaderNodeValue':
            if node.name in drv:
                numbers.add(drv[node.name])
                var = 'u_' + drv[node.name]
            else:
                var = f(node.outputs[0].default_value)
        elif t == 'ShaderNodeTexImage':
            img = node.image.name
            base = not node.inputs[0].links                # Meshy's paint, by his own map; the page has it as `map`
            tex = 'map' if base else 't_' + ''.join(c if c.isalnum() else '_' for c in img)
            if (node.name, 'rgba') not in done:
                if base:
                    lines.append('vec4 %s = texture2D(map, vMapUv);' % i)
                else:
                    images[tex] = img
                    uv = get(node.inputs[0], 'vec3')
                    lines.append('vec4 %s = texture2D(%s, vec2((%s).x, 1.0 - (%s).y));' % (i, tex, uv, uv))
                done[(node.name, 'rgba')] = True
            var, k = ('%s.a' % i, 'float') if sock.name == 'Alpha' else ('%s.rgb' % i, 'vec3')
        elif t == 'ShaderNodeMath':
            a, b, c = (get(node.inputs[j], 'float') for j in range(3))
            e = {'ADD': '%s + %s' % (a, b), 'SUBTRACT': '%s - %s' % (a, b), 'MULTIPLY': '%s * %s' % (a, b),
                 'MAXIMUM': 'max(%s, %s)' % (a, b), 'MINIMUM': 'min(%s, %s)' % (a, b), 'MULTIPLY_ADD': '%s * %s + %s' % (a, b, c),
                 'ABSOLUTE': 'abs(%s)' % a, 'GREATER_THAN': '(%s > %s ? 1.0 : 0.0)' % (a, b), 'SQRT': 'sqrt(max(%s, 0.0))' % a}[node.operation]
            if node.use_clamp:
                e = 'clamp(%s, 0.0, 1.0)' % e
            lines.append('float %s = %s;' % (i, e))
            var = i
        elif t == 'ShaderNodeMapRange':
            if node.interpolation_type != 'LINEAR' or node.data_type != 'FLOAT':
                raise ValueError('paint_code: a Map Range of a kind it does not know: ' + node.name)
            x, a0, a1, b0, b1 = (get(node.inputs[j], 'float') for j in range(5))
            e = '(%s - %s) / (%s - %s)' % (x, a0, a1, a0)
            if node.clamp:
                e = 'clamp(%s, 0.0, 1.0)' % e
            lines.append('float %s = %s + (%s) * (%s - %s);' % (i, b0, e, b1, b0))
            var = i
        elif t == 'ShaderNodeMix':
            if node.blend_type != 'MIX':
                raise ValueError('paint_code: a Mix of a kind it does not know: ' + node.name)
            fac = get(node.inputs[0], 'float')
            if node.clamp_factor:
                fac = 'clamp(%s, 0.0, 1.0)' % fac
            if node.data_type == 'FLOAT':
                lines.append('float %s = mix(%s, %s, %s);' % (i, get(node.inputs[2], 'float'), get(node.inputs[3], 'float'), fac))
                var = i
            else:
                j = 4 if node.data_type == 'VECTOR' else 6
                lines.append('vec3 %s = mix(%s, %s, %s);' % (i, get(node.inputs[j], 'vec3'), get(node.inputs[j + 1], 'vec3'), fac))
                var, k = i, 'vec3'
        elif t == 'ShaderNodeCombineXYZ':
            lines.append('vec3 %s = vec3(%s, %s, %s);' % ((i,) + tuple(get(node.inputs[j], 'float') for j in range(3))))
            var, k = i, 'vec3'
        elif t == 'ShaderNodeVectorMath':
            a, b = get(node.inputs[0], 'vec3'), get(node.inputs[1], 'vec3')
            if node.operation == 'DIVIDE':
                lines.append('vec3 %s = vec3((%s).x / (%s).x, (%s).y / (%s).y, 0.0);' % (i, a, b, a, b))
            else:
                lines.append('vec3 %s = %s %s %s;' % (i, a, {'ADD': '+', 'SUBTRACT': '-'}[node.operation], b))
            var, k = i, 'vec3'
        else:
            raise ValueError('paint_code: a kind of step it does not know: %s (%s)' % (t, node.name))
        done[key] = (var, k)
        return done[key]

    bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    final = get(bsdf.inputs['Base Color'], 'vec3')
    pictures = {}
    for tex, name in images.items():
        img = bpy.data.images[name]
        p = os.path.join(tempfile.gettempdir(), tex + '.png')
        img.save(filepath=p)
        pictures[tex] = {'size': list(img.size), 'png': base64.b64encode(open(p, 'rb').read()).decode()}
    return {'glsl': '\n'.join(lines), 'out': final, 'attributes': attrs, 'numbers': sorted(numbers), 'pictures': pictures,
            'dials': list(WEB_DIALS), 'dial': WEB_DIAL, 'shade': SHADE}


def web_copy(path=None, names=None, keep=0.17, picture=2048, step=1, quality=88, face=True):
    """Write a lighter copy of him for a web page: one .glb file with his shape, bones, face shapes and motions.

    It is made from a copy; the model itself is not touched, and the copy is cleared away afterwards.
      - His body's points are thinned to `keep` of them. The points his face shapes move (lids, brows, mouth,
        cheeks) are left as they are: thinned, his lids would not meet.
      - His look is painted into one picture, `picture` across: Meshy's paint with what this script adds by
        Blender's own material tricks (toe beans, mouth line, eyes), as he looks at rest.
      - Every motion (or just `names`) is written as its own animation, bones and face shapes together, a frame
        every `step`. His own are named without their prefix; Meshy's by WEB_NAMES.
    His saucer goes too, as a thing called 'saucer' held by his root bone: a model file cannot say when it shows,
    so the page shows or hides it.

    With `face` (the usual way) his face works on the page as it does here. His lids' paint, lash line, the
    inside of his mouth and his pupils' moving are sums the material does, not a picture, so the page must do
    them too. For that the file carries: Meshy's own paint rather than a painted-in picture; the masks on his
    points (as _j5_...); and each number the sums read (WEB_DIALS) as a bone, dial_<name>, that rises by that
    number in every motion. Beside the file goes <name>-paint.json, from paint_code(): the sums as shader code,
    and his eye pictures. His party hat goes as a thing called 'hat', for the page to show by dial_hat.

    Without `face`, his look is painted into one picture as he looks at rest, and the page needs no code: but his
    eyes do not move, a shut lid has the open eye's picture on it, and there is no hat. (At 1024 across that
    picture left his eyes' rims ragged; 2048 costs a fifth of a megabyte more.) Returns what was written."""
    a, ob = arm(), skin()
    scene, layer = bpy.context.scene, bpy.context.view_layer
    path = path or os.path.normpath(os.path.join(HERE, os.path.dirname(GLB), 'web', NAME.lower() + '-web.glb'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    names = names or sorted(x.name for x in bpy.data.actions if x.name.startswith((OWN, PRE)))
    rest()
    tag_ = NAME + '-web'
    coll = bpy.data.collections.new(tag_)
    scene.collection.children.link(coll)
    a2 = a.copy()
    a2.name = tag_
    a2.animation_data_clear()
    coll.objects.link(a2)
    ob2 = ob.copy()
    ob2.data = ob.data.copy()
    ob2.name = SKIN + '-web'
    coll.objects.link(ob2)
    ob2.parent = a2
    ob2.hide_render = False
    for m in ob2.modifiers:
        if m.type == 'ARMATURE':
            m.object = a2
    craft, craft2 = bpy.data.objects.get(NAME + '-saucer'), None
    if craft:                                              # his saucer goes too, full size, as a thing called 'saucer' for the page to show or hide
        craft2 = craft.copy()
        craft2.name = 'saucer'
        craft2.animation_data_clear()
        craft2.scale = (1.0, 1.0, 1.0)
        craft2.hide_render = False
        coll.objects.link(craft2)
        craft2.parent = a2
    bones2, hat2 = None, None
    if face:
        bones2 = a2.data = a.data.copy()                   # bones of his own, so the dials can be added without touching the model
        for o in layer.objects:
            o.select_set(False)
        a2.select_set(True)
        layer.objects.active = a2
        bpy.ops.object.mode_set(mode='EDIT')
        for k in WEB_DIALS:
            b = bones2.edit_bones.new(WEB_DIAL + k)
            b.head, b.tail = (0.0, 0.0, 0.0), (0.0, 0.0, 0.1)    # pointing up, so its own "along" is up on the page too
            b.use_deform = False
        bpy.ops.object.mode_set(mode='OBJECT')
        for k in WEB_DIALS:
            d = a2.pose.bones[WEB_DIAL + k].driver_add('location', 1).driver
            d.type = 'SUM'
            v = d.variables.new()
            v.targets[0].id = a2
            v.targets[0].data_path = '["%s"]' % k
        hat = bpy.data.objects.get(NAME + '-hat')
        if hat:
            hat2 = hat.copy()
            hat2.name = 'hat'
            hat2.animation_data_clear()
            hat2.scale = (1.0, 1.0, 1.0)
            hat2.hide_render = False
            hat2.hide_viewport = False
            coll.objects.link(hat2)
            hat2.parent = a2
            hat2.parent_type, hat2.parent_bone = 'BONE', hat.parent_bone
            hat2.matrix_parent_inverse = hat.matrix_parent_inverse.copy()
    strands, strands2 = bpy.data.objects.get(NAME + '-whiskers'), None
    if strands:                                            # his whiskers go as they are, a thing called 'whiskers' held by his head bone
        strands2 = strands.copy()
        strands2.name = 'whiskers'
        strands2.hide_render = False
        coll.objects.link(strands2)
        strands2.parent = a2
        strands2.parent_type, strands2.parent_bone = 'BONE', strands.parent_bone
        strands2.matrix_parent_inverse = strands.matrix_parent_inverse.copy()
    me = ob2.data
    if me.shape_keys and me.shape_keys.animation_data:     # his face shapes follow the copy's numbers, not the model's
        for d in me.shape_keys.animation_data.drivers:
            for v in d.driver.variables:
                for t in v.targets:
                    if t.id == a:
                        t.id = a2
    # Meshy's mesh is in patches, one for each island of his picture: along every seam there are two points in the
    # same place, one for each side. Whole, that never shows. Thinned, each patch's edge moves by itself and the
    # seams open into holes. So join each such pair first. Only pairs held by the same bones, though: where his
    # legs were cut free of his body (free_legs) the two sides also share a place, and must stay apart.
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    held = bm.verts.layers.deform.verify()

    def alike(p, q):
        p, q = p[held], q[held]
        return set(p.keys()) == set(q.keys()) and all(abs(p[k] - q[k]) < 1e-4 for k in p.keys())
    pairs = {v: t for v, t in bmesh.ops.find_doubles(bm, verts=bm.verts, dist=1e-5)['targetmap'].items() if alike(v, t)}
    bmesh.ops.weld_verts(bm, targetmap=pairs)
    # Small holes in him, all by his whiskers, are closed where they can be: with nothing drawn behind his skin on
    # a page, whatever is past him shows through. Only on his head: lower down the openings are where his legs
    # were cut free, and a face laid across one would stretch when the leg moved. Blender's own hole filler gives
    # up on most of these, because two holes often meet at one point. So such a point is split in two for a
    # moment, each hole's rim is walked round and given a face, and the point is joined again. A new face takes
    # its place in the picture from the faces beside it. Rims tangled with a whisker strand are left: Meshy's
    # whiskers are scraps and strands of mesh, torn where they meet his skin, in the model itself.
    top = a.data.bones['head']
    heads = {ob2.vertex_groups[b.name].index for b in [top] + list(top.children_recursive) if b.name in ob2.vertex_groups}

    def rims():
        return [e for e in bm.edges if e.is_boundary and all(sum(w for g, w in v[held].items() if g in heads) > 0.5 for v in e.verts)]
    apart = []
    for v in {v for e in rims() for v in e.verts if sum(1 for k in v.link_edges if k.is_boundary) > 2}:
        apart += list(bmesh.utils.vert_separate(v, v.link_edges))
    rim = rims()
    at = {}
    for e in rim:
        for v in e.verts:
            at.setdefault(v, []).append(e)
    old, used, closed = set(bm.faces), set(), []
    for first in rim:
        if first in used:
            continue
        ring, v, shut = [first.verts[0]], first.verts[1], False
        used.add(first)
        while True:
            if v is ring[0]:
                shut = True
                break
            ring.append(v)
            on = [k for k in at.get(v, []) if k not in used]
            if not on:
                break
            used.add(on[0])
            v = on[0].other_vert(v)
        if shut and 3 <= len(ring) <= HOLE:
            try:
                closed.append(bm.faces.new(ring))
            except ValueError:                             # a loose scrap of a face, or a rim that crosses itself
                pass
    uv = bm.loops.layers.uv.active
    for f in closed:
        for l in f.loops:                                  # facing the way its neighbour does: along the edge they share, the two run opposite ways
            beside = [k for k in l.edge.link_loops if k.face in old]
            if beside:
                if beside[0].vert is l.vert:
                    f.normal_flip()
                break
        if uv:
            for l in f.loops:
                near = [k for k in l.vert.link_loops if k.face in old]
                if near:
                    l[uv].uv = near[0][uv].uv
    bmesh.ops.remove_doubles(bm, verts=[v for v in apart if v.is_valid], dist=1e-6)
    closed = len(closed)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    me.update()
    before = len(me.vertices)
    # thin his body, leaving his face
    n = len(me.vertices)
    moved = np.zeros(n, bool)
    if me.shape_keys:
        keys = me.shape_keys.key_blocks
        basis = np.zeros(n * 3, 'f4')
        keys[0].data.foreach_get('co', basis)
        for k in keys[1:]:
            if k.name == BREATH:
                continue
            c = np.zeros(n * 3, 'f4')
            k.data.foreach_get('co', c)
            moved |= np.abs(c - basis).reshape(-1, 3).max(1) > 1e-5
    body = ~moved

    def thin(obj, body):
        m = obj.data
        m.vertices.foreach_set('select', body)
        ev = np.zeros(len(m.edges) * 2, 'i4')
        m.edges.foreach_get('vertices', ev)
        m.edges.foreach_set('select', body[ev.reshape(-1, 2)].all(1))
        first, loops = np.zeros(len(m.polygons), 'i4'), np.zeros(len(m.loops), 'i4')
        m.polygons.foreach_get('loop_start', first)
        m.loops.foreach_get('vertex_index', loops)
        m.polygons.foreach_set('select', np.add.reduceat((~body[loops]).astype('i4'), first) == 0)
        m.update()
        for o in layer.objects:
            o.select_set(False)
        obj.select_set(True)
        layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.mesh.decimate(ratio=keep)
        bpy.ops.object.mode_set(mode='OBJECT')

    # His true surface, kept for working out his shade below.
    from mathutils.bvhtree import BVHTree
    co = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    true_ = BVHTree.FromPolygons([tuple(c) for c in co], [tuple(f.vertices) for f in me.polygons])
    thin(ob2, body)
    if face:
        # A page draws no shadows, so a part tucked under another is lit as if out in the open: the ledge of his
        # neck under the back of his head showed as a pale patch. So how open each point is goes with him: from it,
        # look out along a spread of lines and count those that get clear of him.
        import math
        spread = []
        for i in range(24):
            z = (i + 0.5) / 24
            r, t = math.sqrt(1.0 - z * z), i * 2.399963
            spread.append((r * math.cos(t), r * math.sin(t), z))
        whole = sum(z for _, _, z in spread)
        m = len(me.vertices)
        openness = np.ones(m, 'f4')
        for v in me.vertices:
            up = v.normal
            if up.length < 0.5:
                continue
            side = up.orthogonal().normalized()
            other = up.cross(side)
            start, clear = v.co + up * 0.006, 0.0
            for x, y, z in spread:
                if true_.ray_cast(start, side * x + other * y + up * z, 0.45)[0] is None:
                    clear += z
            openness[v.index] = clear / whole
        ev = np.zeros(len(me.edges) * 2, 'i4')
        me.edges.foreach_get('vertices', ev)
        ev = ev.reshape(-1, 2)
        for _ in range(2):                                 # evened out with its neighbours, twice: point by point it is speckled
            total, count = openness.copy(), np.ones(m, 'f4')
            np.add.at(total, ev[:, 0], openness[ev[:, 1]])
            np.add.at(total, ev[:, 1], openness[ev[:, 0]])
            np.add.at(count, ev[:, 0], 1.0)
            np.add.at(count, ev[:, 1], 1.0)
            openness = total / count
        me.attributes.new(SHADE, 'FLOAT', 'POINT').data.foreach_set('value', openness)
    # his look: Meshy's own paint for the page to work on, or all of it painted into one picture
    if face:
        img = next(k.image for k in ob2.data.materials[0].node_tree.nodes if k.type == 'TEX_IMAGE' and not k.inputs[0].links and k.outputs['Color'].links)
    else:
        img = bpy.data.images.new(tag_, picture * 2, picture * 2, alpha=False)
        nt = ob2.data.materials[0].node_tree
        node = nt.nodes.new('ShaderNodeTexImage')
        node.image = img
        for k in nt.nodes:
            k.select = False
        node.select = True
        nt.nodes.active = node
        engine = scene.render.engine
        try:
            scene.render.engine = 'CYCLES'
            scene.cycles.device = 'CPU'
            scene.cycles.samples = 1
            bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'}, margin=6, use_clear=True)
        finally:
            scene.render.engine = engine
            nt.nodes.remove(node)
        img.scale(picture, picture)
        img.pack()
    mat = bpy.data.materials.new(tag_)
    mat.use_nodes = True
    nt = mat.node_tree
    for k in list(nt.nodes):
        nt.nodes.remove(k)
    out_, bsdf, tex = nt.nodes.new('ShaderNodeOutputMaterial'), nt.nodes.new('ShaderNodeBsdfPrincipled'), nt.nodes.new('ShaderNodeTexImage')
    tex.image = img
    nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    nt.links.new(bsdf.outputs['BSDF'], out_.inputs['Surface'])
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Roughness'].default_value = 0.75
    me.materials.clear()
    me.materials.append(mat)
    masks = paint_code()['attributes'] + [SHADE] if face else []
    for name in [x.name for x in me.attributes if x.name.startswith('j5_') or x.data_type in ('FLOAT_COLOR', 'BYTE_COLOR')]:
        at = me.attributes[name]
        if name not in masks:
            me.attributes.remove(at)                       # masks the page has no use for
        elif at.data_type in ('FLOAT_COLOR', 'BYTE_COLOR'):    # a colour goes as three plain numbers: a file's own colours are for another use
            c = np.zeros(len(me.vertices) * 4, 'f4')
            at.data.foreach_get('color', c)
            me.attributes.remove(at)
            at = me.attributes.new('_' + name, 'FLOAT_VECTOR', 'POINT')
            at.data.foreach_set('vector', c.reshape(-1, 4)[:, :3].ravel())
        else:
            at.name = '_' + name                           # a file keeps only the extra masks whose names start with _
    # each motion on a track of its own
    ad = a2.animation_data_create()
    called = {}
    for name in names:
        act = bpy.data.actions[name]
        short = name[len(OWN):] if name.startswith(OWN) else WEB_NAMES.get(name[len(PRE):], name[len(PRE):])
        track = ad.nla_tracks.new()
        track.name = short
        track.strips.new(short, int(act.frame_range[0]), act)
        called[short] = int(act.frame_range[1] - act.frame_range[0]) + 1
    # write it, with nothing else in the scene to slow the frames down
    tops = layer.layer_collection.children
    was = {c.name: c.exclude for c in tops}
    hidden = {o.name: o.hide_get() for o in layer.objects}
    try:
        for c in tops:
            c.exclude = c.name != tag_
        for o in layer.objects:
            o.select_set(False)
        a2.select_set(True)
        ob2.select_set(True)
        if craft2:
            craft2.select_set(True)
        if hat2:
            hat2.select_set(True)
        if strands2:
            strands2.select_set(True)
        layer.objects.active = a2
        # his scene only: another scene in the file (the Wall's 'Space') can have things selected too, and the
        # exporter took them (25 Nov 2026: planets and satellites came out inside his web copy)
        bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True, use_active_scene=True, export_animation_mode='NLA_TRACKS',
                                  export_frame_step=step, export_force_sampling=True, export_optimize_animation_size=True,
                                  export_optimize_animation_keep_anim_armature=True, export_anim_slide_to_zero=True,   # every bone in every motion: a bone left out goes back to standing, and most of his moves are sitting
                                  export_morph=True, export_morph_normal=False, export_morph_animation=True,
                                  export_image_format='JPEG', export_jpeg_quality=quality, export_tangents=False,
                                  export_extras=False, export_cameras=False, export_lights=False, export_def_bones=False, export_attributes=face)
    finally:
        for c in tops:
            c.exclude = was[c.name]
        for o in layer.objects:
            if o.name in hidden and o.hide_get() != hidden[o.name]:
                o.hide_set(hidden[o.name])
    made = {'path': path, 'bytes': os.path.getsize(path), 'points': (before, len(me.vertices)), 'face': int(moved.sum()), 'joined': len(pairs), 'closed': closed,
            'triangles': sum(len(p.vertices) - 2 for p in me.polygons), 'motions': called}
    bpy.data.objects.remove(ob2, do_unlink=True)
    if craft2:
        bpy.data.objects.remove(craft2, do_unlink=True)
    if hat2:
        bpy.data.objects.remove(hat2, do_unlink=True)
    if strands2:
        bpy.data.objects.remove(strands2, do_unlink=True)
    bpy.data.objects.remove(a2, do_unlink=True)
    if bones2:
        bpy.data.armatures.remove(bones2)
    bpy.data.meshes.remove(me)
    bpy.data.materials.remove(mat)
    if face:
        import json
        doc = paint_code()
        made['paint'] = path[:-4] + '-paint.json'
        json.dump(doc, open(made['paint'], 'w'))
        made['paint_bytes'] = os.path.getsize(made['paint'])
    else:
        bpy.data.images.remove(img)
    bpy.data.collections.remove(coll)
    rest()
    return made


def lay_tail():
    """Lay his tail round him as he sits: out from the top of his rump, down to the floor, round his left side, tip lifting.

    Where the floor is and how far out his haunch and paws reach are measured from him as posed, so the tail lies
    against him without cutting through. The first tail bone is rump and is left as modelled (see tail_up())."""
    a, ob, bones = arm(), skin(), tail_bones()
    if NAME in LAID:
        for bone, basis in LAID[NAME]['pose'].items():
            a.pose.bones[bone].matrix_basis = basis.copy()
        bpy.context.view_layer.update()
        return
    sweep, lift_from, lift, radius, lean = TAIL
    for bone in bones:
        a.pose.bones[bone].matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    made = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    n = len(made.vertices)
    co = np.zeros(n * 3, 'f4')
    made.vertices.foreach_get('co', co)
    W = np.array(ob.matrix_world)
    co = co.reshape(-1, 3).astype('f8') @ W[:3, :3].T + W[:3, 3]
    groups = {ob.vertex_groups[b].index for b in bones[1:] if b in ob.vertex_groups}
    in_tail = np.array([sum(g.weight for g in v.groups if g.group in groups) for v in ob.data.vertices])
    body = co[in_tail < 0.2]
    floor = GROUND[NAME]['floor'] if NAME in GROUND else float(body[:, 2].min())   # the floor his paws are on, if they have been set on one
    low = body[body[:, 2] < floor + 0.25]
    centre = low[:, :2].mean(0)
    turn_ = math.radians(facing())
    back = np.array([math.sin(turn_), math.cos(turn_)])
    left = np.array([back[1], -back[0]])
    rel = low[:, :2] - centre
    ang, far = np.degrees(np.arctan2(rel @ left, rel @ back)), np.linalg.norm(rel, axis=1)
    grid = np.arange(-60.0, 241.0, 5.0)
    reach = np.array([far[np.abs((ang - g + 180) % 360 - 180) < 14].max(initial=0.0) for g in grid])

    start = np.array(a.matrix_world @ a.pose.bones[bones[1]].matrix.to_translation())
    r0 = float(np.linalg.norm(start[:2] - centre))
    a0 = math.degrees(math.atan2((start[:2] - centre) @ left, (start[:2] - centre) @ back))

    reached = [sweep]                  # how far round the tail gets: found by laying it, then laid again to fit
    rise = [0.0]                       # and how much higher than its radius says it must lie to rest on the floor, not in it

    def path(u):
        angle = a0 + u * (reached[0] - a0)
        up = ease((u - lift_from) / (1.0 - lift_from))
        r = r0 + (float(np.interp(angle, grid, reach)) + radius + 0.015 - r0) * ease(u / 0.2) + lean * up
        z = start[2] + (floor + radius + rise[0] - start[2]) * ease(u / 0.25) + lift * up
        d = back * math.cos(math.radians(angle)) + left * math.sin(math.radians(angle))
        return Vector((centre[0] + r * d[0], centre[1] + r * d[1], z))

    to_arm = a.matrix_world.to_3x3().inverted()
    for level in range(4):
        for fit in range(5):
            for bone in bones[1:]:
                a.pose.bones[bone].matrix_basis = Matrix.Identity(4)
            u = 0.0
            for bone, child in zip(bones[1:], bones[2:]):
                bpy.context.view_layer.update()
                here = a.matrix_world @ a.pose.bones[bone].matrix.to_translation()
                length = (a.data.bones[child].head_local - a.data.bones[bone].head_local).length
                lo = u
                hi = u
                while hi < 3.0 and (path(hi) - here).length < length:   # past 1 the path just carries on round
                    hi += 0.01
                for _ in range(30):
                    mid = (lo + hi) / 2
                    if (path(mid) - here).length < length:
                        lo = mid
                    else:
                        hi = mid
                u = hi
                aim(bone, child, to_arm @ (path(u) - here))
            if abs(u - 1.0) < 0.03:
                break
            reached[0] = a0 + (reached[0] - a0) * u        # ended short or long: fit the path to the tail and lay it again
        # his tail is thicker than its radius says, and thicker in some places than others: measure where its
        # underside has come to, and lay it again that much higher (or lower) until it rests on the floor
        bpy.context.view_layer.update()
        made = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
        now = np.zeros(n * 3, 'f4')
        made.vertices.foreach_get('co', now)
        under = float((now.reshape(-1, 3).astype('f8') @ W[:3, :3].T + W[:3, 3])[in_tail > 0.5, 2].min())
        if abs(under - floor) < 0.004:
            break
        rise[0] += floor - under
    bpy.context.view_layer.update()
    LAID[NAME] = {'pose': {bone: a.pose.bones[bone].matrix_basis.copy() for bone in bones},
                  'floor': floor + radius * (1.0 if NAME in GROUND else 0.8) + rise[0], 'centre': Vector((centre[0], centre[1], 0.0)), 'rise': rise[0], 'under': under}


def curl_tail(degrees):
    """Curl his laid tail, a bone at a time from the one after his rump: up off the floor (and a little out, away
    from him) for a positive number, down for a negative one. Nothing goes below the floor, so a tail that is
    already lying flat stays flat. Lifting and lowering its tip is what a sitting cat's tail does; swinging it
    sideways along the floor would run it into him."""
    a, bones = arm(), tail_bones()
    if NAME not in LAID:
        was = snap()
        lay_tail()
        put(was)
    floor, centre = LAID[NAME]['floor'], LAID[NAME]['centre']
    up = Vector((0.0, 0.0, 1.0))
    for bone, child, amount in zip(bones[1:], bones[2:], degrees):
        bpy.context.view_layer.update()
        pb = a.pose.bones[bone]
        M = pb.matrix.copy()
        here = M.to_translation()
        d = a.pose.bones[child].matrix.to_translation() - here
        length = d.length
        flat = Vector((d.x, d.y, 0.0)).normalized()
        if amount:
            out = 1.0 if up.cross(flat).dot(here - centre) > 0 else -1.0
            turn_ = Matrix.Rotation(math.radians(amount), 4, flat.cross(up)) @ Matrix.Rotation(math.radians(out * 0.4 * abs(amount)), 4, 'Z')
            pb.matrix = Matrix.Translation(here) @ turn_ @ Matrix.Translation(-here) @ M
            bpy.context.view_layer.update()
            d = a.pose.bones[child].matrix.to_translation() - here
        if here.z + d.z < floor and here.z >= floor - 0.02:          # keep the next joint off the floor
            flat = Vector((d.x, d.y, 0.0)).normalized()
            drop = max(-1.0, min(1.0, (floor - here.z) / length))
            aim(bone, child, flat * math.sqrt(1.0 - drop * drop) + up * drop)
    bpy.context.view_layer.update()


def swing_tail(root, mid, tip, face=0.0):
    """Swing the three parts of his tail by degrees, from where they are. With TAIL it is laid round him, and the
    swing lifts or lowers it (curl_tail())."""
    if TAIL:
        bones = tail_bones()[1:-1]                         # the first bone is rump and stays put
        third = max(1, len(bones) // 3)
        sizes = (third, third, len(bones) - 2 * third)
        each = []
        for size, degrees, share in zip(sizes, (root, mid, tip), (0.3, 0.7, 1.0)):
            each += [degrees * share * 2.0 / size] * size
        if any(each):
            curl_tail(each)
        return
    bones = tail_bones()
    third = max(1, len(bones) // 3)
    for k, degrees in enumerate((root, mid, tip)):
        if degrees:
            for bone in bones[third * k:third * (k + 1)]:
                nudge(bone, z=degrees * 2.0 / third, face=face)


def brow_at(name, t):
    """His brows' added move (BROW_MOVES) for the move called `name`, at t (0 to 1), eased between its entries."""
    track = BROW_MOVES.get(name) if BROW else None
    if not track:
        return (0.0, 0.0, 0.0, 0.0)
    if t <= track[0][0]:
        return tuple(track[0][1])
    for (t0, v0), (t1, v1) in zip(track, track[1:]):
        if t <= t1:
            k = ease((t - t0) / max(t1 - t0, 1e-6))
            return tuple(x + (y - x) * k for x, y in zip(v0, v1))
    return tuple(track[-1][1])


def with_brows(name, looks, frames):
    """A move's face numbers, frame by frame, with its added brow move on top. `looks` may be None."""
    if not BROW or (looks is None and name not in BROW_MOVES):
        return looks
    out = []
    for i in range(frames):
        look = looks[i] if looks is not None else 0.0
        look = list(look) if isinstance(look, (tuple, list)) else [look]
        look += [0.0] * (6 + len(BROWS) - len(look))
        add = brow_at(name, i / max(frames - 1, 1))
        out.append(tuple(look[:6] + [x + y for x, y in zip(look[6:], add)]))
    return out


def skeleton():
    """His bones for working poses out by sums, without asking Blender: their names parents first, each one's
    parent, and where each sits in its parent at rest."""
    order, parent = [], {}

    def walk(bone):
        order.append(bone.name)
        parent[bone.name] = bone.parent.name if bone.parent else None
        for child in bone.children:
            walk(child)
    for bone in arm().data.bones:
        if bone.parent is None:
            walk(bone)
    at_rest = {bone.name: bone.matrix_local.copy() for bone in arm().data.bones}
    local = {name: (at_rest[parent[name]].inverted() @ at_rest[name]) if parent[name] else at_rest[name] for name in order}
    return order, parent, local


def solve(pose, bones):
    """Where every bone is, in his own space, for a pose (as snap() gives one)."""
    order, parent, local = bones
    M = {}
    for name in order:
        M[name] = (M[parent[name]] @ local[name] if parent[name] else local[name]) @ pose[name]
    return M


def held(pose, M, bones, name, target):
    """Change a pose so that one bone sits at `target`, given where its parent is in M."""
    up = bones[1][name]
    pose[name] = ((M[up] @ bones[2][name]) if up else bones[2][name]).inverted() @ target


def spun(pose, bones, name, turn_):
    """Change a pose by turning one bone about its own joint (a 3 by 3 turn, in his own space)."""
    M = solve(pose, bones)
    here = Matrix.Translation(M[name].to_translation())
    held(pose, M, bones, name, here @ turn_.to_4x4() @ here.inverted() @ M[name])


def turned(x=0.0, y=0.0, z=0.0, face=0.0):
    """A turn by degrees, as nudge() makes one: x nods, y rolls, z turns, about axes lined up with him."""
    F = Matrix.Rotation(math.radians(-face), 3, 'Z')
    D = Matrix.Rotation(math.radians(z), 3, 'Z') @ Matrix.Rotation(math.radians(x), 3, 'X') @ Matrix.Rotation(math.radians(y), 3, 'Y')
    return F @ D @ F.inverted()


def keyed(keys, t):
    """A number at t (0 to 1), eased between (t, number) pairs."""
    if t <= keys[0][0]:
        return float(keys[0][1])
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * ease((t - t0) / max(t1 - t0, 1e-6))
    return float(keys[-1][1])


def reach(pose, bones, upper, lower, wrist, target):
    """Bend a leg at its two joints so that its wrist comes to `target`, keeping its elbow on the side it is on."""
    M = solve(pose, bones)
    S, E, W = (M[name].to_translation() for name in (upper, lower, wrist))
    first, second = (E - S).length, (W - E).length
    way = target - S
    far = max(abs(first - second) + 1e-4, min(first + second - 1e-4, way.length))
    way.normalize()
    along = (first * first - second * second + far * far) / (2 * far)
    out = math.sqrt(max(first * first - along * along, 0.0))
    pole = (E - S) - way * (E - S).dot(way)
    if pole.length < 1e-6:
        pole = Vector((0.0, 1.0, 0.0)) - way * way.y
    pole.normalize()
    bend = S + way * along + pole * out
    here = Matrix.Translation(S)
    held(pose, M, bones, upper, here @ (E - S).rotation_difference(bend - S).to_matrix().to_4x4() @ here.inverted() @ M[upper])
    M = solve(pose, bones)
    E, W = M[lower].to_translation(), M[wrist].to_translation()
    here = Matrix.Translation(E)
    held(pose, M, bones, lower, here @ (W - E).rotation_difference(S + way * far - E).to_matrix().to_4x4() @ here.inverted() @ M[lower])


def alive(name, poses, looks, base, face, loop=False):
    """Bring a move to life. A move as first written is a head, ears and tail moving on a body that never stirs.

    Given its poses, frame by frame, and its face numbers (`looks`, which may be None), this adds, by rule:
      - his ears, for the moves in EAR_MOVES;
      - wind-up, overshoot and settle: each bone's turning is sharpened, so it dips a little the other way before
        a move, goes a little past where it is heading, and eases back;
      - a slow drift of his head while a pose is held, so that a hold is never frozen;
      - his body: his spine takes up part of each turn, tilt and nod of his head, so his shoulders follow. His
        front paws stay planted where they stand (unless a paw is in the air): his legs bend to keep them there;
      - a blink on each big turn of his head, and one or two in any long move that has none;
      - his breathing, in moves long enough for a breath.
    `loop` is for a move that goes round and round. Returns the new poses and face numbers. All of it is sums on
    the poses; Blender is not asked to pose him."""
    if not ALIVE:
        return poses, looks
    bones = skeleton()
    Mb = solve(base, bones)
    frames = len(poses)
    poses = [dict(pose) for pose in poses]
    moving = frames >= 8
    # his ears
    kind, keys = EAR_MOVES.get(name, (None, None))
    if kind:
        shape = EAR_SHAPES[kind]
        for i, pose in enumerate(poses):
            k = keyed(keys, i / max(frames - 1, 1))
            if abs(k) < 1e-4:
                continue
            with_head = solve(pose, bones)['head'].to_3x3() @ Mb['head'].to_3x3().inverted()
            for side, mirror in (('L', 1.0), ('R', -1.0)):
                for bone, share in (('ear.%s_01' % side, 0.6), ('ear.%s_02' % side, 0.4)):
                    R = turned(shape[0] * k * share, shape[1] * k * share * mirror, shape[2] * k * share * mirror, face)
                    spun(pose, bones, bone, with_head @ R @ with_head.inverted())
    # wind-up, overshoot and settle
    sharp, width = ALIVE['sharp']
    if moving and sharp:
        span = int(math.ceil(3 * width))
        kernel = np.exp(-0.5 * (np.arange(-span, span + 1) / width) ** 2)
        kernel /= kernel.sum()
        count = np.arange(frames)
        fade = np.ones(frames) if loop else np.clip(np.minimum(count, count[::-1]) / 4.0, 0.0, 1.0)   # a move starts and ends exactly as written

        def sharpened(X, wrap):
            P = np.pad(X, ((span, span), (0, 0)), mode='wrap' if wrap else 'edge')
            B = np.stack([np.convolve(P[:, c], kernel, mode='valid') for c in range(X.shape[1])], 1)
            return X + fade[:, None] * sharp * (X - B)
        up_ = bones[2]['root'].to_3x3()
        for bone in base:
            parts = [pose[bone].decompose() for pose in poses]
            Q = np.array([[q.w, q.x, q.y, q.z] for _, q, _ in parts])
            for i in range(1, frames):
                if Q[i] @ Q[i - 1] < 0:
                    Q[i] = -Q[i]
            L = np.array([list(l) for l, _, _ in parts])
            if np.abs(Q - Q[0]).max() < 1e-6 and np.abs(L - L[0]).max() < 1e-7:
                continue
            wrap = loop and Q[0] @ Q[-1] > 0
            Q2 = sharpened(Q, wrap)
            Q2 /= np.linalg.norm(Q2, axis=1)[:, None]
            L2 = sharpened(L, wrap)
            if bone == 'root':                             # he may dip before a hop, but only a hair: his back paws are on the floor
                low = min((up_ @ Vector(l)).z for l in L) - 0.015
                for i in range(frames):
                    z = (up_ @ Vector(L2[i])).z
                    if z < low:
                        L2[i] += np.array(up_.inverted() @ Vector((0.0, 0.0, low - z)))
            for i, pose in enumerate(poses):
                pose[bone] = Matrix.Translation(Vector(L2[i])) @ Quaternion(Q2[i]).to_matrix().to_4x4()
    # a held pose is never quite still
    drift = ALIVE['sway']
    holding = loop and name.startswith('pose-')            # a pose to hold: it goes round and round
    if drift and holding:
        for i, pose in enumerate(poses):
            spun(pose, bones, 'head', turned(x=0.7 * drift * math.sin(2 * math.pi * i / frames + 1.0),
                                             y=drift * math.sin(2 * math.pi * i / frames), face=face))
    if drift and frames >= 30 and not loop:
        for i, pose in enumerate(poses):
            t = i / (frames - 1)
            k = ease(min(t, 1.0 - t) / 0.2)
            if k:
                spun(pose, bones, 'head', turned(x=0.7 * drift * k * math.sin(2 * math.pi * i / 87.0 + 1.0),
                                                 y=drift * k * math.sin(2 * math.pi * i / 69.0), face=face))
    # his body follows his head, and his front paws stay where they stand
    share = ALIVE['body']
    spine = [bone for bone in ('spine2', 'spine_03', 'spine_04', 'chest') if bone in base]
    legs = [chain('clavicle.' + side) for side in ('L', 'R')]
    turns = []
    F = Matrix.Rotation(math.radians(-face), 3, 'Z')
    for pose in poses:
        M = solve(pose, bones)
        e = (F.inverted() @ M['head'].to_3x3() @ Mb['head'].to_3x3().inverted() @ F).to_euler('YXZ')
        nod, tilt, turn_ = math.degrees(e.x), math.degrees(e.y), math.degrees(e.z)
        turns.append(turn_)
        if max(abs(nod), abs(tilt), abs(turn_)) < 0.05 or not spine:
            continue
        facing_ = M['head'].to_3x3()
        for bone in spine:
            spun(pose, bones, bone, turned(nod * share[2] / len(spine), tilt * share[1] / len(spine), turn_ * share[0] / len(spine), face))
        now = solve(pose, bones)
        held(pose, now, bones, 'head', Matrix.Translation(now['head'].to_translation()) @ facing_.to_4x4())   # his head looks where it was told to
        now = solve(pose, bones)
        carried_ = now['root'] @ Mb['root'].inverted()     # a hop takes his paws with it
        for limb in ([] if name in FREE else legs):
            upper, lower, wrist = limb[1], limb[2], limb[3]
            stood = {bone: carried_ @ Mb[bone] for bone in limb[3:]}
            off = (M[wrist].to_translation() - stood[wrist].to_translation()).length
            grip = 1.0 - max(0.0, min(1.0, (off - 0.02) / 0.06))   # a paw in the air goes with his shoulder instead
            if grip <= 0.0:
                continue
            want = {}
            for bone in limb[3:]:
                l0, q0, _ = now[bone].decompose()
                l1, q1, _ = stood[bone].decompose()
                want[bone] = Matrix.Translation(l0.lerp(l1, grip)) @ q0.slerp(q1, grip).to_matrix().to_4x4()
            reach(pose, bones, upper, lower, wrist, want[wrist].to_translation())
            now = solve(pose, bones)
            for bone in limb[3:]:
                held(pose, now, bones, bone, want[bone])
                now[bone] = want[bone]
    if frames < 3:
        return poses, looks
    # his face numbers, with blinks and breathing
    wide = 6 + len(BROWS) + 1 + len(MOUTHS) + len(GAZE)
    air = 6 + len(BROWS)                                   # where his breath is among the numbers; his mouth's two come after it
    out = []
    for i in range(frames):
        look = looks[i] if looks is not None else 0.0
        look = list(look) if isinstance(look, (tuple, list)) else [look]
        out.append(look + [0.0] * (wide - len(look)))
    lids = lambda i: max(out[i][0], out[i][1], out[i][2], out[i][4])
    marks, span = [], 6
    swing = [abs(turns[min(i + span, frames - 1)] - turns[max(i - span, 0)]) for i in range(frames)]
    for i in range(span, frames - span):
        if swing[i] >= 12.0 and swing[i] >= max(swing[max(0, i - span):i + span + 1]) - 1e-9:
            marks.append(i)                                # the middle of a big turn of his head
    marks.sort(key=lambda i: -swing[i])                    # the biggest turn first: turns close together share one blink
    if frames >= ALIVE['blink'] and not loop and not any(lids(i) > 0.05 for i in range(frames)):
        marks += [int(frames * 0.36)] + ([int(frames * 0.78)] if frames >= 2 * ALIVE['blink'] + 10 else [])
    kept = []
    for mark in marks:
        if 4 <= mark < frames - 6 and all(abs(mark - k) >= 30 for k in kept) \
                and not any(lids(i) > 0.05 for i in range(mark - 3, mark + 5)):
            kept.append(mark)
            for j, v in enumerate(BLINKS):
                out[mark - 2 + j][0] = max(out[mark - 2 + j][0], v)
    if holding and max(lids(i) for i in range(frames)) < 0.6:   # and he blinks once each time round, unless his eyes are shut
        mark = int(frames * 0.4)
        for j, v in enumerate(BLINKS):
            out[mark - 2 + j][0] = max(out[mark - 2 + j][0], v)
    size, breath = ALIVE['breath']
    breaths = max(1, int(round(frames / breath))) if (loop or frames >= breath // 2) else 0
    deep = min(1.0, frames / float(breath))                # a short move has room only for a shallow breath
    for i in range(frames):
        out[i][air] = deep * (0.5 - 0.5 * math.cos(2 * math.pi * breaths * i / (frames if loop else frames - 1)))
    shapes = MOUTH_MOVES.get(name)
    if shapes:
        for i in range(frames):
            t = i / max(frames - 1, 1)
            for k in range(len(MOUTHS)):
                out[i][air + 1 + k] = keyed([(when, values[k]) for when, values in shapes], t)
    where = GAZE_MOVES.get(name)
    if where:
        for i in range(frames):
            t = i / max(frames - 1, 1)
            for k in range(len(GAZE)):
                out[i][air + 1 + len(MOUTHS) + k] = keyed([(when, values[k]) for when, values in where], t)
    return poses, [tuple(look) for look in out]


def blink_in(action, marks):
    """Give one of Meshy's motions blinks, at the frames in `marks`."""
    a = arm()
    a.animation_data_create()
    a.animation_data.action = action
    if action.slots:
        a.animation_data.action_slot = action.slots[0]
    first, last = int(action.frame_range[0]), int(action.frame_range[1])
    shut = {frame: 0.0 for frame in range(first, last + 1)}
    for mark in marks:
        for j, v in enumerate(BLINKS):
            if first < mark - 2 + j < last:
                shut[mark - 2 + j] = v
    for frame, v in shut.items():
        a[LIDS[0]] = v
        a.keyframe_insert('["%s"]' % LIDS[0], frame=frame)
    a[LIDS[0]] = 0.0
    a.animation_data.action = None


def _grow(m, times=1):
    """A set of points in a picture, grown by one point all round, `times` times."""
    for _ in range(times):
        m = m | np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)
    return m
def _soft(a, times=2):
    """A set of points in a picture as numbers, 1 inside, softened at its edge."""
    a = a.astype('f4')
    for _ in range(times):
        a = (a + np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1)) / 5.0
    return a
def _filled(picture, todo, known, rounds):
    """A copy of a picture with the points in `todo` filled in from the `known` ones beside them, then smoothed."""
    out, done = picture.copy(), known.copy()

    def beside(use):
        total, many = np.zeros_like(out), np.zeros(use.shape, 'f4')
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            total += np.roll(out * use[..., None], (dy, dx), (0, 1))
            many += np.roll(use, (dy, dx), (0, 1))
        return total, many
    for _ in range(600):
        total, many = beside(done)
        fill = todo & ~done & (many > 0)
        if not fill.any():
            break
        out[fill] = total[fill] / many[fill][:, None]
        done |= fill
    use = (todo & done) | known
    for _ in range(rounds):
        total, many = beside(use)
        again = todo & done & (many > 0)
        out[again] = total[again] / many[again][:, None]
    return out


def _blobs(m):
    """The separate patches of a set of points in a picture, biggest first."""
    left, out = m.copy(), []
    while left.any():
        y, x = np.argwhere(left)[0]
        one = np.zeros_like(m)
        one[y, x] = True
        while True:
            more = _grow(one) & left
            if more.sum() == one.sum():
                break
            one = more
        out.append(one)
        left &= ~one
    return sorted(out, key=lambda b: -int(b.sum()))


def pupils(sheet):
    """Split the picture of his two eyes (rows of points, each red, green, blue, 1) into the eyes with no pupils and
    the pupils on their own.

    Returns the first (its fourth number is where a pupil may show: inside the eye), the second (its fourth number is
    where the pupil is), the glints on their own (white; the fourth number is how strong), and what was found.

    The glints are light on the surface of his eye, so they stay where they are when the eye turns: left on the
    pupil they moved with it and read as marks on it. Each is lifted off (with the dark halo Meshy drew round it,
    which showed as a ring once the glint was gone) and the pupil under it filled from the dark beside it."""
    rgb = sheet[..., :3]
    hi, lo = rgb.max(2), rgb.min(2)
    sat = (hi - lo) / np.maximum(hi, 1e-6)
    gold = (sat > 0.3) & (hi > 0.6) & (rgb[..., 0] > rgb[..., 2] * 1.3)
    h, w = gold.shape
    pupil = np.zeros(gold.shape, bool)
    yy, xx = np.mgrid[0:h, 0:w]
    for x0 in (0, w // 2):                         # each eye in its own half
        half = np.zeros(gold.shape, bool)
        half[:, x0:x0 + w // 2] = True
        black = np.argwhere((hi < 0.14) & half & _grow(gold, 60))   # the blackest of the pupil, near the gold
        if len(black) < 50:
            continue
        cy, cx = np.median(black, axis=0)
        # out from the middle of the pupil in every direction: the first gold met is the pupil's edge, where there is
        # a good run of gold beyond it. Where his pupil runs down to the rim, or goes under the rim by his nose (it
        # does, seen from the front), there is none, or a sliver that is the rim's line and not the pupil's: so an
        # upright oval is fitted to the edge that was found, and carries on under the rim as the pupil does
        turn = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        step = np.arange(8, 0.45 * w // 2)
        px = np.clip((cx + np.outer(np.cos(turn), step)).round().astype(int), 0, w - 1)
        py = np.clip((cy + np.outer(np.sin(turn), step)).round().astype(int), 0, h - 1)
        met = gold[py, px] & half[py, px]
        first = met.argmax(1)
        run = np.minimum(first[:, None] + np.arange(6), met.shape[1] - 1)
        ok = met.any(1) & np.take_along_axis(met, run, 1).all(1)
        ex, ey = step[first][ok] * np.cos(turn[ok]), step[first][ok] * np.sin(turn[ok])
        if ok.sum() < 60:
            continue
        fit = np.linalg.lstsq(np.stack([ex * ex, ey * ey, ex, ey], 1), np.ones(len(ex)), rcond=None)[0]
        dx, dy = xx - cx, yy - cy
        pupil |= half & ((fit[0] * dx * dx + fit[1] * dy * dy + fit[2] * dx + fit[3] * dy) < 1.05) & (np.hypot(dx, dy) < 0.5 * w // 2)
    dull = gold | ((rgb[..., 0] > rgb[..., 2] * 1.25) & (hi > 0.5))      # gold, and the dull line of it under his pupil
    light = (lo > 0.45) & (sat < 0.25) & pupil                           # his glints: grey or white, inside a pupil
    glint = _grow(~_grow(~light, 3), 4) & light                          # the blobs of them: specks and thin edges go
    ink = (hi < 0.16) | ((sat > 0.45) & (hi < 0.55) & ~gold)             # the pupil's own paint: black, brown lower down, and its olive edge
    under = pupil & ~ink & ~dull & (sat < 0.35) & ~_grow(glint, 8)       # the part of a pupil's oval that is under the rim: the picture there is the rim's fur
    inside = ~_grow(~_grow(dull | (pupil & ~under), 2), 2)
    # the eye without its pupil: where the pupil was is filled with the gold round it, then smoothed. Only solid gold
    # is copied from: the thin line of it under his pupil is dull, and the rim beside it is dark
    todo = _grow(pupil, 3) & inside
    bare = _filled(sheet, todo, ~_grow(~gold, 2) & ~todo, 150)
    bare[..., 3] = _soft(~_grow(~inside, 2), 2)     # where a pupil may show: a hair inside the rim, soft at its edge
    # the pupil on its own: whatever in its oval is not its own paint (gold caught at its edge, his glints with the
    # dark halo Meshy drew round each, the rim's fur where the pupil goes under it) is painted over from its own paint
    # beside it. (Filled from the pupil's edge as well, a pale smudge was left where the big glint had been.)
    lifted = (_grow(glint, 6) | _grow(light, 2)) & pupil
    own = pupil & ink & ~lifted
    alone = _filled(sheet, pupil & ~own, own, 40)
    lo2 = lo
    alone[..., 3] = _soft(pupil, 1)
    # Each glint is drawn again as a clean disc. His biggest sits on the pupil's edge and Meshy painted only the
    # part of it over the pupil: once the pupil moved from under it, it was a chipped pale patch. The disc is fitted
    # to the glint's edge away from the pupil's own edge, and is as strong as the glint was, worked out as light:
    # the paint mixes as light, not as the picture's numbers (as numbers, his grey glints came out white)
    strength = np.clip((lo2 ** 2.2 - 0.006) / 0.994, 0.0, 1.0)
    shine = np.ones_like(sheet)
    shine[..., 3] = 0.0
    deep = ~_grow(~pupil, 5)
    for blob in _blobs(glint):
        if blob.sum() < 12:
            continue
        edge = blob & _grow(~blob)
        use = edge & deep if (edge & deep).sum() >= 8 else edge
        py, px = np.where(use)
        fit = np.linalg.lstsq(np.c_[2.0 * px, 2.0 * py, np.ones(len(px))], (px ** 2 + py ** 2).astype('f8'), rcond=None)[0]
        size, plain = float(np.sqrt(max(fit[2] + fit[0] ** 2 + fit[1] ** 2, 0.0))), float(np.sqrt(blob.sum() / np.pi))
        disc = np.clip(size - np.hypot(xx - fit[0], yy - fit[1]) + 0.5, 0.0, 1.0) if 0.6 * plain < size < 1.7 * plain else blob.astype('f4')
        most = strength[blob]
        shine[..., 3] = np.maximum(shine[..., 3], disc * float(np.median(most[most > 0.5 * most.max()])))
    shine[..., 3] *= bare[..., 3]                                              # only inside the eye
    return bare, alone, shine, {'gold': float(gold.mean()), 'pupil': float(pupil.mean()), 'glint': float((shine[..., 3] > 0.1).mean())}


def gaze():
    """Let his pupils move. His eyes are paint on his skin, and Meshy's paint is in scraps, so it cannot be slid.

    Each eye's paint is copied into a picture laid out as the eye is seen from straight in front of his face, domed
    as it is (his right eye in the left half, his left in the right half), and that is split in three (pupils()):
    the eye with no pupil, gold right across; the pupil on its own; and its glints. Inside each eye his skin takes
    its colour from the first, with the pupil laid over it, slid by the numbers GAZE on his bones, and the glints
    over that, never moved. His lids, painted over the top later, are untouched.

    The pupil is slid whole across that front view, so from the front it stays the clean oval Meshy painted, and it
    is made narrower the way it has gone, as a disc on a ball is when the ball turns (GAZE_BALL). Each eye goes less
    far toward his nose than away, and its own distance up and down (GAZE_REACH).

    What was tried before this, and why it went: stretching the one picture (the pupil bent and read as nothing);
    sliding the pupil flat (one shape wherever it was, like a sticker); turning a ball laid out flat across his
    face (the pupil was sheared as it crossed his domed eye: a leaning egg); turning a ball fitted to his domed eye
    (clean across, but his dome is not a ball and his pupil not a round patch on one, so it squared off going
    down, and the eye's own picture tore at its rim)."""
    a, ob, me = arm(), skin(), skin().data
    n = len(me.vertices)

    def spot(name):
        values = np.zeros(n, 'f4')
        me.attributes[name].data.foreach_get('value', values)
        return values.astype('f8')
    smooth_ = lambda t: np.clip(t, 0.0, 1.0) ** 2 * (3 - 2 * np.clip(t, 0.0, 1.0))
    far, side = spot('j5_eyem'), spot('j5_side')
    near = far < 1.2
    size = 384
    # where each point of an eye is as seen from in front of his face: across (toward his left) and up from the
    # eye's middle, in half-widths of the eye. Measured on his eyes as domed, so that it is what a viewer sees
    co = np.zeros(n * 3, 'f4')
    keys = me.shape_keys.key_blocks
    (keys.get('eyeball') or keys['Basis']).data.foreach_get('co', co)
    co = co.reshape(-1, 3).astype('f8')
    across, up, seen = np.zeros(n), np.zeros(n), {}
    for which_, label in ((0.0, 'his right'), (1.0, 'his left')):
        mine = (np.abs(side - which_) < 0.5) & (far < 2.0)
        p = co[mine & (far < 1.0)]
        middle, half_ = (p.min(0) + p.max(0)) / 2, (p[:, 0].max() - p[:, 0].min()) / 2
        across[mine], up[mine] = (co[mine, 0] - middle[0]) / half_, (co[mine, 2] - middle[2]) / half_
        seen[label] = {'half_width': round(float(half_), 4)}
    u = np.where(near, (side + np.clip(across / GAZE_SPAN, -1, 1) * 0.5 + 0.5) / 2, 0.0)
    v = np.where(near, np.clip(up / GAZE_SPAN, -1, 1) * 0.5 + 0.5, 0.0)
    mat = me.materials[0]
    nt = mat.node_tree
    tex = next(k for k in nt.nodes if k.type == 'TEX_IMAGE' and k.image and k.outputs['Color'].is_linked)
    img = tex.image
    w0, h0 = img.size
    src = np.zeros(w0 * h0 * 4, 'f4')
    img.pixels.foreach_get(src)
    src = src.reshape(h0, w0, 4)
    me.calc_loop_triangles()
    count = len(me.loop_triangles)
    tv, tl = np.zeros(count * 3, 'i4'), np.zeros(count * 3, 'i4')
    me.loop_triangles.foreach_get('vertices', tv)
    me.loop_triangles.foreach_get('loops', tl)
    tv, tl = tv.reshape(-1, 3), tl.reshape(-1, 3)
    uv = np.zeros(len(me.loops) * 2, 'f4')
    me.uv_layers.active.data.foreach_get('uv', uv)
    uv = uv.reshape(-1, 2).astype('f8')

    def drawn(us, vs):
        """His eyes' paint as a picture, each point of his skin put where `us`, `vs` say."""
        sheet, done = np.zeros((size, 2 * size, 4), 'f4'), np.zeros((size, 2 * size), bool)
        px, py = us * 2 * size - 0.5, vs * size - 0.5
        for t in np.where((far[tv] < 1.1).all(1))[0]:
            i0, i1, i2 = tv[t]
            if not side[i0] == side[i1] == side[i2]:
                continue
            x, y = px[[i0, i1, i2]], py[[i0, i1, i2]]
            area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
            if abs(area) < 1e-9:
                continue
            xs = np.arange(max(0, int(math.floor(x.min()))), min(2 * size - 1, int(math.ceil(x.max()))) + 1)
            ys = np.arange(max(0, int(math.floor(y.min()))), min(size - 1, int(math.ceil(y.max()))) + 1)
            if not len(xs) or not len(ys):
                continue
            gx, gy = np.meshgrid(xs, ys)
            b1 = ((gx - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (gy - y[0])) / area
            b2 = ((x[1] - x[0]) * (gy - y[0]) - (gx - x[0]) * (y[1] - y[0])) / area
            b0 = 1 - b1 - b2
            inside = (b0 >= -0.02) & (b1 >= -0.02) & (b2 >= -0.02)
            if not inside.any():
                continue
            at = b0[..., None] * uv[tl[t][0]] + b1[..., None] * uv[tl[t][1]] + b2[..., None] * uv[tl[t][2]]
            fx, fy = np.clip(at[..., 0] * w0 - 0.5, 0, w0 - 1.001), np.clip(at[..., 1] * h0 - 0.5, 0, h0 - 1.001)
            ix, iy = fx.astype(int), fy.astype(int)
            kx, ky = (fx - ix)[..., None], (fy - iy)[..., None]
            colour = (src[iy, ix] * (1 - kx) + src[iy, ix + 1] * kx) * (1 - ky) + (src[iy + 1, ix] * (1 - kx) + src[iy + 1, ix + 1] * kx) * ky
            sheet[gy[inside], gx[inside]] = colour[inside]
            done[gy[inside], gx[inside]] = True
        for _ in range(24):                                    # the few specks no triangle reached take their neighbours' colour
            if done.all():
                break
            total, many = np.zeros_like(sheet), np.zeros(done.shape)
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                total += np.roll(sheet * done[..., None], (dy, dx), (0, 1))
                many += np.roll(done, (dy, dx), (0, 1))
            fill = ~done & (many > 0)
            sheet[fill] = total[fill] / many[fill][:, None]
            done |= fill
        sheet[..., 3] = 1.0
        return sheet
    bare, alone, shine, found = pupils(drawn(u, v))
    pictures = []
    for label, pixels in (('-eyes', bare), ('-pupils', alone), ('-glints', shine)):
        old = bpy.data.images.get(NAME + label)
        if old:
            bpy.data.images.remove(old)
        picture = bpy.data.images.new(NAME + label, 2 * size, size, alpha=True)
        picture.colorspace_settings.name = img.colorspace_settings.name
        picture.alpha_mode = 'CHANNEL_PACKED'
        picture.pixels.foreach_set(np.clip(pixels, 0.0, 1.0).astype('f4').ravel())
        picture.pack()
        pictures.append(picture)
    middles = []                                           # the middle of each pupil in the picture: it is made narrower about this
    for x0 in (0, size):
        ys, xs = np.where(alone[:, x0:x0 + size, 3] > 0.5)
        middles.append(((x0 + xs.mean() + 0.5) / (2 * size), (ys.mean() + 0.5) / size, 0.0) if len(xs) else ((x0 + size / 2) / (2 * size), 0.5, 0.0))
    mask = np.where(near, 1.0 - smooth_((far - 0.86) / 0.10), 0.0)   # the pictures are used over the whole eye and its rim (a pupil left in the old paint at the rim showed as a smudge), the old paint beyond
    place = np.zeros((n, 3), 'f4')
    place[:, 0], place[:, 1] = u, v
    att = me.attributes.get('j5_gzuv') or me.attributes.new('j5_gzuv', 'FLOAT_VECTOR', 'POINT')
    att.data.foreach_set('vector', place.ravel())
    att = me.attributes.get('j5_gzm') or me.attributes.new('j5_gzm', 'FLOAT', 'POINT')
    att.data.foreach_set('value', mask.astype('f4'))
    for spare in ('j5_gzn', 'j5_gzuv2'):                   # Eevee draws a skin with more than 15 kinds of number on its points bright pink all over, and his is nearly full: keep none that are not used
        if me.attributes.get(spare):
            me.attributes.remove(me.attributes[spare])
    for k in [k for k in nt.nodes if k.name.startswith('jg_')]:
        nt.nodes.remove(k)

    def node(kind, label, **props):
        k = nt.nodes.new(kind)
        k.name = 'jg_' + label
        for key, value in props.items():
            setattr(k, key, value)
        return k
    goes_to = [link.to_socket for link in tex.outputs['Color'].links]

    def sum_(label, operation, first, second):
        k = node('ShaderNodeMath', label, operation=operation)
        for socket, value in zip(k.inputs, (first, second)):
            if isinstance(value, (int, float)):
                socket.default_value = value
            else:
                nt.links.new(value, socket)
        return k.outputs[0]

    def split(label, value, plus, minus):
        """A distance from a number: `plus` at 1 and `minus` the other way at -1."""
        return sum_(label + '_all', 'ADD', sum_(label + '_p', 'MULTIPLY', sum_(label + '_pos', 'MAXIMUM', value, 0.0), plus),
                    sum_(label + '_m', 'MULTIPLY', sum_(label + '_neg', 'MINIMUM', value, 0.0), minus))
    values = []
    for prop in GAZE:
        if prop not in a:
            a[prop] = 0.0
        a.id_properties_ui(prop).update(min=-1.0, max=1.0)
        value = node('ShaderNodeValue', prop)
        drv = value.outputs[0].driver_add('default_value').driver
        drv.type = 'SUM'
        var = drv.variables.new()
        var.type = 'SINGLE_PROP'
        var.targets[0].id = a
        var.targets[0].data_path = '["%s"]' % prop
        values.append(value.outputs[0])
    # across: each eye goes less far toward his nose than away from it. In the picture his right eye is on the left,
    # its nose side to the right, which is the way his left is: so for that eye "toward his left" is "toward his nose"
    which = sum_('which', 'GREATER_THAN', node('ShaderNodeAttribute', 'side', attribute_name='j5_side').outputs['Fac'], 0.5)
    nose = sum_('nose', 'SUBTRACT', 1.0, sum_('nose2', 'MULTIPLY', which, 2.0))       # 1 for his right eye, -1 for his left
    gone_x = sum_('gone_x', 'MULTIPLY', split('lx', sum_('inward', 'MULTIPLY', values[0], nose), *GAZE_REACH[0]), nose)   # how far the pupil has gone, in half-widths of the eye
    gone_y = split('ly', values[1], *GAZE_REACH[1])

    def thin(label, gone):
        """How much narrower a disc on a ball of radius GAZE_BALL looks when it has gone that far from the middle."""
        part = sum_(label + '_part', 'MULTIPLY', gone, 1.0 / GAZE_BALL)
        return sum_(label, 'SQRT', sum_(label + '_left', 'SUBTRACT', 1.0, sum_(label + '_sq', 'MULTIPLY', part, part)), 0.0)

    def three(label, x, y, z=None):
        k = node('ShaderNodeCombineXYZ', label)
        for socket, value in zip(k.inputs, (x, y, z)):
            if value is None:
                continue
            if isinstance(value, (int, float)):
                socket.default_value = value
            else:
                nt.links.new(value, socket)
        return k.outputs[0]

    def both_(label, operation, first, second):
        k = node('ShaderNodeVectorMath', label, operation=operation)
        nt.links.new(first, k.inputs[0])
        nt.links.new(second, k.inputs[1])
        return k.outputs[0]
    where = node('ShaderNodeAttribute', 'place', attribute_name='j5_gzuv')
    centre_ = node('ShaderNodeMix', 'centre', data_type='VECTOR')        # the middle of this eye's pupil in the picture
    nt.links.new(which, centre_.inputs[0])
    centre_.inputs[4].default_value, centre_.inputs[5].default_value = middles[0], middles[1]
    step_ = three('step', sum_('step_x', 'MULTIPLY', gone_x, 0.5 / GAZE_SPAN / 2), sum_('step_y', 'MULTIPLY', gone_y, 0.5 / GAZE_SPAN), 0.0)   # from half-widths of the eye to places in the picture
    # the pupil is looked up where this point was before the pupil went, and before it was made narrower about its middle
    from_middle = both_('off', 'SUBTRACT', both_('rel', 'SUBTRACT', where.outputs['Vector'], centre_.outputs[1]), step_)
    moved = both_('moved', 'ADD', both_('wide', 'DIVIDE', from_middle, three('thin', thin('thin_x', gone_x), thin('thin_y', gone_y), 1.0)), centre_.outputs[1])
    eye = node('ShaderNodeTexImage', 'eyes', image=pictures[0], interpolation='Linear', extension='EXTEND')
    nt.links.new(where.outputs['Vector'], eye.inputs['Vector'])
    pupil = node('ShaderNodeTexImage', 'pupils', image=pictures[1], interpolation='Linear', extension='EXTEND')
    nt.links.new(moved, pupil.inputs['Vector'])
    shows = sum_('shows', 'MULTIPLY', eye.outputs['Alpha'], pupil.outputs['Alpha'])   # the pupil shows only inside the eye: its rim and lids hide the rest
    whole = node('ShaderNodeMix', 'whole', data_type='RGBA')
    nt.links.new(shows, whole.inputs[0])
    nt.links.new(eye.outputs['Color'], whole.inputs[6])
    nt.links.new(pupil.outputs['Color'], whole.inputs[7])
    # his glints, laid over last and never moved: light on the eye stays put when the eye turns under it
    glints = node('ShaderNodeTexImage', 'glints', image=pictures[2], interpolation='Linear', extension='EXTEND')
    nt.links.new(where.outputs['Vector'], glints.inputs['Vector'])
    lit_ = node('ShaderNodeMix', 'lit', data_type='RGBA')
    nt.links.new(glints.outputs['Alpha'], lit_.inputs[0])
    nt.links.new(whole.outputs[2], lit_.inputs[6])
    lit_.inputs[7].default_value = (1.0, 1.0, 1.0, 1.0)
    mix = node('ShaderNodeMix', 'mix', data_type='RGBA')
    nt.links.new(node('ShaderNodeAttribute', 'mask', attribute_name='j5_gzm').outputs['Fac'], mix.inputs[0])
    nt.links.new(tex.outputs['Color'], mix.inputs[6])
    nt.links.new(lit_.outputs[2], mix.inputs[7])
    for socket in goes_to:
        nt.links.new(mix.outputs[2], socket)
    return {'triangles': int((far[tv] < 1.1).all(1).sum()), 'pupil': found['pupil'], 'glint': found['glint'], 'seen': seen, 'middles': [[round(float(c), 4) for c in m[:2]] for m in middles]}


WOKE = {}                                # how many of his face's drivers wake() found stopped, by where it was called


def wake(label=None):
    """Set going again any of the drivers of his face, mouth, eyes and hat that have stopped.

    Blender marks a driver as broken the first time it fails to read one of its numbers, and never tries it again:
    his face then stays as it was whatever a motion says. This looks at every driver on his skin's shapes, its
    paint and the things made for him, and gives each stopped one its expression again, which starts it. Returns
    how many had stopped (0 is the healthy answer), and notes it in WOKE under `label`."""
    me = skin().data
    owners = [me.shape_keys] + [m.node_tree for m in me.materials if m and m.node_tree] + list(bpy.data.collections[NAME].objects)
    count = 0
    for owner in owners:
        if owner is not None and owner.animation_data:
            for fc in owner.animation_data.drivers:
                if not fc.driver.is_valid:
                    fc.driver.expression = fc.driver.expression
                    count += 1
    if label:
        WOKE[label] = count
    return count


def refit(action, fix, bones):
    """Go through one of Meshy's motions a frame at a time: `fix(frame)` poses him again, and `bones` are keyed as it left them."""
    a = arm()
    a.animation_data_create()
    a.animation_data.action = action
    if action.slots:
        a.animation_data.action_slot = action.slots[0]
    last = {}
    for frame in range(int(action.frame_range[0]), int(action.frame_range[1]) + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        fix(frame)
        bpy.context.view_layer.update()
        for bone in bones:
            pb = a.pose.bones[bone]
            q = pb.rotation_quaternion.copy()
            if bone in last and last[bone].dot(q) < 0:
                q.negate()
                pb.rotation_quaternion = q
            last[bone] = q
            pb.keyframe_insert('rotation_quaternion', frame=frame)
            pb.keyframe_insert('location', frame=frame)
    a.animation_data.action = None


def nose_up():
    """How far above level his nose points, as posed, in degrees, and the way it points along the floor."""
    a = arm()
    way = a.matrix_world.to_3x3() @ a.pose.bones['head'].matrix.to_3x3() @ a.data.bones['head'].matrix_local.to_3x3().inverted() @ Vector((0.0, -1.0, 0.0))
    way.normalize()
    return math.degrees(math.asin(max(-1.0, min(1.0, way.z)))), Vector((way.x, way.y, 0.0))


REARED = (20.0, 60.0)                    # he is reared up, in Meshy's motions, as his nose goes from this many degrees above level to this many
BEG = ((0.0, -0.92, 0.38), (0.0, -0.60, -0.80))   # where his forearm and his paw point as he begs: forearm out and a little up from a bent elbow, paw hanging


def reared():
    """How far he is reared up as posed (0 to 1), by how far up his nose points, and the way he faces along the floor."""
    pitch, flat = nose_up()
    return ease((pitch - REARED[0]) / (REARED[1] - REARED[0])), flat


def reared_through(action):
    """How far he is reared up (0 to 1) at each frame of one of Meshy's motions, as Meshy made it."""
    a = arm()
    a.animation_data_create()
    a.animation_data.action = action
    if action.slots:
        a.animation_data.action_slot = action.slots[0]
    out = {}
    for frame in range(int(action.frame_range[0]), int(action.frame_range[1]) + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        out[frame] = reared()[0]
    a.animation_data.action = None
    return out


UPRIGHT = (('spine2', 0.6), ('spine_03', 1.0), ('spine_04', 1.2), ('chest', 1.2))   # the bones that bring him upright, and each one's share: his back only. His neck and head are left as Meshy set them on his chest, which is close to how they sit when he sits (with a share each, tried, his head was pushed forward and down off his neck)


def level_head(action, rear, down=58.0):
    """Keep his face in view through one of Meshy's motions that rears him up with his head thrown back.

    Meshy arches his whole back to do it: with his neck set as it is when he sits, his nose is still over 60 degrees
    up. So his neck alone cannot bring his face down without his head leaving his body (tried: it looked stuck on).
    While he is reared up (`rear`, from reared_through()) his back comes forward by `down` degrees, each bone of it
    a share (UPRIGHT), so he stands upright; his neck and head ride on it as they were, so his head sits on his
    neck as it does when he sits. It is the same amount on every frame, so each nod and sway Meshy gave him is
    still there."""
    a = arm()
    bones = [(name, share) for name, share in UPRIGHT if name in a.pose.bones]
    whole = sum(share for _, share in bones)
    to_arm = a.matrix_world.to_3x3().inverted()

    def fix(frame):
        how = rear.get(frame, 0.0)
        flat = nose_up()[1]
        if how <= 0.0 or flat.length < 1e-4:
            return
        axis = to_arm @ Vector((0.0, 0.0, 1.0)).cross(flat.normalized())
        for bone, share in bones:
            pb = a.pose.bones[bone]
            M = pb.matrix.copy()
            h = Matrix.Translation(M.to_translation())
            pb.matrix = h @ Matrix.Rotation(math.radians(down * how * share / whole), 4, axis) @ h.inverted() @ M
            bpy.context.view_layer.update()
    refit(action, fix, [name for name, _ in bones])


def beg_paws(action, rear):
    """Fold his front legs while he is reared up in one of Meshy's motions: elbows bent, paws hanging by his chest.

    Meshy's dog holds them straight out in front; on him that is two stiff arms and, under them, the patches where
    his legs were cut free. Now his upper legs stay against his chest as when he sits, and from the elbow his
    forearms come out with the paws hanging (BEG), the two bobbing together: a cat begging. Run this after
    level_head(), which moves his chest; `rear` is from reared_through()."""
    a = arm()
    limbs = {side: chain('clavicle.' + side) for side in ('L', 'R')}
    lower, wrist = BEG
    first = int(action.frame_range[0])
    sit()
    sat = snap()                                           # his upper front legs lie against his chest as they do when he sits: there the cut does not show

    def fix(frame):
        how = rear.get(frame, 0.0)
        if how <= 0.0:
            return
        was = snap()
        bob = 0.14 * math.sin(2 * math.pi * (frame - first) / 16.0)
        for side, mirror in (('L', 1.0), ('R', -1.0)):
            limb = limbs[side]
            for bone in limb[:2]:
                a.pose.bones[bone].matrix_basis = sat[bone].copy()
            bpy.context.view_layer.update()
            way = lambda v, lift=0.0: (v[0] * mirror, v[1], v[2] + lift)
            rest_ = limb[2:]
            for bone, child, v in zip(rest_, rest_[1:], [way(lower, bob)] * (len(rest_) - 3) + [way(wrist, bob)]):
                aim(bone, child, v)
        put(blend(was, snap(), how))
    refit(action, fix, [bone for limb in limbs.values() for bone in limb])


def look_in(action, looks):
    """Give one of Meshy's motions a face: `looks` is {a number on his bones: [(how far through, value), ...]}."""
    a = arm()
    a.animation_data_create()
    a.animation_data.action = action
    if action.slots:
        a.animation_data.action_slot = action.slots[0]
    first, last = int(action.frame_range[0]), int(action.frame_range[1])
    for prop, keys in looks.items():
        if prop not in a:
            continue
        for frame in range(first, last + 1):
            a[prop] = float(keyed(keys, (frame - first) / max(last - first, 1)))
            a.keyframe_insert('["%s"]' % prop, frame=frame)
        a[prop] = 0.0
    a.animation_data.action = None


def sit_tail(action, seated):
    """In one of Meshy's motions in which he sits down, lay his tail round him while he sits, as in his own sitting pose.

    Carried up as he stands (tail_up()), his tail stood in a hook behind him once he sat. It is laid round him at
    the frame `seated` (lay_tail()), and each frame takes as much of that as he is sitting: none standing, all sat."""
    a, bones = arm(), tail_bones()
    a.animation_data_create()
    a.animation_data.action = action
    if action.slots:
        a.animation_data.action_slot = action.slots[0]
    height = lambda: (a.matrix_world @ a.pose.bones[bones[0]].matrix).to_translation().z
    bpy.context.scene.frame_set(int(action.frame_range[0]))
    bpy.context.view_layer.update()
    stood = height()
    bpy.context.scene.frame_set(seated)
    bpy.context.view_layer.update()
    sat = height()
    kept = (LAID.pop(NAME, None), GROUND.pop(NAME, None))  # the floor and the fit are measured afresh, on this pose
    lay_tail()
    laid = {bone: a.pose.bones[bone].matrix_basis.copy() for bone in bones[1:]}
    LAID.pop(NAME, None)
    for store, value in zip((LAID, GROUND), kept):
        if value is not None:
            store[NAME] = value

    def fix(frame):
        w = ease(((stood - height()) / max(stood - sat, 1e-6) - 0.35) / 0.55)
        if w <= 0.0:
            return
        for bone in bones[1:]:
            pb = a.pose.bones[bone]
            l0, q0, _ = pb.matrix_basis.decompose()
            l1, q1, _ = laid[bone].decompose()
            pb.matrix_basis = Matrix.Translation(l0.lerp(l1, w)) @ q0.slerp(q1, w).to_matrix().to_4x4()
    refit(action, fix, bones[1:])


def cat_ways():
    """Make Meshy's motions his (review of 24 Nov, suggestions 22 to 24).

    Act cute: he sits up on his haunches with his head thrown right back, so his face cannot be seen. His head is
    brought level. Being petted: his tail is laid round him while he sits. And each gets a face: happy eyes at the
    height of act cute and while he is petted, brows up as he wags."""
    squeeze = lambda t0, t1: [(0, 0), (t0, 0), (t0 + 0.06, 1), (t1, 1), (t1 + 0.06, 0), (1, 0)]
    cute = bpy.data.actions[motion('cat_sajiao')]
    rear = reared_through(cute)
    level_head(cute, rear)
    beg_paws(cute, rear)
    eyes_ = squeeze(0.36, 0.6)
    _plead = [(0, 0), (0.12, -1.0), (0.34, -1.0), (0.42, -0.3), (0.6, -0.3), (0.68, -1.0), (0.88, -1.0), (1, 0)]
    look_in(cute, {LIDS[1]: eyes_, LIDS[2]: eyes_, JOY: eyes_,
                   BROWS[0]: [(0, 0), (0.12, 0.5), (0.88, 0.5), (1, 0)], BROWS[1]: [(0, 0), (0.12, 0.5), (0.88, 0.5), (1, 0)],
                   # begging: his brows' inner ends up while his eyes are open, easing while they are squeezed shut and happy
                   BROWS[2]: _plead, BROWS[3]: _plead})
    pet = bpy.data.actions[motion('dog_beifumo')]
    first, last = int(pet.frame_range[0]), int(pet.frame_range[1])
    sit_tail(pet, (first + last) // 2)
    eyes_ = squeeze(0.38, 0.72)
    look_in(pet, {LIDS[1]: eyes_, LIDS[2]: eyes_, JOY: eyes_,
                  BROWS[0]: [(0, 0), (0.3, 0.25), (0.8, 0.25), (1, 0)], BROWS[1]: [(0, 0), (0.3, 0.25), (0.8, 0.25), (1, 0)]})
    wag = bpy.data.actions[motion('dog_yaoweiba')]
    look_in(wag, {BROWS[0]: [(0, 0.3), (1, 0.3)], BROWS[1]: [(0, 0.3), (1, 0.3)]})


WHISKERS = dict(fan=((15.0, 0.29), (1.0, 0.33), (-13.0, 0.28)),   # each whisker: degrees up from level, and how long
                roots=((0.080, 0.008), (0.095, -0.006), (0.082, -0.020)),   # where each starts on his muzzle: how far out from the middle of his face, and how far above the corner of his mouth (below, if less than 0)
                back=26.0, droop=0.03,                             # swept back by this many degrees; how far the tip sags
                thick=(0.0026, 0.0005), sink=0.012, lift=0.3,                # how thick at root and tip (Ant, 25 Nov: half of the first go); how far under his skin the root starts
                colour=(0.80, 0.78, 0.76, 1.0))
UNWHISKER = dict(box=((0.07, 0.52), (-0.11, 0.07)), front=-0.5,   # the part of his face Meshy's painted whiskers cross: out from the middle, up and down from the corner of his mouth, and no further back than this
                 size=11, down=4, grey=0.11, fur=0.36, purple=-0.012, keep=0.03)   # see unwhisker()
PADS = ('pad_L', 'pad_R')                              # where the middle of each whisker pad is, kept on his skin by mend() for whiskers()


def mend(ob=None, rings=1, most=80):
    """Close the tears in his whisker pads, and note where the pads are.

    Meshy drew his whiskers as paint, and where each met his skin it left a tear: slits and scraps, about a hundred
    open edges, all on his two pads. Lit from the front they were small black gaps (Ant, 25 Nov: "holes in his mesh").
    The tears are tangled (two meet at a point, loose scraps lie in them), so they are not filled as they are: the
    faces round each are taken out, which leaves a clean rim a little further back, and that rim is given new skin.
    Along the seams of his picture Meshy keeps two points in one place, so "open" here goes by place, not by point.
    New skin takes its paint from the rim; where a rim lies across a seam the piece is painted flat from one corner.
    Run on him as he comes in, before anything is added to his skin. Returns what was done."""
    import bmesh
    ob = ob or skin()
    me = ob.data
    top = arm().data.bones['head']
    heads = {ob.vertex_groups[b.name].index for b in [top] + list(top.children_recursive) if b.name in ob.vertex_groups}
    bm = bmesh.new()
    bm.from_mesh(me)
    held = bm.verts.layers.deform.verify()
    uv = bm.loops.layers.uv.active
    place = {v: (round(v.co.x, 5), round(v.co.y, 5), round(v.co.z, 5)) for v in bm.verts if sum(w for g, w in v[held].items() if g in heads) > 0.5}

    def rims():
        seen = {}
        for e in bm.edges:
            p, q = e.verts
            if p in place and q in place and place[p] != place[q]:
                seen.setdefault(frozenset((place[p], place[q])), []).append(e)
        return [es for es in seen.values() if sum(len(e.link_faces) for e in es) == 1]
    first = rims()
    torn = {place[v] for es in first for e in es for v in e.verts}
    if not torn:
        bm.free()
        return {'torn_edges': 0}
    for side, ps in (('L', [p for p in torn if p[0] > 0]), ('R', [p for p in torn if p[0] < 0])):
        if ps:
            skin()[PADS[side == 'R']] = [sum(p[i] for p in ps) / len(ps) for i in range(3)]
    gone = []
    for _ in range(rings):
        gone = [f for f in bm.faces if any(place.get(v) in torn for v in f.verts)]
        torn |= {place[v] for f in gone for v in f.verts if v in place}
    bmesh.ops.delete(bm, geom=gone, context='FACES')
    place = {v: p for v, p in place.items() if v.is_valid}
    old = set(bm.faces)
    at, rim = {}, []
    for es in rims():
        e = next(k for k in es if k.link_faces)
        rim.append(e)
        for v in e.verts:
            at.setdefault(place[v], []).append(e)
    used, made, left = set(), [], 0
    for first_ in rim:
        if first_ in used:
            continue
        ring, nxt, shut = [first_.verts[0]], first_.verts[1], False
        where = place[nxt]
        used.add(first_)
        while True:
            if where == place[ring[0]]:
                shut = True
                break
            ring.append(nxt)
            on = [k for k in at.get(where, []) if k not in used]
            if len(on) != 1:
                break
            used.add(on[0])
            p, q = on[0].verts
            nxt = q if place[p] == where else p
            where = place[nxt]
        if shut and 3 <= len(ring) <= most:
            try:
                made.append(bm.faces.new(ring))
                continue
            except ValueError:
                pass
        left += 1
    sizes = [len(f.verts) for f in made]
    for f in made:
        for l in f.loops:                                  # facing the way its neighbour does: along the edge they share, the two run opposite ways
            beside = [k for k in l.edge.link_loops if k.face in old]
            if beside:
                if beside[0].vert is l.vert:
                    f.normal_flip()
                break
    new = bmesh.ops.triangulate(bm, faces=made)['faces']
    if uv:
        for f in new:
            got = []
            for l in f.loops:
                near = [k for k in l.vert.link_loops if k.face in old]
                got.append(near[0][uv].uv.copy() if near else None)
            base = next((g for g in got if g is not None), None)
            if base is None:
                continue
            apart = any(g is None or (g - base).length > 0.05 for g in got)
            for l, g in zip(f.loops, got):
                l[uv].uv = base if apart else g
    for f in new:
        f.smooth = True
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    me.update()
    return {'torn_edges': len(first), 'faces_out': len(gone), 'closed': sizes, 'left': left, 'new_faces': len(new)}


def unwhisker(ob=None):
    """Paint out the whiskers Meshy painted on his cheeks, now that he has whiskers of his own.

    They are pale thin streaks in his picture, running out from his whisker pads. Where the picture lies on that
    part of his face, each dot of fur is given the usual colour of the fur round it (the middle value of a patch
    UNWHISKER['size'] x UNWHISKER['down'] dots across). Fur only: a dot is left alone if it or its surroundings are
    not fur-coloured (his eye, nose, headset and its microphone, collar), or if it is darker than the fur round it
    by more than `keep`, which is his mouth line and the dots on his pads. Needs the middle of each pad from mend(). Returns how many dots
    of the picture changed."""
    ob = ob or skin()
    me = ob.data
    U = UNWHISKER
    img = next(k.image for k in me.materials[0].node_tree.nodes if k.type == 'TEX_IMAGE' and not k.inputs[0].links and k.outputs['Color'].links)
    w, h = img.size
    px = np.zeros(w * h * 4, 'f4')
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    pl, pr = skin()[PADS[0]], skin()[PADS[1]]
    mid, level = (pl[0] + pr[0]) / 2.0, (pl[2] + pr[2]) / 2.0
    n = len(me.vertices)
    co = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    out_, up = np.abs(co[:, 0] - mid), co[:, 2] - level
    there = (co[:, 1] < U['front']) & (out_ > U['box'][0][0]) & (out_ < U['box'][0][1]) & (up > U['box'][1][0]) & (up < U['box'][1][1])
    me.calc_loop_triangles()
    tv, tl = np.zeros(len(me.loop_triangles) * 3, 'i4'), np.zeros(len(me.loop_triangles) * 3, 'i4')
    me.loop_triangles.foreach_get('vertices', tv)
    me.loop_triangles.foreach_get('loops', tl)
    uv = np.zeros(len(me.loops) * 2, 'f4')
    me.uv_layers[0].data.foreach_get('uv', uv)
    uv = uv.reshape(-1, 2) * (w, h)
    mask = np.zeros((h, w), bool)
    for t in tl.reshape(-1, 3)[there[tv.reshape(-1, 3)].all(1)]:       # every dot of the picture inside each of those faces
        a, b, c = uv[t]
        x0, x1 = int(max(0, math.floor(min(a[0], b[0], c[0])))), int(min(w - 1, math.ceil(max(a[0], b[0], c[0]))))
        y0, y1 = int(max(0, math.floor(min(a[1], b[1], c[1])))), int(min(h - 1, math.ceil(max(a[1], b[1], c[1]))))
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-9 or x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        l1 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / d
        l2 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / d
        mask[y0:y1 + 1, x0:x1 + 1] |= (l1 >= -0.02) & (l2 >= -0.02) & (l1 + l2 <= 1.02)
    k, half = U['down'], U['size'] // 2
    small = px[:h // k * k, :w // k * k, :3].reshape(h // k, k, w // k, k, 3).mean((1, 3))     # the picture, k times smaller: the patch is then cheap to take
    want = mask[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k).any((1, 3))
    def furry(c):                                          # his fur is a dark purple-grey: less green than red and blue, not strongly coloured, not the black round the picture's pieces
        return (c[..., 1] - (c[..., 0] + c[..., 2]) / 2.0 < U['purple']) & (c.max(-1) - c.min(-1) < U['grey']) & (c.mean(-1) > 0.05) & (c.mean(-1) < U['fur'])
    usual = small.copy()
    known = np.zeros(small.shape[:2], bool)
    ys, xs = np.nonzero(want)
    only = np.where(furry(small)[..., None], small, np.nan)    # the patch's middle value is taken over fur alone
    pad = np.pad(only, ((half, half), (half, half), (0, 0)), mode='edge')
    for i in range(0, len(ys), 4000):
        yy, xx = ys[i:i + 4000], xs[i:i + 4000]
        patch = np.stack([pad[yy + dy, xx + dx] for dy in range(U['size']) for dx in range(U['size'])], 1)
        enough = np.isfinite(patch[..., 0]).sum(1) > U['size']
        with np.errstate(all='ignore'):
            value = np.nanmedian(patch, 1)
        usual[yy[enough], xx[enough]] = value[enough]
        known[yy[enough], xx[enough]] = True
    for _ in range(2):                                     # evened out where it is known, so its steps do not show
        total, count = np.zeros_like(usual), np.zeros(usual.shape[:2])
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ok = np.roll(np.roll(known, dy, 0), dx, 1)
                total += np.where(ok[..., None], np.roll(np.roll(usual, dy, 0), dx, 1), 0.0)
                count += ok
        usual = np.where(known[..., None], total / np.maximum(count, 1)[..., None], usual)
    big, sure = np.repeat(np.repeat(usual, k, 0), k, 1), np.repeat(np.repeat(known, k, 0), k, 1)
    soft = sum(np.roll(np.roll(big, dy, 0), dx, 1) for dy in range(-2, 2) for dx in range(-2, 2)) / 16.0
    part = px[:big.shape[0], :big.shape[1], :3]
    hit = mask[:big.shape[0], :big.shape[1]] & sure & furry(part) & furry(soft) & (part.mean(-1) > soft.mean(-1) - U['keep'])
    part[hit] = soft[hit]
    img.pixels.foreach_set(px.ravel())
    img.update()
    img.pack()
    return int(hit.sum())


def whiskers(pads=None, ob=None):
    """Give him whiskers of his own: three a side, thin tapering strands fanned out from his whisker pads.

    Meshy's were paint and tears in his skin (mend()). These are a thing of their own held by his head bone, as his
    hat is, so they turn with his head; they do not follow his cheeks as his face moves, which is why each is rooted
    a little under his skin. `pads` is where the middle of each pad is (mend() leaves it on his skin). Make them with
    his bones at rest. Returns their size."""
    import bmesh
    from mathutils.bvhtree import BVHTree
    a, ob = arm(), ob or skin()
    name = NAME + '-whiskers'
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    pads = pads or {'L': list(skin()[PADS[0]]), 'R': list(skin()[PADS[1]])}
    W = WHISKERS
    src = bmesh.new()
    src.from_mesh(ob.data)
    tree = BVHTree.FromBMesh(src)
    bm = bmesh.new()
    around, along = 6, 9
    mid, level = (pads['L'][0] + pads['R'][0]) / 2.0, (pads['L'][2] + pads['R'][2]) / 2.0     # the middle of his face, and the height of the corners of his mouth: the tears were there
    for side, sign in (('L', 1.0), ('R', -1.0)):
        for (rise, length), (out, up_) in zip(W['fan'], W['roots']):
            # the same place each side, found by looking straight at his face from in front: the first go rooted them
            # at the nearest skin to the tears, which on one side was the corner of his mouth (Ant, 25 Nov)
            on, normal, _, _ = tree.ray_cast(Vector((mid + sign * out, -5.0, level + up_)), Vector((0.0, 1.0, 0.0)))
            if on is None:
                continue
            e, s = math.radians(rise), math.radians(W['back'])
            way = Vector((sign * math.cos(e) * math.cos(s), math.cos(e) * math.sin(s), math.sin(e))).normalized()
            if normal.y > 0:
                normal = -normal
            way = (way + normal * W['lift']).normalized()   # leaning out from his skin, so that it leaves it where it is rooted and not further along
            start = on - way * W['sink']
            across = way.cross(Vector((0.0, 0.0, 1.0))).normalized()
            up = across.cross(way).normalized()
            rows = []
            for k in range(along + 1):
                t = k / along
                p = start + way * (length + W['sink']) * t + Vector((0.0, 0.6 * W['droop'] * t * t, -W['droop'] * t * t))
                r = W['thick'][0] + (W['thick'][1] - W['thick'][0]) * t
                rows.append([bm.verts.new(p + (across * math.cos(2 * math.pi * j / around) + up * math.sin(2 * math.pi * j / around)) * r) for j in range(around)])
            for k in range(along):
                for j in range(around):
                    bm.faces.new((rows[k][j], rows[k][(j + 1) % around], rows[k + 1][(j + 1) % around], rows[k + 1][j])).smooth = True
            bm.faces.new(rows[-1])
    src.free()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mat = bpy.data.materials.get('JimmyWhiskers') or bpy.data.materials.new('JimmyWhiskers')
    mat.use_nodes = True
    bsdf = next(node for node in mat.node_tree.nodes if node.type == 'BSDF_PRINCIPLED')
    bsdf.inputs['Base Color'].default_value = W['colour']
    bsdf.inputs['Roughness'].default_value = 0.6
    mesh.materials.append(mat)
    thing = bpy.data.objects.new(name, mesh)
    bpy.data.collections[NAME].objects.link(thing)
    thing.parent, thing.parent_type, thing.parent_bone = a, 'BONE', 'head'
    bpy.context.view_layer.update()
    thing.matrix_world = ob.matrix_world.copy()
    return {'points': len(mesh.vertices), 'pads': {k: [round(float(x), 3) for x in v] for k, v in pads.items()}}


def party_hat():
    """Make his party hat: a striped cone with a bobble, sitting on his head a little to one side.

    It is a thing of its own, held by his head bone, and it is there all the time at no size: the number HAT on his
    bones (0 to 1) is its size, so a pose shows it by setting that number (write()). Its stripes are his collar's
    red and blue and its bobble his tag's gold."""
    import bmesh
    a, ob, me = arm(), skin(), skin().data
    name = NAME + '-hat'
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    n = len(me.vertices)
    co = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3).astype('f8')
    head = ob.vertex_groups['head'].index
    held_ = np.array([sum(g.weight for g in v.groups if g.group == head) for v in me.vertices])
    mine = held_ > 0.5
    mid = float(np.median(co[mine, 0]))
    high = float(co[mine, 2].max())
    low = float(np.percentile(co[mine, 2], 5))
    size = high - low                                      # his head's height, to size the hat by
    at = mid + 0.08 * size                                 # a little toward his left ear
    strip = mine & (np.abs(co[:, 0] - at) < 0.03 * size)
    top = co[strip][np.argmax(co[strip, 2])]
    radius, height = 0.20 * size, 0.56 * size
    bm = bmesh.new()
    rings, around = 8, 32
    rows = []
    for r in range(rings + 1):
        k = r / rings
        rows.append([bm.verts.new((radius * (1 - k) * math.cos(2 * math.pi * j / around), radius * (1 - k) * math.sin(2 * math.pi * j / around), height * k))
                     for j in range(around)] if r < rings else [bm.verts.new((0.0, 0.0, height))])
    for r in range(rings):
        for j in range(around):
            a0, a1 = rows[r][j], rows[r][(j + 1) % around]
            if r < rings - 1:
                f = bm.faces.new((a0, a1, rows[r + 1][(j + 1) % around], rows[r + 1][j]))
            else:
                f = bm.faces.new((a0, a1, rows[r + 1][0]))
            f.material_index = ((j + 2 * r) // 4) % 2      # stripes that wind round it
            f.smooth = True
    bobble = bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=0.16 * radius * 2.2)
    for v in bobble['verts']:
        v.co.z += height
        for f in v.link_faces:
            f.material_index = 2
            f.smooth = True
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for label, colour in (('red', (0.62, 0.05, 0.04, 1.0)), ('blue', (0.05, 0.16, 0.6, 1.0)), ('gold', (0.9, 0.55, 0.08, 1.0))):
        mat = bpy.data.materials.get('JimmyHat-' + label) or bpy.data.materials.new('JimmyHat-' + label)
        mat.use_nodes = True
        bsdf = next(node for node in mat.node_tree.nodes if node.type == 'BSDF_PRINCIPLED')
        bsdf.inputs['Base Color'].default_value = colour
        bsdf.inputs['Roughness'].default_value = 0.7
        mesh.materials.append(mat)
    hat = bpy.data.objects.new(name, mesh)
    bpy.data.collections[NAME].objects.link(hat)
    hat.parent, hat.parent_type, hat.parent_bone = a, 'BONE', 'head'
    bpy.context.view_layer.update()
    lean = Matrix.Rotation(math.radians(14.0), 4, 'Y') @ Matrix.Rotation(math.radians(6.0), 4, 'X')    # out toward his ear, and back a touch
    hat.matrix_world = ob.matrix_world @ Matrix.Translation(Vector(top) - Vector((0.0, 0.0, 0.05 * size))) @ lean
    a[HAT] = 0.0
    a.id_properties_ui(HAT).update(min=0.0, max=1.0)
    for i in range(3):
        hat.driver_remove('scale', i)
        drv = hat.driver_add('scale', i).driver
        drv.type = 'SCRIPTED'
        var = drv.variables.new()
        var.name, var.type = 'h', 'SINGLE_PROP'
        var.targets[0].id = a
        var.targets[0].data_path = '["%s"]' % HAT
        drv.expression = 'max(h, 0.0001)'
    return {'top': [round(float(v), 3) for v in top], 'head': round(size, 3)}


def saucer():
    """Make his flying saucer: a small disc with a well he sits in, a lip, a stripe, lights round its edge and a
    glow underneath.

    Like his party hat it is a thing of its own, there all the time at no size: the number ABOARD on his bones is
    its size. It is fitted to him as he sits (how far out his paws and tail reach, and the floor they are on) and
    held by his root bone, so whatever lifts or tips him takes it too. Returns its sizes."""
    import bmesh
    a, ob = arm(), skin()
    name = NAME + '-saucer'
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    rest()
    sit()
    bpy.context.view_layer.update()
    n = len(ob.data.vertices)
    made = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    co = np.zeros(n * 3, 'f4')
    made.vertices.foreach_get('co', co)
    W = np.array(ob.matrix_world)
    co = co.reshape(-1, 3).astype('f8') @ W[:3, :3].T + W[:3, 3]
    floor = float(np.percentile(co[:, 2], 0.2))
    low = co[co[:, 2] < floor + 0.5]
    cx, cy = float((low[:, 0].min() + low[:, 0].max()) / 2), float((low[:, 1].min() + low[:, 1].max()) / 2)
    reach = float(np.linalg.norm(low[:, :2] - np.array([cx, cy]), axis=1).max())
    Rc, Ro, rim = reach + SAUCER['room'], SAUCER['out'], SAUCER['rim']
    slope = lambda r: (rim - 0.05) + (0.07 - (rim - 0.05)) * (r - (Rc + 0.15)) / (Ro - (Rc + 0.15))   # the height of its top at r from the middle
    parts = [  # each a line turned round the middle: (what it is painted, [(how far out, how high), ...])
        ('hull', [(Rc - 0.02, rim - 0.02), (Rc + 0.035, rim + 0.045), (Rc + 0.10, rim + 0.02), (Rc + 0.15, rim - 0.05), (Ro, 0.07), (Ro + 0.04, 0.0),
                  (Ro, -0.07), (Ro - 0.22, -0.19), (Rc - 0.05, -0.30), (0.42, -0.36)]),
        ('well', [(0.0, -0.02), (Rc - 0.06, -0.02), (Rc - 0.02, rim - 0.02)]),
        ('trim', [(Rc - 0.025, rim - 0.01), (Rc + 0.035, rim + 0.055), (Rc + 0.105, rim + 0.025)]),
        ('stripe', [(Rc + 0.17, slope(Rc + 0.17) + 0.006), (Rc + 0.24, slope(Rc + 0.24) + 0.006)]),
        ('trim', [(Ro - 0.02, 0.07), (Ro + 0.055, 0.0), (Ro - 0.02, -0.07)]),
        ('under', [(0.43, -0.355), (0.30, -0.43), (0.0, -0.47)]),
    ]
    paints = {'hull': (SAUCER['hull'], 0.35, 0.32, 0.0), 'well': (SAUCER['well'], 0.0, 0.7, 0.35), 'trim': (SAUCER['trim'], 0.4, 0.35, 0.0),
              'stripe': (SAUCER['stripe'], 0.2, 0.35, 0.25), 'under': (SAUCER['under'], 0.0, 0.4, 2.2),
              'light0': (SAUCER['lights'][0], 0.0, 0.4, 1.8), 'light1': (SAUCER['lights'][1], 0.0, 0.4, 1.8)}
    order = list(paints)
    bm = bmesh.new()
    for paint, line in parts:
        before = set(bm.faces)
        vs = [bm.verts.new((r, 0.0, z)) for r, z in line]
        es = [bm.edges.new((p, q)) for p, q in zip(vs, vs[1:])]
        bmesh.ops.spin(bm, geom=vs + es, cent=(0, 0, 0), axis=(0, 0, 1), angle=2 * math.pi, steps=64, use_merge=True)
        for f in set(bm.faces) - before:
            f.material_index = order.index(paint)
    count = 8
    for k in range(count):                                 # its lights, round the edge, two colours turn about
        t, r = 2 * math.pi * (k + 0.5) / count, Ro - 0.06
        before = set(bm.faces)
        bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=0.05,
                                  matrix=Matrix.Translation((r * math.cos(t), r * math.sin(t), slope(r) + 0.01)) @ Matrix.Diagonal((1.0, 1.0, 0.6, 1.0)))
        for f in set(bm.faces) - before:
            f.material_index = order.index('light%d' % (k % 2))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    for f in bm.faces:
        f.smooth = True
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for label in order:
        colour, metal, rough, lit_ = paints[label]
        mat = bpy.data.materials.get('JimmySaucer-' + label) or bpy.data.materials.new('JimmySaucer-' + label)
        mat.use_nodes = True
        bsdf = next(node for node in mat.node_tree.nodes if node.type == 'BSDF_PRINCIPLED')
        bsdf.inputs['Base Color'].default_value = (*colour, 1.0)
        bsdf.inputs['Metallic'].default_value = metal
        bsdf.inputs['Roughness'].default_value = rough
        bsdf.inputs['Emission Color'].default_value = (*colour, 1.0)
        bsdf.inputs['Emission Strength'].default_value = lit_
        mesh.materials.append(mat)
    craft = bpy.data.objects.new(name, mesh)
    bpy.data.collections[NAME].objects.link(craft)
    craft.parent, craft.parent_type, craft.parent_bone = a, 'BONE', 'root'
    bpy.context.view_layer.update()
    craft.matrix_world = Matrix.Translation((cx, cy, floor))   # where he sits now; his root bone holds it from here on
    a[ABOARD] = 0.0
    a.id_properties_ui(ABOARD).update(min=0.0, max=1.0)
    for i in range(3):
        craft.driver_remove('scale', i)
        drv = craft.driver_add('scale', i).driver
        drv.type = 'SCRIPTED'
        var = drv.variables.new()
        var.name, var.type = 's', 'SINGLE_PROP'
        var.targets[0].id = a
        var.targets[0].data_path = '["%s"]' % ABOARD
        drv.expression = 'max(s, 0.0001)'
    rest()
    return {'middle': (round(cx, 3), round(cy, 3)), 'floor': round(floor, 3), 'well': round(Rc, 3), 'out': Ro, 'rim': rim}


def aboard(on=True):
    """Play his other motions with him in his saucer (or, with on=False, out of it again). The 'pose-saucer' poses
    show it whatever this says."""
    RIDING[0] = bool(on)
    a = arm()
    if ABOARD in a:
        a[ABOARD] = 1.0 if on else 0.0
        a.update_tag()
    return RIDING[0]


def breathing():
    """Let him breathe: a shape ('breath') that fills his chest, on a number of the same name on his bones.

    The skin his chest and upper spine hold moves out along the way it faces, mostly sideways and forward, and up a
    touch; his shoulders a little. It is smoothed over his skin so tufts of fur move as one, and his name tag rides
    on it without changing shape."""
    a, ob, me = arm(), skin(), skin().data
    n = len(me.vertices)
    co = np.zeros(n * 3, 'f4')
    me.shape_keys.key_blocks['Basis'].data.foreach_get('co', co)
    co = co.reshape(-1, 3).astype('f8')
    no = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('normal', no)
    no = no.reshape(-1, 3).astype('f8')
    parts = {ob.vertex_groups[name].index: part for name, part in
             (('chest', 1.0), ('spine_04', 1.0), ('spine_03', 0.7)) if name in ob.vertex_groups}
    held_ = np.zeros(n)
    for v in me.vertices:
        held_[v.index] = sum(g.weight * parts[g.group] for g in v.groups if g.group in parts)
    held_ = np.clip(held_, 0.0, 1.0)
    # not round the tops of his front legs: a leg does not breathe, and where a leg has been cut free of his
    # body (free_legs()) skin that moved on one side of the cut and not the other would open it
    clear = np.ones(n)
    for side in ('L', 'R'):
        limb = chain('clavicle.' + side)
        top, bend = (np.array(a.data.bones[name].head_local) for name in limb[1:3])
        top = top + (top - bend) * 0.4
        along = np.clip((co - top) @ (bend - top) / ((bend - top) @ (bend - top)), 0.0, 1.0)
        off = np.linalg.norm(co - (top + np.outer(along, bend - top)), axis=1)
        clear *= np.array([ease(v) for v in (off - 0.15) / 0.10])
    held_ *= clear
    size = ALIVE['breath'][0]
    move = no * np.array((1.0, 1.0, 0.3))
    move = move * (size * held_)[:, None]
    move[:, 2] += 0.25 * size * held_
    one, edges = welded(co, me)
    total, count = np.zeros((n, 3)), np.zeros(n)
    np.add.at(total, one, move)
    np.add.at(count, one, 1.0)
    move = total / np.maximum(count, 1.0)[:, None]
    for _ in range(6):
        total, count = np.zeros((n, 3)), np.zeros(n)
        np.add.at(total, edges[:, 0], move[edges[:, 1]])
        np.add.at(total, edges[:, 1], move[edges[:, 0]])
        np.add.at(count, edges[:, 0], 1.0)
        np.add.at(count, edges[:, 1], 1.0)
        move = np.where(count[:, None] > 0, 0.5 * move + 0.5 * total / np.maximum(count, 1.0)[:, None], move)
    move = move[one] * clear[:, None]
    paint = paint_at_points().astype('f8')
    hi, lo = paint.max(1), paint.min(1)
    d = np.maximum(hi - lo, 1e-6)
    r, g, b = paint[:, 0], paint[:, 1], paint[:, 2]
    hue = np.where(hi == r, ((g - b) / d) % 6, np.where(hi == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    sat = (hi - lo) / np.maximum(hi, 1e-6)
    sure = (hue > 0.07) & (hue < 0.16) & (sat > 0.4) & (hi > 0.55) & (co[:, 2] < TAG)
    if sure.sum() > 20:                                    # his tag, and the fur it lies on, move as one piece
        off = np.linalg.norm(co - np.median(co[sure], axis=0), axis=1)
        gold = (hue > 0.03) & (hue < 0.20) & (sat > 0.25) & (hi > 0.15) & (off < 0.13)
        ring = ~gold & (off > 0.10) & (off < 0.17) & (no[:, 1] < -0.3)
        if ring.any():
            rigid = (1.0 - np.array([ease(v) for v in (off - 0.14) / 0.09])) * np.clip(held_ * 3.0, 0.0, 1.0)   # only skin his chest holds: not his chin above it
            move = move * (1.0 - rigid)[:, None] + move[ring].mean(0) * rigid[:, None]
    old = me.shape_keys.key_blocks.get('breath')
    if old:
        old.driver_remove('value')
        ob.shape_key_remove(old)
    key = ob.shape_key_add(name='breath', from_mix=False)
    key.data.foreach_set('co', (co + move).astype('f4').ravel())
    if BREATH not in a:
        a[BREATH] = 0.0
    drv = key.driver_add('value').driver
    drv.type = 'SUM'
    var = drv.variables.new()
    var.type = 'SINGLE_PROP'
    var.targets[0].id = a
    var.targets[0].data_path = '["%s"]' % BREATH
    return {'points': int((held_ > 0.05).sum()), 'most': round(float(np.linalg.norm(move, axis=1).max()), 4)}


def flat_ears(amount=1.0):
    """Airplane ears: both ears out sideways and a little back, like wings. A cat does it when it is cross or wary.

    `amount` (0 to 1) is how far from where they are now."""
    start = snap()
    for side, mirror in (('L', 1.0), ('R', -1.0)):
        bones = ['ear.%s_01' % side]
        while arm().data.bones[bones[-1]].children:
            bones.append(arm().data.bones[bones[-1]].children[0].name)
        for bone, child, way in zip(bones, bones[1:], FLAT):
            aim(bone, child, (way[0] * mirror, way[1], way[2]))
    if amount < 1.0:
        put(blend(start, snap(), amount))


def lick_paw(amount=1.0):
    """Lift his left front paw to in front of his mouth, by way of the tuck (amount 0.5), to lick it (LICK).

    His paw turns over first, on the way to the tuck, so that it comes up to his mouth already turned and its pads
    never face us (Ant, 25 Nov 13:02: "turn the paw over first")."""
    start = snap()
    raise_paw(0.0, 0.5)
    tuck = snap()
    limb = chain('clavicle.L')
    over = LICK[3] if LICK and len(LICK) > 3 else 0.0

    def turned_over():                                     # shared down his forearm, wrist and paw so none is wrung
        for bone, share in zip(limb[2:5], (0.3, 0.4, 0.3)):
            twist(bone, over * share)
        return snap()
    if LICK_TUCK:                                          # folded down at the wrist
        aim(limb[3], limb[4], LICK_TUCK[0])
        aim(limb[4], limb[5], LICK_TUCK[1])
        tuck = snap()
    if over:
        tuck = turned_over()
    if amount <= 0.5 or not LICK:
        put(blend(start, tuck, min(amount, 0.5) / 0.5))
        return
    bones = skeleton()
    put(start)
    raise_paw(0.0, 0.5)
    pose = snap()
    reach(pose, bones, limb[1], limb[2], limb[3], Vector(LICK[0]))
    put(pose)
    aim(limb[3], limb[4], LICK[1])
    aim(limb[4], limb[5], LICK[2] if len(LICK) > 2 else LICK[1])   # his paw, bent back toward his mouth
    if over:
        turned_over()
    put(blend(tuck, snap(), (amount - 0.5) / 0.5))


def raise_paw(swing=0.0, amount=1.0, tip=None, to=None, side='L'):
    """Lift his left front paw beside his head, from the shoulder: shoulder out, elbow out, forearm up, pads to the front.

    `swing` (-1 to 1) tips the forearm from side to side, for waving; `tip` does the same for his wrist and paw, so
    that they can trail a little behind the forearm. `amount` (0 to 1) is how far up the paw is. On the way it
    passes through TUCK (paw lifted in front of his chest, elbow bent), as a cat's does: going straight from the
    floor to the wave would sweep the whole leg out sideways. `to` is somewhere else to lift it to, in place of the
    wave: where each part of his leg points, and how far his paw is turned (as WAVE and PALM). `side='R'` lifts
    his right paw instead, as his left's mirror image."""
    tip = swing if tip is None else tip
    WAVE, PALM = to or (globals()['WAVE'], globals()['PALM'])
    TUCK, way_ = globals()['TUCK'], (1.0 if side == 'L' else -1.0)
    if side != 'L':
        across = lambda ways: tuple((-w[0], w[1], w[2]) if isinstance(w, tuple) else w for w in ways)
        WAVE, TUCK, PALM = across(WAVE), (across(TUCK) if TUCK else TUCK), -PALM
    limb, start = chain('clavicle.' + side), snap()

    def point(ways, swing_, tip_, palm):
        upper, lower, wrist, paw = ways[-4:]
        if len(ways) == 5:
            aim(limb[0], limb[1], ways[0])
        aim(limb[1], limb[2], upper)
        rest_ = limb[2:]
        if isinstance(paw, (int, float)):
            # His toes are not aimed. The joint at the end of his paw sits a hair behind the one before it, so
            # aiming that last short bone forward turned his toes right over: toe beans under the big pad,
            # upside down. His toes keep the set they have on the floor, curled by `paw` degrees.
            dirs = [lower] * (len(rest_) - 3) + [wrist]
            sway = [0.30 * swing_] * (len(rest_) - 3) + [FLOP * tip_]
        else:
            dirs = [lower] * (len(rest_) - 3) + [wrist, paw]
            sway = [0.30 * swing_] * (len(rest_) - 3) + [0.45 * tip_, 0.50 * tip_]
        for bone, child, way, k in zip(rest_, rest_[1:], dirs, sway):
            aim(bone, child, (way[0] + k * way_, way[1], way[2]))
        if isinstance(paw, (int, float)):
            for bone in rest_[-2:]:
                arm().pose.bones[bone].matrix_basis = start[bone].copy()
            toes = arm().pose.bones[rest_[-2]]
            toes.matrix_basis = toes.matrix_basis @ Matrix.Rotation(math.radians(paw), 4, 'X')
            bpy.context.view_layer.update()
        if palm:                                           # shared between two joints, so his wrist is not wrung
            twist(limb[2], palm / 2)
            twist(limb[3], palm / 2)
        return snap()

    if not TUCK:
        up = point(WAVE, swing, tip, PALM)
        if amount < 1.0:
            put(blend(start, up, amount))
        return
    if amount >= 1.0:
        point(WAVE, swing, tip, PALM)
        return
    tuck = point(TUCK, 0.0, 0.0, 0.0)
    put(start)
    if amount <= 0.5:
        put(blend(start, tuck, amount / 0.5))
    else:
        put(blend(tuck, point(WAVE, swing, tip, PALM), (amount - 0.5) / 0.5))


def tail_bones():
    """His tail's bones, from his rump to the tip."""
    bone, out = arm().data.bones['tail_01'], []
    while bone:
        out.append(bone.name)
        bone = bone.children[0] if bone.children else None
    return out


TALKING = (1, 1, 0.75, 0.45, 0.3, 0.55, 0.85, 0.6, 0.35, 0.5, 0.9, 1, 0.7, 0.4, 0.25, 0.45, 0.8, 1, 1, 0.8, 0.5, 0.35, 0.6, 0.9, 1, 1)


def moves():
    """Write his own motions, each starting and ending in his sitting pose (the tail swish loops instead).

    30 frames a second, like Meshy's. His paw leaves the ground only to wave or to point. Returns each motion's length in frames."""
    sit()
    base, face = snap(), facing()

    def waving(swing, tip=None, amount=1.0):
        put(base)
        raise_paw(swing, amount, tip)
        nudge('head', y=5.0 * amount, face=face)
        return snap()

    def looking(turn_, tilt=0.0, nod=0.0):
        put(base)
        nudge(neck(), z=0.35 * turn_, face=face)
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
        bones = tail_bones()
        if TAIL:                                           # laid round him: a slow wave of lift down its far half
            curl_tail([size * (0.5 + 1.6 * k) * math.sin(phase - 0.7 * k) if k > 1 else 0.0 for k in range(len(bones) - 2)])
            return snap()
        for k, bone in enumerate(bones):
            k = k * 6.0 / len(bones)
            nudge(bone, z=size * (8 + 3.5 * k) * math.sin(phase - 0.7 * k) * 6.0 / len(bones), face=face)
        return snap()

    def talking(shut):
        put(base)
        if JAW:
            turn('jaw_01', x=SHUT * shut)
        return snap()

    def to(a, b, frames):
        return [blend(a, b, ease((i + 1) / frames)) for i in range(frames)]

    made = {}
    # wave: paw up beside his head, three swings, paw down again
    # his wrist and paw trail three frames behind his forearm, and it comes up and goes down by way of the tuck
    corner = {(s, t): waving(s, t) for s in (-0.5, 1.0) for t in (-0.5, 1.0)}
    tuck = waving(0.0, 0.0, 0.5)

    def swung(i):
        s, t = (0.5 + 0.5 * math.sin(2 * math.pi * k / 14) for k in (i, i - 3))
        return blend(blend(corner[-0.5, -0.5], corner[1.0, -0.5], s), blend(corner[-0.5, 1.0], corner[1.0, 1.0], s), t)

    def by_tuck(up, frames, rising):
        out = []
        for i in range(frames):
            k = ease((i + 1) / frames)
            k = k if rising else 1.0 - k
            out.append(blend(base, tuck, k / 0.5) if k <= 0.5 else blend(tuck, up, (k - 0.5) / 0.5))
        return out

    swings = [swung(i + 1) for i in range(42)]
    made[OWN + 'wave'] = [base] * 4 + by_tuck(swung(0), 12, True) + swings + by_tuck(swings[-1], 13, False) + [base] * 4
    if POINT:
        # point: showing you something. His eyes go to it, his head follows, and his paw comes up by way of the tuck
        # and is held out toward it. He gives it a small push ("that"), looks back at you with his paw still out,
        # gives it another ("that one"), and puts his paw down. His paw and his head are worked out apart and put
        # together frame by frame, so that neither waits for the other.
        # 'point' is to the right of the picture (his left paw); 'point-left' is its mirror, with his right paw,
        # named as the flat rig names its sides (tilt and tilt-left).
        count = 84
        paw_at = [(0, 0), (0.06, 0), (0.22, 1), (0.30, 1), (0.36, 0.86), (0.43, 1), (0.60, 1), (0.66, 0.86), (0.73, 1), (0.80, 1), (0.95, 0), (1, 0)]
        head_at = [(0, 0), (0.04, 0), (0.20, 1), (0.46, 1), (0.58, 0.14), (0.80, 0.14), (0.94, 0), (1, 0)]
        for name, side, way_ in (('point', 'L', 1.0), ('point-left', 'R', -1.0)):
            def held_out(up):
                put(base)
                raise_paw(0.0, up, to=(POINT, POINT_PALM), side=side)
                return snap()
            limb = set(chain('clavicle.' + side))
            tucked, out_, toward = held_out(0.5), held_out(1.0), looking(22 * way_, 3 * way_)

            def pointing(up, turn_):
                paw = blend(base, tucked, up / 0.5) if up <= 0.5 else blend(tucked, out_, (up - 0.5) / 0.5)
                head = blend(base, toward, turn_)
                return {bone: (paw if bone in limb else head)[bone] for bone in base}
            made[OWN + name] = [pointing(keyed(paw_at, i / (count - 1)), keyed(head_at, i / (count - 1))) for i in range(count)]
            made[OWN + 'pose-' + name] = [pointing(1.0, 0.82)] * (HOLD if ALIVE else 2)
    if SAUCER:
        # in his saucer: two poses to hold, each a loop. Hovering: a slow bob and a lazy tip from side to side.
        # Travelling: tipped forward, nose down, with a quicker bob. His root bone is moved, so the saucer goes too.
        def floating(up, nod=0.0, tip=0.0):
            put(base)
            lift(up)
            nudge('root', x=nod, y=tip, face=face)
            return snap()
        turn_ = lambda i, k=1, at=0.0: math.sin(2 * math.pi * k * i / HOLD + at)
        made[OWN + 'pose-saucer'] = [floating(0.035 * turn_(i), 0.8 * turn_(i, 1, 1.2), 1.6 * turn_(i, 1, 2.6)) for i in range(HOLD)] if ALIVE else [base, base]
        made[OWN + 'pose-saucer-travel'] = [floating(0.025 * turn_(i, 2), 8.0 + 1.0 * turn_(i, 2, 1.0), 1.2 * turn_(i, 1, 2.0)) for i in range(HOLD)] if ALIVE else [base, base]
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
    made[OWN + 'tail-swish'] = [tail(2 * math.pi * i / 48, 1.5) for i in range(48)]
    # talk: jaw only
    made[OWN + 'talk'] = [talking(v) for v in TALKING]
    # sit: just sitting, for holding still
    made[OWN + 'sit'] = [base, base]
    looks = {}
    if ALIVE:
        # idle: what he does when nothing else is playing, round and round. His head drifts, one ear flicks, the tip
        # of his tail lifts and settles, and he blinks: once, and later twice. alive() adds his breathing.
        def idling(i, count):
            put(base)
            t = 2 * math.pi * i / count
            turn_, tilt, nod = 2.5 * math.sin(t), 1.2 * math.sin(2 * t + 1.0), 0.8 * math.sin(t + 2.0)
            nudge(neck(), z=0.35 * turn_, face=face)
            nudge('head', z=0.65 * turn_, y=tilt, x=nod, face=face)
            flick = math.sin(math.pi * min(max((i - 0.36 * count) / 7.0, 0.0), 1.0)) * 16.0
            if flick:
                nudge('ear.R_01', y=-flick * 0.5, face=face)
                nudge('ear.R_02', y=-flick * 0.5, face=face)
            tip = 9.0 * math.sin(math.pi * min(max((i - 0.52 * count) / (0.22 * count), 0.0), 1.0)) ** 2
            if TAIL and tip:
                curl_tail([0, 0, 0, 0, 0.3 * tip, 0.7 * tip, tip])
            return snap()
        count = 192
        made[OWN + 'idle'] = [idling(i, count) for i in range(count)]
        shut_at = [0.0] * count
        for mark in (44, 128, 138):
            for j, v in enumerate(BLINKS):
                shut_at[mark - 2 + j] = max(shut_at[mark - 2 + j], v)
        looks[OWN + 'idle'] = shut_at
    # blink: lids down and up, eyes shut for two frames
    shut = (0, 0) + BLINKS + (0, 0)
    made[OWN + 'blink'] = [base] * len(shut)
    looks[OWN + 'blink'] = shut
    if FLAT:
        # annoyed: airplane ears, narrowed eyes, head down a touch, and the end of his tail thumping
        def cross(thump=0.0):
            put(base)
            nudge(neck(), x=3.0, face=face)
            nudge('head', x=5.0, y=-3.0, face=face)
            flat_ears()
            if TAIL and thump:
                curl_tail([0, 0, 0, 0.3 * thump, 0.7 * thump, thump, thump])
            return snap()
        held = cross()
        thumps = [cross(11.0 * math.sin(2 * math.pi * i / 16) - 3.0) for i in range(48)]   # three thumps, mostly up off the floor
        made[OWN + 'annoyed'] = [base] * 3 + to(base, held, 7) + to(held, thumps[0], 3) + thumps + to(thumps[-1], held, 3) + [held] * 4 + to(held, base, 12) + [base] * 3
        made[OWN + 'pose-annoyed'] = [held] * (HOLD if ALIVE else 2)
        glare = (0.46, 0.0, 0.0, 0.0, 0.0, 1.0, -0.45, -0.45, 1.0, 1.0)    # lids nearly half down and slanted toward his nose; brows down, inner ends lowest
        went = [ease(k / 7) for k in range(1, 8)]
        back = [1.0 - ease(k / 12) for k in range(1, 13)]
        how = [0.0] * 3 + went + [1.0] * (3 + 48 + 3 + 4) + back + [0.0] * 3
        looks[OWN + 'annoyed'] = [tuple(v * k for v in glare) for k in how]
        looks[OWN + 'pose-annoyed'] = [glare] * (HOLD if ALIVE else 2)
    for name, poses in made.items():
        lids = looks.get(name)
        if name == OWN + 'talk' and not JAW:
            lids = [(0.0, 0.0, 0.0, 1.0 - v) for v in TALKING]
        short = name[len(OWN):]
        poses, lids = alive(short, poses, with_brows(short, lids, len(poses)), base, face, short in LOOPS or short.startswith('pose-'))
        write(name, poses, lids)
    if ALIVE and TAIL and not JAW:
        # sit down and stand up: there was no move between his sitting pose and Meshy's standing ones. Both are cut
        # from Meshy's petted motion, which sits him down and stands him up again, with their ends eased in to
        # Meshy's standing idle at one end and his own sitting pose at the other. (alive() is not for these: it
        # keeps his front paws where they are in his sitting pose.)
        def taken(action, frames):
            play(action.name)
            out = []
            for f in frames:
                bpy.context.scene.frame_set(int(f))
                out.append(snap())
            return out
        pet, idle = bpy.data.actions[motion('dog_beifumo')], bpy.data.actions[motion('cat_idle')]
        stood = taken(idle, [idle.frame_range[0]])[0]
        down, up = taken(pet, range(26, 61)), taken(pet, range(162, 195))
        n = len(down)
        down = [blend(stood, p, ease((i + 1) / 7.0)) if i < 6 else p for i, p in enumerate(down)]
        down = [blend(p, base, ease((i - (n - 12)) / 11.0)) if i > n - 12 else p for i, p in enumerate(down)] + [base] * 3
        n = len(up)
        up = [blend(base, p, ease((i + 1) / 11.0)) if i < 10 else p for i, p in enumerate(up)]
        up = [base] * 3 + [blend(p, stood, ease((i - (n - 8)) / 7.0)) if i > n - 8 else p for i, p in enumerate(up)]
        for name, poses in ((OWN + 'sit-down', down), (OWN + 'stand-up', up)):
            shut = [0.0] * len(poses)
            for j, v in enumerate(BLINKS):
                shut[len(poses) // 2 - 2 + j] = v
            write(name, poses, shut)
            made[name] = poses
    wake('moves')
    rest()
    return {name: len(poses) for name, poses in made.items()}


def flat_moves(path=None):
    """Carry every move and pose of the flat rig over to him, by rule, from moves.json.

    moves.json is dumped from jimmy-rig.js (see ../dump-moves.cjs), so a move changed there is changed here by
    dumping and building again. The flat rig turns and slides flat parts; each of those is given its nearest
    meaning on a body:
      head r, dx, dy     his head tilts, turns and nods (his neck takes a third)
      ear r, sy          an ear splays out, or perks in
      tail, -mid, -tip   his tail's bones turn, two to each
      lid dyh, closed o  his eyelids, each eye on its own; 'closed' is the happy arch, 'lid' the sleepy dip
      brow dyh, r        his brows lift, drop and tilt (if he has brows that move: BROW)
      mouth-talk o       his jaw
      body dy, sy        all of him lifts or settles
      paw r              his paw comes up beside his head and swings (wave, and the ready pose)
    Left in the flat rig means on the left of the picture, which is his right.
    Not carried: pupils (his eyes are painted), the party hat, the tag's swing and the
    speech bubbles. So 'thinking' and 'chatting', which are only bubbles there, get a small move of their own here.
    Each move is written as OWN + its name, each pose as OWN + 'pose-' + its name. Returns their lengths in frames."""
    import json
    if path is None:                                       # beside this file, or one folder up (as in the repo)
        path = next(p for p in (os.path.join(HERE, 'moves.json'), os.path.join(HERE, '..', 'moves.json')) if os.path.exists(p))
    with open(path) as f:
        data = json.load(f)
    sit()
    base, face = snap(), facing()

    def at(track, t):
        """A track's values at time t (0 to 1), eased between its keys."""
        if t <= track[0][0]:
            return dict(track[0][1])
        for (t0, v0), (t1, v1) in zip(track, track[1:]):
            if t <= t1:
                k = ease((t - t0) / max(t1 - t0, 1e-6))
                return {key: v0.get(key, 0.0) + (v1.get(key, 0.0) - v0.get(key, 0.0)) * k for key in set(v0) | set(v1)
                        if isinstance(v0.get(key, 0.0), (int, float)) and isinstance(v1.get(key, 0.0), (int, float))}
        return dict(track[-1][1])

    def posed(vals, paw=None):
        """His pose, and how shut his eyes are, for one moment of the flat rig."""
        def get(part, prop):
            value = vals.get(part, {}).get(prop, 0.0)
            return value if isinstance(value, (int, float)) else 0.0
        put(base)
        up = -get('body', 'dy') * 0.012 + get('body', 'sy') * 0.6
        if up:
            lift(up)
            if FEET and -0.1 < up < 0:                     # a settle: he sinks on to his haunches, not through the floor
                plant(base)
        lean = get('body', 'r')                            # his chest leans to one side (degrees, the same way as his head tilts)
        if lean:
            upper = [bone for bone in ('spine2', 'spine_03', 'spine_04', 'chest') if bone in base]
            for bone in upper:
                nudge(bone, y=lean / len(upper), face=face)
        pitch = get('body', 'p')                           # his chest pitches forward and down (degrees)
        if pitch:
            upper = [bone for bone in ('spine2', 'spine_03', 'spine_04', 'chest') if bone in base]
            for bone in upper:
                nudge(bone, x=pitch / len(upper), face=face)
        hang = get('body', 'hang')                         # picked up: his front paws hang from the wrist and his back legs unfold and dangle
        if hang:
            for side in ('L', 'R'):
                limb = chain('clavicle.' + side)
                nudge(limb[1], x=-12.0 * hang, face=face)
                nudge(limb[3], x=40.0 * hang, face=face)
                back = chain('rearUpr.' + side)
                for bone, degrees in zip(back, (35.0, -30.0, 20.0)):
                    nudge(bone, x=degrees * hang, face=face)
        tilt, turn_, nod = get('head', 'r'), get('head', 'dx') * 6.0, get('head', 'dy') * 4.0
        if paw:
            tilt += 5.0 * paw[0]                           # his head tips toward the paw he lifts
        if tilt or turn_ or nod:
            nudge(neck(), y=tilt * 0.35, z=turn_ * 0.35, x=nod * 0.35, face=face)
            nudge('head', y=tilt * 0.65, z=turn_ * 0.65, x=nod * 0.65, face=face)
        for flat, mine, sign in (('left', 'R', -1.0), ('right', 'L', 1.0)):
            out = get('ear-' + flat, 'r') - sign * get('ear-' + flat, 'sy') * 120.0
            if out:
                nudge('ear.%s_01' % mine, y=out * 0.5, face=face)
                nudge('ear.%s_02' % mine, y=out * 0.5, face=face)
        swing_tail(get('tail', 'r') * 1.5, get('tail-mid', 'r') * 1.5, get('tail-tip', 'r') * 1.5, face)
        if get('mouth-talk', 'o') and JAW:
            turn('jaw_01', x=SHUT * (1.0 - 0.6 * get('mouth-talk', 'o')))
        pose = snap()
        if paw:                                            # (how far up, which way it has swung, its tip, and how far toward his mouth)
            raise_paw(paw[1], paw[0], paw[2])
            up_pose = snap()
            if len(paw) > 3 and paw[3] > 0:                # to lick it: held in front of his mouth instead (lick_paw())
                put(pose)
                lick_paw(paw[0])
                up_pose = blend(up_pose, snap(), min(paw[3], 1.0))
                put(up_pose)
            pose = up_pose
        right = max(get('lid-left', 'dyh') / 1.05, get('closed-left', 'o'))
        left = max(get('lid-right', 'dyh') / 1.05, get('closed-right', 'o'))
        joy = max(get('closed-left', 'o'), get('closed-right', 'o'))      # the flat rig's 'closed' eyes are its happy ones
        # brows: the flat rig lifts one by fractions of its own height (up is negative) and turns it by degrees;
        # turning the brow on the left of the picture clockwise drops its inner end
        up_r, up_l = -get('brow-left', 'dyh') * 1.4, -get('brow-right', 'dyh') * 1.4
        tilt_r, tilt_l = get('brow-left', 'r') / 16.0, -get('brow-right', 'r') / 16.0
        return pose, (0.0, left, right, 0.0 if JAW else 0.75 * get('mouth-talk', 'o'), joy, 0.0, up_l, up_r, tilt_l, tilt_r)

    made = {}
    for name, move in list(data['ACTIONS'].items()) + [(k, v) for k, v in MORE.items() if k not in data['ACTIONS']]:
        if FLAT and name in HIS_OWN:
            continue
        move = REWORK.get(name, move) if ALIVE else move
        frames = max(2, int(round(move['ms'] / 1000.0 * 30)) + 1)
        if name == 'blink' and ALIVE:                      # the flat rig's blink opens as fast as it shuts; his own does not
            frames = len(BLINKS) + 4
        # pose him at every key of every track and at twelfths between, then blend those for the frames
        times = sorted({round(k / 12.0, 4) for k in range(13)} | {round(t, 4) for track in move['tracks'].values() for t, _ in track} | {0.2, 0.8})
        keys = []
        for t in times:
            vals = {part: at(track, t) for part, track in move['tracks'].items()}
            paw = None
            if 'paw' in vals:                              # up in the first fifth, down in the last
                up_ = vals['paw']['up'] if 'up' in vals['paw'] else ease(min(t / 0.2, (1.0 - t) / 0.2, 1.0))   # 'up' holds it part-way (0.5 is the tuck)
                paw = (up_, vals['paw'].get('r', 0.0) / 14.0,
                       at(move['tracks']['paw'], max(0.0, t - 0.04)).get('r', 0.0) / 14.0,      # his paw's tip trails
                       vals['paw'].get('lick', 0.0))
            if name == 'thinking' and name not in REWORK:  # only bubbles in the flat rig
                vals = {'head': {'r': -4.0 * math.sin(math.pi * t) ** 2, 'dy': -1.0 * math.sin(math.pi * t) ** 2},
                        'tail-tip': {'r': 10.0 * math.sin(2 * math.pi * t)}}
            if name == 'chatting' and name not in REWORK:
                vals = {'mouth-talk': {'o': 0.5 - 0.5 * math.cos(2 * math.pi * 4 * t)}, 'head': {'r': 2.0 * math.sin(2 * math.pi * t)}}
            keys.append(posed(vals, paw))
        poses, lids = [], []
        for i in range(frames):
            t = i / (frames - 1)
            k = max(j for j, tj in enumerate(times) if tj <= t + 1e-9)
            k = min(k, len(times) - 2)
            part = (t - times[k]) / max(times[k + 1] - times[k], 1e-6)
            part = max(0.0, min(1.0, part))
            poses.append(blend(keys[k][0], keys[k + 1][0], part))
            lids.append(tuple(x + (y - x) * part for x, y in zip(keys[k][1], keys[k + 1][1])))
        if name == 'blink' and ALIVE:
            lids = [(look[0], v, v) + tuple(look[3:]) for v, look in zip((0, 0) + BLINKS + (0, 0), lids)]
        if name in BROW_FLIP:
            lids = [look[:8] + tuple(-v for v in look[8:10]) + look[10:] for look in lids]
        poses, lids = alive(name, poses, with_brows(name, lids, frames), base, face, name in LOOPS)
        write(OWN + name, poses, lids)
        made[OWN + name] = frames
    for name, vals in data['POSES'].items():
        if FLAT and name in HIS_OWN:
            continue
        afresh = bool(ALIVE) and name in POSE_REWORK
        vals = dict(POSE_REWORK[name] if afresh else vals)
        if name == 'sunk' and not afresh:                  # the flat rig slides him down out of his frame
            vals['body'] = {'dy': 45.0}
        pose, lid = posed(vals, (0.5, 0.0, 0.0) if 'paw' in vals else None)   # the flat rig's 'ready' is his paw lifted in front of him
        count = HOLD if ALIVE else 2
        held_, lids = alive('pose-' + name, [pose] * count, with_brows('pose-' + name, [lid] * count, count), base, face, True)
        write(OWN + 'pose-' + name, held_, lids)
        made[OWN + 'pose-' + name] = count
    wake('flat_moves')
    rest()
    return made


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


def floof():
    """Stop his chest fur coming along when a front leg lifts.

    Meshy's rig lets each upper front leg pull on the fur of his chest beside it. On the side toward the middle of
    his chest, keep that pull only on points near the leg's own bone; what is taken away goes to his chest. His
    shoulder, on the outside, still moves with the leg, so the leg hinges there."""
    from mathutils import geometry
    a, ob, me = arm(), skin(), skin().data
    chest = ob.vertex_groups['chest']
    moved = 0
    for side in ('L', 'R'):
        bone = a.data.bones['frontUpr.' + side]
        p0, p1 = a.matrix_world @ bone.head_local, a.matrix_world @ a.data.bones['frontLwr.' + side].head_local
        group = ob.vertex_groups['frontUpr.' + side]
        for v in me.vertices:
            w = next((g.weight for g in v.groups if g.group == group.index), 0.0)
            if w < 0.01:
                continue
            point, t = geometry.intersect_point_line(v.co, p0, p1)
            t = max(0.0, min(1.0, t))
            on_bone = p0 + (p1 - p0) * t
            far = (v.co - on_bone).length
            inner = (on_bone.x - v.co.x) * (1.0 if side == 'L' else -1.0)      # how far toward the middle of his chest
            keep = 1.0 - ease((far - FLOOF[0]) / (FLOOF[1] - FLOOF[0])) * ease(inner / 0.05)
            if keep > 0.999:
                continue
            group.add([v.index], w * keep, 'REPLACE')
            chest.add([v.index], w * (1.0 - keep), 'ADD')
            moved += 1
    return moved


def welded(co, me):
    """Meshy's skin is separate patches with a point each side of every join. For each point, the one point that
    stands for its place, and his skin's edges between those."""
    _, firsts, which = np.unique(np.round(co, 5), axis=0, return_index=True, return_inverse=True)
    one = firsts[which.ravel()]
    edges = np.zeros(len(me.edges) * 2, 'i4')
    me.edges.foreach_get('vertices', edges)
    edges = one[edges.reshape(-1, 2)]
    return one, edges[edges[:, 0] != edges[:, 1]]


def headset_paint(co, me, rings=2):
    """Which points are his headset or its microphone, by their paint, and `rings` rings of points round them."""
    paint = paint_at_points().astype('f8')
    hi, lo = paint.max(1), paint.min(1)
    d = np.maximum(hi - lo, 1e-6)
    r, g, b = paint[:, 0], paint[:, 1], paint[:, 2]
    hue = np.where(hi == r, ((g - b) / d) % 6, np.where(hi == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    teal = (hue > 0.40) & (hue < 0.62) & ((hi - lo) / np.maximum(hi, 1e-6) > 0.12) & (hi > 0.2)
    one, edges = welded(co, me)
    place = np.zeros(len(co), bool)
    np.logical_or.at(place, one, teal)
    for _ in range(rings):
        grown = place.copy()
        grown[edges[place[edges[:, 1]], 0]] = True
        grown[edges[place[edges[:, 0]], 1]] = True
        place = grown
    return place[one]


def shoulders():
    """Tidy how each front leg pulls on the skin round its shoulder, so a lifted paw does not bring a ragged mass of
    fur up with it.

    Meshy's rig shares the skin there out between leg and body point by point, with no pattern: the tip of a tuft
    may follow the leg and its root the chest, and the pull reaches his flank and the fur under his arm. Lift the
    leg and that comes up as spikes and webs. Two things are done to how much of each point the leg holds (all the
    leg's bones together):
      - away from the upper leg's bone it is eased off to nothing (not below the elbow: the lower leg is left
        alone, or it would tear away from the skin above it);
      - it is smoothed over his skin, so that neighbours, and the root and tip of a tuft, move alike.
    What the leg gives up goes to whatever else held the point; what it gains comes from them. Points that share
    a place are treated as one, so no seam opens."""
    from mathutils import geometry
    a, ob, me = arm(), skin(), skin().data
    n = len(me.vertices)
    co = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3).astype('f8')
    one, edges = welded(co, me)
    count = len(ob.vertex_groups)
    chest = ob.vertex_groups['chest'].index
    whole, ends, reach, rounds = SHOULDER
    changed = 0
    for side in ('L', 'R'):
        limb = [b for b in chain('clavicle.' + side) if b in ob.vertex_groups]
        cols = [ob.vertex_groups[b].index for b in limb]
        joint, elbow = a.data.bones[limb[1]].head_local, a.data.bones[limb[2]].head_local
        away = np.linalg.norm(co - np.array(joint), axis=1)
        mine = np.where(away < reach)[0]
        W = np.zeros((len(mine), count))
        for row, i in enumerate(mine):
            for g in me.vertices[i].groups:
                W[row, g.group] = g.weight
        places, back = np.unique(one[mine], return_inverse=True)       # one row for each place
        P = np.zeros((len(places), count))
        np.add.at(P, back, W)
        P /= np.bincount(back)[:, None]
        pos = co[places]
        A = P[:, cols].copy()                              # what each of the leg's bones holds
        was = A.sum(1)
        far = np.array([(Vector(p) - (joint + (elbow - joint) * max(0.0, min(1.0, geometry.intersect_point_line(Vector(p), joint, elbow)[1])))).length for p in pos])
        below = np.clip(A[:, 2:].sum(1) / 0.3, 0.0, 1.0)   # how much the point belongs to the leg below the elbow
        off = np.array([ease(v) for v in (far - whole) / (ends - whole)]) * (1.0 - below)
        A[:, :2] *= (1.0 - off)[:, None]
        l0, l1 = np.searchsorted(places, edges[:, 0]), np.searchsorted(places, edges[:, 1])
        ok = (l0 < len(places)) & (l1 < len(places))
        ok[ok] &= (places[l0[ok]] == edges[ok, 0]) & (places[l1[ok]] == edges[ok, 1])
        l0, l1 = l0[ok], l1[ok]
        links = np.maximum(np.bincount(l0, minlength=len(places)) + np.bincount(l1, minlength=len(places)), 1)
        fixed = (np.linalg.norm(pos - np.array(joint), axis=1) > reach - 0.06) | (below > 0.99)
        share = A / np.maximum(A.sum(1), 1e-9)[:, None]
        total = A.sum(1)
        for _ in range(int(rounds)):
            for field in (total, share):
                summed = np.zeros_like(field)
                np.add.at(summed, l0, field[l1])
                np.add.at(summed, l1, field[l0])
                smooth = 0.5 * field + 0.5 * summed / (links if field.ndim == 1 else links[:, None])
                smooth[fixed] = field[fixed]
                field[...] = smooth
        total *= 1.0 - off * (1.0 - below)                 # smoothing must not carry the leg's pull back out again
        share /= np.maximum(share.sum(1), 1e-9)[:, None]
        new = total[:, None] * share
        others = P.copy()
        others[:, cols] = 0.0
        spare = others.sum(1)
        room = np.maximum(was + spare - total, 0.0)        # what is left for everything else
        scale = np.where(spare > 1e-6, room / np.maximum(spare, 1e-6), 0.0)
        others *= scale[:, None]
        others[spare <= 1e-6, chest] = room[spare <= 1e-6]
        others[:, cols] = new
        moved = np.abs(others - P).max(1) > 0.003
        for row, i in enumerate(mine):
            p = back[row]
            if not moved[p]:
                continue
            for g in np.where((others[p] > 1e-4) | (W[row] > 0.0))[0]:
                ob.vertex_groups[int(g)].add([int(i)], float(others[p, g]), 'REPLACE')
            changed += 1
    return changed


def sat_bones():
    """How each bone moves his skin when he sits, by the number of its group on his skin."""
    a, ob = arm(), skin()
    hold(motion(SIT[0]), SIT[1])
    if FEET:
        ground()                       # as he really sits: with his paws set on the floor
    bpy.context.view_layer.update()
    sat = {}
    for g in ob.vertex_groups:
        pb = a.pose.bones.get(g.name)
        if pb:
            sat[g.index] = np.array(pb.matrix @ pb.bone.matrix_local.inverted())
    rest()
    return sat


def carried(w, sat):
    """Where bones sharing a point so ({group: share}) carry it when he sits, as a matrix."""
    m, total = np.zeros((4, 4)), 0.0
    for g, x in w.items():
        if g in sat:
            m += sat[g] * x
            total += x
    return m / total if total > 1e-9 else np.eye(4)


SEAMS = {}                             # where free_legs() last cut, for looking at


def free_legs():
    """Cut each front leg free of his body, from the elbow up to the shoulder, so that a lifted leg is only a leg.

    Meshy sculpted him sitting, with each upper front leg sunk into his chest and flank: half a tube on one skin
    with his body. However the pull of the leg is shared out, lifting it must stretch that skin from where the leg
    is to where it was, and that is a wing of fur under his arm. So:
      - his skin is cut along both sides of the upper leg (what is within SLEEVE[0] of the leg's bone is leg), from
        below the elbow, where the leg is already a whole tube, to a little above the shoulder joint. It stays
        joined across the top of the shoulder, where a turning leg hardly moves;
      - the leg's side of the cut follows the leg alone; the body's side does not follow it at all. Above the
        joint both ease back to what they held before, and away from the joint the body holds none of the leg;
      - the open back of the leg is closed with a sleeve: a smooth half-round skin from one side of the cut to the
        other, joined to the leg along it;
      - the hole left in his body is covered by another copy, pressed nearly flat, which follows his body.
      - where the leg came up through the skin under his chest, the hole is plugged.
    At rest all of that is hidden: inside him, or under the leg. And he sits exactly as he sat: every point whose
    pull changes is moved, in the pose he was modelled in, to where it must be for that."""
    import bmesh
    from mathutils import kdtree
    a, ob, me = arm(), skin(), skin().data
    if me.shape_keys:
        raise RuntimeError('free_legs() must run before his skin has shape keys')
    radius, rise, flat, bend, depth, pad = SLEEVE
    n, nf = len(me.vertices), len(me.polygons)
    co = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3).astype('f8')
    cen = np.zeros(nf * 3, 'f4')
    me.polygons.foreach_get('center', cen)
    cen = cen.reshape(-1, 3).astype('f8')
    starts, totals = np.zeros(nf, 'i4'), np.zeros(nf, 'i4')
    me.polygons.foreach_get('loop_start', starts)
    me.polygons.foreach_get('loop_total', totals)
    corner = np.zeros(len(me.loops), 'i4')
    me.loops.foreach_get('vertex_index', corner)
    faces_down = np.zeros(nf * 3, 'f4')
    me.polygons.foreach_get('normal', faces_down)
    faces_down = faces_down.reshape(-1, 3)[:, 2] < -0.5
    place = np.unique(np.round(co, 5), axis=0, return_inverse=True)[1].ravel()
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    dl = bm.verts.layers.deform.verify()
    chest = ob.vertex_groups['chest'].index
    # He was cut as Meshy modelled him, but he is seen sitting, and there the leg and the body, no longer sharing
    # their pull, would part along the cut. So every point whose pull is changed is moved, as modelled, to where it
    # must be for him to sit exactly as he sat before.
    sat = sat_bones()
    sitting = lambda w: carried(w, sat)
    out = {}
    for side in ('L', 'R'):
        limb = [b for b in chain('clavicle.' + side) if b in ob.vertex_groups]
        cols = [ob.vertex_groups[b].index for b in limb]
        joint, elbow, wrist = (np.array(a.data.bones[b].head_local) for b in limb[1:4])
        top, low = joint[2] + rise, elbow[2] - 0.10
        was = {}                                           # each changed point: how it sat

        def across(p):                                     # how far a point is from the leg's bones, level with it
            z = p[:, 2]
            up = joint[:2] + (elbow[:2] - joint[:2]) * ((z - joint[2]) / (elbow[2] - joint[2]))[:, None]
            down = elbow[:2] + (wrist[:2] - elbow[:2]) * ((z - elbow[2]) / (wrist[2] - elbow[2]))[:, None]
            return np.linalg.norm(p[:, :2] - np.where((z >= elbow[2])[:, None], up, down), axis=1)

        r = across(cen)
        local = np.where((cen[:, 2] > low) & (cen[:, 2] < top) & (r < 0.40))[0]
        is_leg = r[local] < radius
        # the skin under his chest, between his legs, meets the leg at the elbow: that is body, right up to the leg
        is_leg &= ~(faces_down[local] & (cen[local, 2] > elbow[2] - 0.08) & (cen[local, 2] < elbow[2] + 0.05))
        edges = {}
        for row, f in enumerate(local):
            pv = place[corner[starts[f]:starts[f] + totals[f]]]
            for k in range(len(pv)):
                key = (pv[k], pv[k - 1]) if pv[k] < pv[k - 1] else (pv[k - 1], pv[k])
                edges.setdefault(key, []).append(row)
        nbr = [[] for _ in local]
        for rows in edges.values():
            for i in rows:
                for j in rows:
                    if i != j:
                        nbr[i].append(j)

        def flood(seeds, among):
            got = np.zeros(len(local), bool)
            got[seeds] = True
            todo = list(seeds)
            while todo:
                i = todo.pop()
                for j in nbr[i]:
                    if among[j] and not got[j]:
                        got[j] = True
                        todo.append(j)
            return got

        for _ in range(4):                                 # a straighter cut: no single faces sticking out of either side
            flip = [i for i in range(len(local)) if len(nbr[i]) > 1
                    and sum(1 for j in nbr[i] if is_leg[j] != is_leg[i]) * 2 > len(nbr[i])]
            if not flip:
                break
            is_leg[flip] = ~is_leg[flip]
        # a tuft of the leg that reaches past the radius is still leg; a bit of body fur inside it is still body
        body = flood(np.where(~is_leg & ((r[local] > 0.25) | (cen[local, 2] > top - 0.012)))[0], ~is_leg)
        is_leg = ~body
        leg = flood(np.where(is_leg & (cen[local, 2] < elbow[2] - 0.03) & (r[local] < radius))[0], is_leg)
        kind = np.zeros(nf, 'i1')                          # 1 leg, 2 body beside it
        kind[local[leg]] = 1
        kind[local[~leg]] = 2
        leg_faces = [bm.faces[int(f)] for f in local[leg]]
        cut, seam = set(), []
        for f in leg_faces:
            for e in f.edges:
                if any(0 <= g.index < nf and kind[g.index] == 2 for g in e.link_faces):
                    cut.add(e)
        for e in cut:
            seam.extend(tuple(np.round(v.co, 5)) for v in e.verts)
        seam_places = set(seam)
        bmesh.ops.split_edges(bm, edges=list(cut))
        leg_verts = {v for f in leg_faces for v in f.verts}
        edge_verts = [v for f in (bm.faces[int(i)] for i in local[~leg]) for v in f.verts
                      if v not in leg_verts and tuple(np.round(v.co, 5)) in seam_places]
        edge_verts = list(dict.fromkeys(edge_verts))
        upper, lower = cols[1], cols[2]
        changed = 0
        for v in bm.verts:
            w = v[dl]
            is_leg_vert = v in leg_verts
            held = sum(w.get(c, 0.0) for c in cols)
            if not is_leg_vert and held < 1e-4:
                continue
            total = sum(w.values())
            if total < 1e-6:
                continue
            held /= total
            z = v.co.z
            if not is_leg_vert and z <= low and across(np.array(v.co)[None])[0] < 0.20:
                if held > 0.3 and z > wrist[2]:            # the leg below the cut is a whole tube: all of it follows the lower leg
                    under = {c: w.get(c, 0.0) for c in cols[2:]}
                    had = sum(under.values())
                    was[v] = sitting(w)
                    w.clear()
                    if had > 1e-6:
                        for c, x in under.items():
                            if x > 1e-5:
                                w[c] = x / had
                    else:
                        w[lower] = 1.0
                    changed += 1
                continue
            t = ease((z - joint[2]) / rise)
            near = 1.0 - ease((float(np.linalg.norm(np.array(v.co) - joint)) - 0.10) / 0.12)
            want = 1.0 - t * (1.0 - held * near) if is_leg_vert else held * t * near
            if abs(want - held) < 1e-4 and abs(total - 1.0) < 1e-4 and not (is_leg_vert and t == 0.0):
                continue
            mine = {c: w.get(c, 0.0) / total for c in cols}
            rest_ = {g: x / total for g, x in w.items() if g not in cols}
            if is_leg_vert and t == 0.0:                   # below the joint the leg's own bones have it, not his collar bone
                mine[upper] += mine[cols[0]]
                mine[cols[0]] = 0.0
            if held > 1e-4:
                mine = {c: x * want / held for c, x in mine.items()}
            else:
                mine = {c: 0.0 for c in cols}
                mine[upper if z >= elbow[2] else lower] = want
            if is_leg_vert and low < z < elbow[2] + bend:   # round the elbow: from upper leg to lower leg evenly along the leg, all the way round it
                u = ease((z - (elbow[2] - bend)) / (2.0 * bend))
                under = {c: mine[c] for c in cols[2:]}
                had = sum(under.values())
                mine = {c: 0.0 for c in cols}
                mine[upper] = want * u
                if had > 1e-6:
                    for c, x in under.items():
                        mine[c] = want * (1.0 - u) * x / had
                else:
                    mine[lower] = want * (1.0 - u)
            spare, room = sum(rest_.values()), 1.0 - want
            if spare > 1e-6:
                rest_ = {g: x * room / spare for g, x in rest_.items()}
            elif room > 1e-6:
                rest_ = {chest: room}
            was[v] = sitting(w)
            w.clear()
            for g, x in list(mine.items()) + list(rest_.items()):
                if x > 1e-5:
                    w[g] = x
            changed += 1
        # His shoulder: the fur round the joint follows his collar bone, which carries the joint itself. When the
        # leg lifts, the collar bone swings the joint out, and the shoulder goes with it as a soft mass, so the leg
        # comes out of a shoulder and not out of a hole in his side.
        collar_bone, sign = cols[0], (1.0 if side == 'L' else -1.0)
        for v in bm.verts:
            if v in leg_verts or v.co.x * sign < 0.02:
                continue
            d = float(np.linalg.norm(np.array(v.co) - joint))
            if d > 0.22:
                continue
            mass = pad * (1.0 - ease((d - 0.05) / 0.17))
            w = v[dl]
            total = sum(w.values())
            if total < 1e-6:
                continue
            has = w.get(collar_bone, 0.0) / total
            if mass <= has + 1e-3:
                continue
            if v not in was:
                was[v] = sitting(w)
            others = {g: x / total for g, x in w.items() if g != collar_bone}
            spare = sum(others.values())
            w.clear()
            w[collar_bone] = mass
            for g, x in others.items():
                if spare > 1e-6 and x * (1.0 - mass) / spare > 1e-5:
                    w[g] = x * (1.0 - mass) / spare
            changed += 1
        # the plane the cut lies in: his body's surface where the leg is sunk into it
        pts = np.array([p for p in seam_places if p[2] > elbow[2] + 0.06])
        q = pts.mean(0)
        normal = np.linalg.svd(pts - q)[2][2]
        if (np.mean([cen[f.index] for f in leg_faces if cen[f.index, 2] > elbow[2] + 0.03], 0) - q) @ normal < 0:
            normal = -normal
        front = [f for f in leg_faces if (cen[f.index] - q) @ normal > -0.01]
        qv, nv = Vector(q), Vector(normal)
        on_seam = lambda v: tuple(np.round(v.co, 5)) in seam_places
        # The back of the leg: a sleeve. What shows of his upper leg is a shallow swell on his side, and a leg is
        # round, so the open back is closed with a smooth half-round skin from one side of the cut to the other,
        # row by row up the leg. Each row starts and ends on the leg's own points along the cut, so it is joined to
        # the leg. The bottom row sits on the rim of his lower leg, where the leg was already whole.
        along_cut = np.cross(normal, (0.0, 0.0, 1.0))
        along_cut /= np.linalg.norm(along_cut)
        rim = [v for v in leg_verts if on_seam(v)]
        sides = ([v for v in rim if (np.array(v.co) - q) @ along_cut < 0.0 and abs((np.array(v.co) - q) @ normal) < 0.05],
                 [v for v in rim if (np.array(v.co) - q) @ along_cut > 0.0 and abs((np.array(v.co) - q) @ normal) < 0.05])
        ring = [v for v in rim if v.co.z < elbow[2] + 0.05 and (np.array(v.co) - q) @ normal < -0.015]
        sleeve_faces = 0
        if all(len(side_) > 4 for side_ in sides):
            uv = bm.loops.layers.uv.active
            z0 = max(min(v.co.z for v in side_) for side_ in sides)
            z1 = min(max(v.co.z for v in side_) for side_ in sides)
            rows, last, fur = [], None, {}
            for z in np.arange(z0, z1 + 1e-6, 0.018):
                ends = tuple(min(side_, key=lambda v: abs(v.co.z - z)) for side_ in sides)
                if ends == last:
                    continue
                last = ends
                pa, pb = np.array(ends[0].co), np.array(ends[1].co)
                mid_, half = (pa + pb) / 2.0, (pa - pb) / 2.0
                width = float(np.linalg.norm(half))
                inward = -normal - (-normal @ half) * half / width ** 2
                inward /= np.linalg.norm(inward)
                deep = depth * width * min(1.0, max(0.0, (z1 + 0.02 - z) / 0.10)) ** 0.5   # rounding in to nothing under his shoulder
                wa, wb = dict(ends[0][dl].items()), dict(ends[1][dl].items())
                ma = was[ends[0]] if ends[0] in was else sitting(wa)
                mb = was[ends[1]] if ends[1] in was else sitting(wb)
                if uv:
                    for v in ends:
                        if v not in fur:
                            fur[v] = v.link_loops[0][uv].uv.copy()
                row = [ends[0]]
                for j in range(1, 10):
                    t = j / 10.0
                    spot = mid_ + half * math.cos(math.pi * t) + inward * deep * math.sin(math.pi * t)
                    if not rows and ring:                  # the bottom row: on the rim of his lower leg
                        near_ = min(ring, key=lambda v: (np.array(v.co) - spot) @ (np.array(v.co) - spot))
                        if (near_.co - Vector(spot)).length < 0.07:
                            spot = np.array(near_.co)
                    v = bm.verts.new(Vector(spot))
                    for g in set(wa) | set(wb):
                        x = wa.get(g, 0.0) * (1.0 - t) + wb.get(g, 0.0) * t
                        if x > 1e-5:
                            v[dl][g] = x
                    was[v] = ma * (1.0 - t) + mb * t
                    row.append(v)
                row.append(ends[1])
                rows.append((row, mid_))
            like = next(f for f in ends[0].link_faces)
            for (low_, mid_), (high, _) in zip(rows, rows[1:]):
                for j in range(10):
                    corners = [low_[j], low_[j + 1], high[j + 1], high[j]]
                    corners = [v for k, v in enumerate(corners) if v not in corners[:k]]
                    if len(corners) < 3:
                        continue
                    try:
                        f = bm.faces.new(corners)
                    except ValueError:
                        continue
                    f.normal_update()
                    centre = np.mean([np.array(v.co) for v in corners], 0)
                    if np.array(f.normal) @ (centre - mid_) < 0.0:
                        f.normal_flip()
                    f.material_index, f.smooth = like.material_index, True
                    if uv:
                        for loop in f.loops:
                            src = loop.vert if loop.vert in fur else (low_[0] if j < 5 else low_[-1])
                            loop[uv].uv = fur[src]         # all of it the fur by the cut
                    sleeve_faces += 1
        # the patch on his body: the leg's front pressed nearly flat, and below the armpit folded back in under him
        tree = kdtree.KDTree(len(edge_verts))
        for i, v in enumerate(edge_verts):
            tree.insert(v.co, i)
        tree.balance()
        pit = elbow[2] + 0.02
        patch = [f for f in front if cen[f.index, 2] >= pit]
        made = bmesh.ops.duplicate(bm, geom=patch)
        cover = made['geom']
        copies = set(g for g in cover if isinstance(g, bmesh.types.BMVert))
        for src, new in made['vert_map'].items():
            if new in copies and src not in copies:
                was[new] = was[src] if src in was else sitting(src[dl])
        for v in (g for g in cover if isinstance(g, bmesh.types.BMVert)):
            h = (v.co - qv).dot(nv)
            mix = {}
            for _, i, d in tree.find_n(v.co, 8):
                k = 1.0 / (d + 0.01)
                for g, x in edge_verts[i][dl].items():
                    mix[g] = mix.get(g, 0.0) + x * k
            total = sum(mix.values())
            w = v[dl]
            w.clear()
            for g, x in mix.items():
                if x / total > 1e-4:
                    w[g] = x / total
            if on_seam(v):
                continue                                   # the patch's edge is the edge of the hole
            v.co -= nv * ((1.0 - flat) * max(h, 0.0) + 0.004)
        # and where the leg came up through the skin under his chest, a plug: a fan from the middle of the hole
        ring = [v for v in edge_verts if v.co.z < elbow[2] + 0.05]
        plugged = 0
        if len(ring) > 5:
            uv = bm.loops.layers.uv.active
            mid = Vector((elbow[0], elbow[1], sum(v.co.z for v in ring) / len(ring)))
            ring.sort(key=lambda v: math.atan2(v.co.y - mid.y, v.co.x - mid.x))
            hub = bm.verts.new(mid)
            mix = {}
            for v in ring:
                for g, x in v[dl].items():
                    mix[g] = mix.get(g, 0.0) + x / len(ring)
            for g, x in mix.items():
                hub[dl][g] = x
            for v0, v1 in zip(ring, ring[1:] + ring[:1]):
                if v0 is v1 or (v0.co - v1.co).length > 0.08 or (v0.co - v1.co).length < 1e-6:
                    continue
                try:
                    f = bm.faces.new((hub, v0, v1))
                except ValueError:
                    continue
                f.normal_update()
                if f.normal.z > 0.0:
                    f.normal_flip()
                like = v0.link_faces[0] if v0.link_faces[0] is not f else v0.link_faces[-1]
                f.material_index, f.smooth = like.material_index, True
                if uv:
                    for loop in f.loops:
                        src = loop.vert if loop.vert is not hub else v0
                        other = next((l for l in src.link_loops if l.face is not f and l.face.is_valid and len(l.face.verts) and hub not in l.face.verts), None)
                        if other:
                            loop[uv].uv = other[uv].uv
                plugged += 1
        for v, m in was.items():                           # as modelled, where it must be to sit as before
            if v.is_valid:
                v.co = Vector((np.linalg.inv(sitting(v[dl])) @ m @ np.array([v.co.x, v.co.y, v.co.z, 1.0]))[:3])
        SEAMS[side] = dict(edge=[tuple(round(x, 4) for x in v.co) for v in edge_verts], q=[float(x) for x in q], normal=[float(x) for x in normal],
                           joint=[float(x) for x in joint], elbow=[float(x) for x in elbow])
        out[side] = dict(leg=len(leg_faces), cut=len(cut), weights=changed, sleeve=sleeve_faces, patch=len(patch), plug=plugged, edge=len(edge_verts),
                         normal=[round(float(x), 2) for x in normal])
        bm.verts.index_update()
        bm.faces.index_update()
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    me.update()
    return out


def head_fur():
    """Have the fur of his head held by his head.

    Meshy's rig gives much of his cheek fur, more on one side than the other, to the bones of his neck. So when his
    head tilts or turns, tufts at the bottom of it stay where they were. Out from the line of his neck and above his
    collar, what his neck bones hold is handed to his head. Under his chin, close to that line, his neck still bends.
    Each point so changed is moved, as modelled, to where it must be for him to sit exactly as he sat before: his
    sitting pose bends his neck, and fur handed to his head would otherwise come down over his collar."""
    a, ob, me = arm(), skin(), skin().data
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
    head_at, low = np.array(a.data.bones['head'].head_local), np.array(a.data.bones[neck_bones()[0]].head_local)
    collar = (((hue > 0.56) & (hue < 0.70) & (sat > 0.55) & (hi > 0.35)) | (((hue > 0.97) | (hue < 0.03)) & (sat > 0.6) & (hi > 0.5))) \
        & (co[:, 2] < head_at[2] - 0.2) & (co[:, 2] > low[2] - 0.15)
    slope, level = np.polyfit(co[collar, 1], co[collar, 2], 1)       # his collar sits lower at the front than the back
    above = co[:, 2] - (slope * co[:, 1] + level)
    axis = (head_at - low) / np.linalg.norm(head_at - low)
    along = (co - low) @ axis
    out = np.linalg.norm((co - low) - np.outer(along, axis), axis=1)
    starts, full, lifts, clear = CHEEKS
    share = np.array([ease(v) for v in (out - starts) / (full - starts)]) * np.array([ease(v) for v in (above - lifts) / (clear - lifts)])
    share[collar] = 0.0
    necks = {ob.vertex_groups[b].index for b in neck_bones()}
    head = ob.vertex_groups['head']
    sat_ = sat_bones()
    moved = 0
    for i in np.where(share > 0.01)[0]:
        freed = 0.0
        v = me.vertices[i]
        was = carried({g_.group: g_.weight for g_ in v.groups}, sat_)
        for g_ in v.groups:
            if g_.group in necks and g_.weight > 0.001:
                freed += g_.weight * share[i]
                ob.vertex_groups[g_.group].add([int(i)], g_.weight * (1.0 - share[i]), 'REPLACE')
        if freed > 0.0:
            head.add([int(i)], freed, 'ADD')
            now = carried({g_.group: g_.weight for g_ in v.groups}, sat_)
            v.co = Vector((np.linalg.inv(now) @ was @ np.array([v.co.x, v.co.y, v.co.z, 1.0]))[:3])   # so that he sits as he sat
            moved += 1
    me.update()
    return moved


def neck_bones():
    """The bones of his neck, lowest first."""
    bone, out = arm().data.bones['head'].parent, []
    while bone and bone.name.startswith('neck'):
        out.insert(0, bone.name)
        bone = bone.parent
    return out


def ears():
    """Make each ear move as a whole, and leave his face alone.

    Meshy's rig lets the first bone of each ear pull on half his head (eye, cheek, headset and all), and only
    partly on the low outside corner of the ear itself. An ear is found here by what it is: a thin flap. From the
    top of the ear, spread across his skin wherever it is thin (measured through to the far side, straight in and
    at a slant) and not headset. That set is the ear, lobe and all. Inside it the first ear bone takes everything his head bone had,
    easing off over a narrow band where the ear joins his head; outside it, his head takes everything the ear
    bone had. It works from whatever the weights are now, so it can be run again."""
    import heapq
    from mathutils.bvhtree import BVHTree
    a, ob, me = arm(), skin(), skin().data
    thin, band, reach = EARS[:3]
    n = len(me.vertices)
    co = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3).astype('f8')
    no = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('normal', no)
    no = no.reshape(-1, 3).astype('f8')
    paint = paint_at_points().astype('f8')
    hi, lo = paint.max(1), paint.min(1)
    d = np.maximum(hi - lo, 1e-6)
    r, g, b = paint[:, 0], paint[:, 1], paint[:, 2]
    hue = np.where(hi == r, ((g - b) / d) % 6, np.where(hi == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    headset = (hue > 0.42) & (hue < 0.60) & ((hi - lo) / np.maximum(hi, 1e-6) > 0.2)
    tree = BVHTree.FromPolygons([tuple(p) for p in co], [tuple(p.vertices) for p in me.polygons])
    edges = np.zeros(len(me.edges) * 2, 'i4')
    me.edges.foreach_get('vertices', edges)
    edges = edges.reshape(-1, 2)
    # Meshy's skin comes in as many separate patches (one for each piece of its paint), with a point on each side of
    # every join. Treat the points that share a place as one, or the spread stops at the first join it meets.
    _, firsts, which = np.unique(np.round(co, 5), axis=0, return_index=True, return_inverse=True)
    one = firsts[which.ravel()]
    edges = one[edges]
    head = ob.vertex_groups['head']
    report = {}
    for side in ('L', 'R'):
        first = a.data.bones['ear.%s_01' % side]
        bones = [first]
        while bones[-1].children:
            bones.append(bones[-1].children[0])
        root = np.array(first.head_local)
        tip = np.array(max((b.head_local for b in bones), key=lambda p: (p - first.head_local).length))
        axis = (tip - root) / np.linalg.norm(tip - root)
        index = {ob.vertex_groups[b.name].index: k for k, b in enumerate(bones)}
        index[head.index] = len(bones)
        along = (co - root) @ axis
        off = np.linalg.norm((co - root) - np.outer(along, axis), axis=1)
        near = np.where((off < reach) & (along > -0.30) & (co[:, 0] * (1.0 if side == 'L' else -1.0) > 0.02))[0]
        w = np.zeros((n, len(bones) + 1))
        for i in near:
            for g_ in me.vertices[i].groups:
                if g_.group in index:
                    w[i, index[g_.group]] = g_.weight
        # the fur tufts on top of his head are thin too, and sit by the inside of each ear: on that side an ear
        # starts a little way up its line; on the outside it runs right down to its lobe
        outward = (co[:, 0] - (root[0] + axis[0] * along)) * (1.0 if side == 'L' else -1.0)
        lowest = 0.08 - 0.38 * np.clip(outward / 0.08, 0.0, 1.0)
        # and an ear is a flap: it keeps to the slice of him, front to back, that its top lies in, and its lobe
        # hangs only a little below its first joint. His fur is grooved all over, so thinness alone finds too much.
        top = near[w[near, 1:len(bones)].sum(1) > 0.5]
        front, back, low = co[top, 1].min() - EARS[3], co[top, 1].max() + EARS[3], root[2] - EARS[4]
        is_thin = np.zeros(n, bool)
        for i in near:                                     # straight through his skin from here: how far to the other side?
            # above where the ear leaves his head, everything round the ear's line is ear, thick or thin, and a
            # little further back: or the back of the ear's base is left standing as a blob when the ear folds down
            up = along[i] > EARS[6] and abs(outward[i]) < 0.20
            if headset[i] or along[i] < lowest[i] or not front < co[i, 1] < (back + 0.10 if up else back) or co[i, 2] < low:
                continue
            if up:
                is_thin[i] = True
                continue
            inward = Vector(-no[i])
            across = inward.orthogonal().normalized()
            start_ = Vector(co[i]) + inward * 0.004
            ways = [inward] + [(inward + Matrix.Rotation(k * math.pi / 3, 3, inward) @ across).normalized() for k in range(6)]
            is_thin[i] = any(tree.ray_cast(start_, way, thin)[0] is not None for way in ways)   # at a rim, straight in runs along the flap
        shared = np.zeros(n, bool)
        np.logical_or.at(shared, one[near], is_thin[near])
        is_thin = shared
        mine = np.zeros(n, bool)
        mine[one[near]] = True
        links = {}
        for i, j in edges[mine[edges[:, 0]] & mine[edges[:, 1]] & (edges[:, 0] != edges[:, 1])]:
            links.setdefault(int(i), []).append(int(j))
            links.setdefault(int(j), []).append(int(i))
        places = np.unique(one[near])
        ear = np.zeros(n, bool)
        todo = [int(one[i]) for i in near if is_thin[one[i]] and w[i, 1:len(bones)].sum() > 0.5]   # the top of the ear
        ear[todo] = True
        while todo:
            i = todo.pop()
            for j in links.get(i, ()):
                if not ear[j] and is_thin[j]:
                    ear[j] = True
                    todo.append(j)
        for _ in range(3):                                 # a stray point inside the ear that measured thick is still ear
            for i in places:
                if not ear[i] and not headset[i]:
                    by = links.get(int(i), ())
                    if by and sum(ear[j] for j in by) >= 0.6 * len(by):
                        ear[i] = True
        # how far each ear point is, over his skin, from where the ear stops
        far = {}
        heap = [(0.0, i) for i in np.where(ear)[0] if any(not ear[j] for j in links.get(int(i), ()))]
        heapq.heapify(heap)
        while heap:
            dist, i = heapq.heappop(heap)
            if i in far:
                continue
            far[int(i)] = dist
            for j in links.get(int(i), ()):
                if ear[j] and j not in far:
                    heapq.heappush(heap, (dist + float(np.linalg.norm(co[i] - co[j])), j))
        group = ob.vertex_groups[first.name]
        both_ears = {g_.index for g_ in ob.vertex_groups if g_.name.startswith('ear')}
        changed = 0
        for v in me.vertices:                              # every point this bone pulls on, wherever it is on him
            i, held, others = v.index, 0.0, []
            for g_ in v.groups:
                if g_.group == group.index:
                    held = g_.weight
                elif g_.group not in both_ears and g_.weight > 0.0:
                    others.append((g_.group, g_.weight))
            if held < 0.001 and not ear[one[i]]:
                continue
            # everything but the ears' bones is pooled: the first ear bone takes its share of the pool, and
            # whatever held the point before (head, neck, a face bone) keeps the rest in the same proportions
            spare = sum(weight for _, weight in others)
            pool = held + spare
            if pool < 0.001:
                continue
            share = ease(far.get(int(one[i]), band) / band) if ear[one[i]] else 0.0
            if abs(pool * share - held) < 0.002:
                continue
            group.add([int(i)], pool * share, 'REPLACE')
            if spare > 0.0:
                for k, weight in others:
                    ob.vertex_groups[k].add([int(i)], weight / spare * pool * (1.0 - share), 'REPLACE')
            else:
                head.add([int(i)], pool * (1.0 - share), 'REPLACE')
            changed += 1
        # Up the ear, Meshy shares its points between the ear's bones one way on its front and another on its back.
        # Bend the ear part-way up and the two faces part: the back stands out behind the front as a blob. So inside
        # the ear the bones share each point by how far up the ear it is and nothing else, the bend spread over a
        # stretch each side of the second joint. Outside the ear the upper bones hold nothing.
        upper = [ob.vertex_groups[b.name] for b in bones[1:]]
        uppers = {g_.index for g_ in upper}
        bend_at = float((np.array(bones[1].head_local) - root) @ axis)
        evened = 0
        for v in me.vertices:
            i = v.index
            held = {g_.group: g_.weight for g_ in v.groups if g_.group == group.index or g_.group in uppers}
            above = sum(x for g_, x in held.items() if g_ in uppers)
            if ear[one[i]]:
                total = sum(held.values())
                if total < 0.001:
                    continue
                f = ease((along[i] - (bend_at - EARS[5])) / (2.0 * EARS[5]))
                if abs(above - total * f) < 0.002 and all(held.get(g_.index, 0.0) < 0.002 for g_ in upper[1:]):
                    continue
                group.add([i], total * (1.0 - f), 'REPLACE')
                upper[0].add([i], total * f, 'REPLACE')
                for g_ in upper[1:]:
                    g_.add([i], 0.0, 'REPLACE')
                evened += 1
            elif above > 0.002:
                for g_ in upper:
                    g_.add([i], 0.0, 'REPLACE')
                head.add([i], above, 'ADD')
                evened += 1
        report[side] = {'ear_places': int(ear.sum()), 'changed': changed, 'evened': evened, 'lowest': round(float(co[ear][:, 2].min()), 3)}
    return report


def beans():
    """Paint pink toe beans under his paws: a big pad and four toes on each."""
    a, ob, me = arm(), skin(), skin().data
    n = len(me.vertices)
    co = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3).astype('f8')
    normal = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('normal', normal)
    normal = normal.reshape(-1, 3)
    pink, found = np.zeros(n), {}
    for leg in ('front', 'rear'):
        for side in ('L', 'R'):
            limb = chain(('clavicle.' if leg == 'front' else 'rearUpr.') + side)
            groups = {ob.vertex_groups[name].index for name in limb[-2:]}
            mine = np.array([sum(g.weight for g in v.groups if g.group in groups) > 0.5 for v in me.vertices])
            heel = np.array(a.matrix_world @ a.data.bones[limb[-2]].head_local)[:2]
            toe = np.array(a.matrix_world @ a.data.bones[limb[-1]].tail_local)[:2]
            # some rigs hang his paw on other bones, so also take what is round the paw's bones, near the ground
            mine |= (np.linalg.norm(co[:, :2] - (heel + toe) / 2, axis=1) < 0.10) & (co[:, 2] < co[:, 2].min() + 0.12)
            floor = co[mine, 2].min()
            under = np.where(mine & (normal[:, 2] < -0.5) & (co[:, 2] < floor + 0.06))[0]
            if len(under) < 10:
                continue
            # which way his toes point: along the bone before the last two. The last joints of a paw can sit a hair
            # behind one another, and going by those painted the beans back to front, the big pad ahead of the toes.
            ball = np.array(a.matrix_world @ a.data.bones[limb[-3]].head_local)[:2]
            ahead = heel - ball if np.linalg.norm(heel - ball) > 0.03 else toe - heel
            ahead = ahead / np.linalg.norm(ahead)
            across = np.array((-ahead[1], ahead[0]))
            flat = co[under, :2] - co[under, :2].mean(0)
            u, v = flat @ across, flat @ ahead
            half = np.percentile(np.abs(u), 95)
            u, v = u / half, (v - (v.max() + v.min()) / 2) / half
            spots = [(0.0, -0.35, 0.55, 0.42)] + [(x, y, 0.21, 0.24) for x, y in ((-0.66, 0.35), (-0.23, 0.62), (0.23, 0.62), (0.66, 0.35))]
            amount = np.zeros(len(under))
            for x, y, rx, ry in spots:
                d = np.sqrt(((u - x) / rx) ** 2 + ((v - y) / ry) ** 2)
                amount = np.maximum(amount, np.clip((1.15 - d) / 0.3, 0.0, 1.0))
            pink[under] = amount
            found[leg + '.' + side] = int((amount > 0.5).sum())
    att = me.attributes.get('j5_bean') or me.attributes.new('j5_bean', 'FLOAT', 'POINT')
    att.data.foreach_set('value', pink.astype('f4'))
    for mat in me.materials:
        nt = mat.node_tree
        old = nt.nodes.get('jb_mix')
        if old:                                            # put things back as they were first
            source = old.inputs[6].links[0].from_socket
            for link in list(old.outputs[2].links):
                nt.links.new(source, link.to_socket)
            for k in [k for k in nt.nodes if k.name.startswith('jb_')]:
                nt.nodes.remove(k)
        tex = next(k for k in nt.nodes if k.type == 'TEX_IMAGE' and k.image and k.outputs['Color'].is_linked)
        users = [link.to_socket for link in tex.outputs['Color'].links]
        mix = nt.nodes.new('ShaderNodeMix')
        mix.name, mix.data_type = 'jb_mix', 'RGBA'
        where = nt.nodes.new('ShaderNodeAttribute')
        where.name, where.attribute_name = 'jb_where', 'j5_bean'
        nt.links.new(where.outputs['Fac'], mix.inputs[0])
        nt.links.new(tex.outputs['Color'], mix.inputs[6])
        mix.inputs[7].default_value = BEANS
        for socket in users:
            nt.links.new(mix.outputs[2], socket)
    return found


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

    His eyes are painted on his head, so there is nothing to shut. Round each eye, every point of his face is told
    where it is in that eye: how high (0 at the bottom, 1 at the top), how far across, and how far out from the
    middle (1 at the edge of the lid's patch). Those run smoothly over his face, so the lid's edge and the lash line
    drawn from them are clean curves however coarse his skin is. The paint is swapped for fur colour above the
    lid's edge. Shut, the whole eye is fur and the line is left as a gentle dip; with joy it is a happy arch.
    The amounts are numbers on his bones ('blink', 'blink_l', 'blink_r', 'joy'), so a motion can key them."""
    from mathutils import kdtree
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
    height, wide, out, which, lean = np.zeros(n), np.zeros(n), np.full(n, 3.0), np.zeros(n), np.zeros(n)
    found, lid = {}, []
    tint, tinted = np.zeros((n, 3)), np.zeros(n, bool)
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
        e = hp[eye]
        c = (e.min(0) + e.max(0)) / 2
        rx, rz = (e[:, 0].max() - e[:, 0].min()) / 2, (e[:, 2].max() - e[:, 2].min()) / 2
        across, up = (hp[:, 0] - c[0]) / rx, (hp[:, 2] - c[2]) / rz
        far = (np.abs(across) ** 3 + np.abs(up) ** 3) ** (1 / 3.0)      # 1 on a rounded box that just holds the eye
        far /= max(1.0, float(np.percentile(far[eye], 99))) * EYE[0]
        near = mine & (far < 2.2) & (np.abs(hp[:, 1] - c[1]) < 2.5 * max(rx, rz))
        height[head[near]], wide[head[near]], out[head[near]] = (up[near] + 1) / 2, across[near] ** 2, far[near]
        lean[head[near]] = across[near] * (1.0 if side == 'his left' else -1.0)     # -1 by his nose, 1 at the outer corner
        which[head[near]] = 1.0 if side == 'his left' else 0.0
        ring = near & fur & (far > 1.05) & (far < 1.35)        # a band of fur round the eye, for the lid's colour
        lid.append(linear(paint[head[ring]]).mean(0))
        if LID:
            # Meshy painted the fur under each eye lighter. Carried up on to the lid it made the shut eye two-tone,
            # light below its line: the shape of an eye again. So the lid takes the fur above and beside the eye.
            ring = near & fur & (far > LID['paint'][1]) & (far < LID['paint'][1] + 0.3) & (up > -0.3)
        # The lid's colour, point by point: the fur round the eye carried in across it, so that a shut eye is the
        # colour of the fur beside it all the way round. One flat colour showed as a patch laid over the eye.
        src, dst = np.where(ring)[0], np.where(near & (far < (LID['paint'][1] + 0.05 if LID else 1.12)))[0]
        tree = kdtree.KDTree(len(src))
        for k, i in enumerate(src):
            tree.insert(hp[i], k)
        tree.balance()
        colours = linear(paint[head[src]])
        carried_ = np.zeros((len(head), 3))
        for i in dst:
            hits = tree.find_n(hp[i], 32)
            w = np.array([1.0 / (d_ * d_ + 4e-4) for _, _, d_ in hits])
            carried_[i] = (colours[[k for _, k, _ in hits]] * w[:, None]).sum(0) / w.sum()
        inside = np.zeros(len(head), bool)
        inside[dst] = True
        for _ in range(8):                                  # and evened out, point to point
            was_ = carried_.copy()
            for i in dst:
                by = [j for j in nb[i] if inside[j]]
                if by:
                    carried_[i] = 0.5 * was_[i] + 0.5 * was_[by].mean(0)
        tint[head[dst]] = carried_[dst]
        tinted[head[dst]] = True
        found[side] = {'points': int(eye.sum()), 'middle': [round(float(v), 3) for v in c], 'half_size': [round(float(rx), 3), round(float(rz), 3)]}
    for name, values in (('j5_eye', height), ('j5_eyea', wide), ('j5_eyem', out), ('j5_side', which), ('j5_eyeo', lean)):
        att = me.attributes.get(name) or me.attributes.new(name, 'FLOAT', 'POINT')
        att.data.foreach_set('value', values.astype('f4'))
    fur = np.mean(lid, axis=0)
    tint[~tinted] = fur
    att = me.attributes.get('j5_lidc') or me.attributes.new('j5_lidc', 'FLOAT_COLOR', 'POINT')
    att.data.foreach_set('color', np.c_[tint, np.ones(n)].astype('f4').ravel())
    for prop in LIDS + (JOY, FROWN):                       # both eyes, his left eye alone, his right eye alone, joy and frown
        if prop not in a:
            a[prop] = 0.0
        a.id_properties_ui(prop).update(min=0.0, max=1.0)
    reach, arch, low, dip, thick, high, bow, slant = EYE
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

        def between(name, t, x, y):
            """x where t is 0, y where t is 1."""
            k = node('ShaderNodeMix', name, data_type='FLOAT')
            for socket, value in zip((k.inputs[0], k.inputs[2], k.inputs[3]), (t, x, y)):
                if isinstance(value, (int, float)):
                    socket.default_value = value
                else:
                    nt.links.new(value, socket)
            return k.outputs[0]

        def spot(name):
            return node('ShaderNodeAttribute', name, attribute_name=name).outputs['Fac']

        said = {}
        for prop in LIDS + (JOY, FROWN):
            value = node('ShaderNodeValue', prop)
            drv = value.outputs[0].driver_add('default_value').driver
            drv.type = 'SUM'
            var = drv.variables.new()
            var.type = 'SINGLE_PROP'
            var.targets[0].id = a
            var.targets[0].data_path = '["%s"]' % prop
            said[prop] = value.outputs[0]
        amount = between('eye', spot('j5_side'), maths('right', 'MAXIMUM', said['blink'], said['blink_r']),
                         maths('left', 'MAXIMUM', said['blink'], said['blink_l']))
        joy = said[JOY]
        # the lid's edge is a curve: it arches as it comes down, and ends as a dip (or, with joy, an arch)
        curve = between('curve', joy, maths('curve0', 'MULTIPLY_ADD', amount, dip + arch, -arch), -bow)
        field = maths('field', 'SUBTRACT', spot('j5_eye'), maths('bend', 'MULTIPLY', curve, spot('j5_eyea')))
        ends = between('ends', amount, 1.25, between('rest', joy, low, high))
        off = maths('off', 'SUBTRACT', maths('off0', 'SUBTRACT', field, ends),
                    maths('slant', 'MULTIPLY', maths('slant0', 'MULTIPLY', said[FROWN], slant), spot('j5_eyeo')))   # a frown: lower by his nose
        on = ramp('on', amount, 0.0, 0.08, 0.0, 1.0)
        patch = ramp('patch', spot('j5_eyem'), LID['paint'][0] if LID else 0.90, LID['paint'][1] if LID else 1.0, 1.0, 0.0)
        lower = None
        if LID:
            # The lower lid rises to meet the upper one, over the last half of the shutting. Its edge starts as the
            # curve of the eye's own lower rim, just under it, and ends as the line the lids meet on, so it comes
            # up out of the bottom of the eye and not as a mound out of his cheek. It carries a thin dark edge, as
            # the eye's rim does.
            rise = ramp('rise', amount, 0.45, 0.95, 0.0, 1.0)
            wait, bend = LID['lower']
            level = between('level', rise, wait, between('rest2', joy, low, high))
            curve_ = between('lcurve', rise, bend, between('fcurve', joy, dip, -bow))
            below = maths('below', 'SUBTRACT', maths('below0', 'SUBTRACT', maths('lfield', 'SUBTRACT', spot('j5_eye'), maths('lbend', 'MULTIPLY', curve_, spot('j5_eyea'))), level),
                          maths('lslant', 'MULTIPLY', maths('lslant0', 'MULTIPLY', maths('lslant1', 'MULTIPLY', said[FROWN], slant), spot('j5_eyeo')), rise))
            under = ramp('under', below, 0.02, 0.06, 1.0, 0.0)   # a little past its edge, so that no eye shows between the two lids when they meet
            lower = maths('ledge', 'MULTIPLY', ramp('lnear', maths('lgap', 'ABSOLUTE', maths('lgap0', 'SUBTRACT', below, 0.03)), thick * 0.3, thick * 0.65, 1.0, 0.0),
                          ramp('lon', rise, 0.02, 0.2, 0.0, 1.0))
        else:
            under = ramp('under', amount, 0.70, 0.92, 0.0, 1.0)     # as he shuts, what is left below the line goes too
        lidded = maths('lidded', 'MULTIPLY', maths('cover', 'MAXIMUM', ramp('above', off, -0.02, 0.02, 0.0, 1.0), under),
                       maths('in', 'MULTIPLY', patch, on))
        tips = ramp('tips', spot('j5_eyea'), 0.62, 0.92, 1.0, 0.0)  # the line stops short of the eye's corners
        edge = ramp('near', maths('gap', 'ABSOLUTE', off), thick * 0.7, thick * 1.3, 1.0, 0.0)
        if lower is not None:
            edge = maths('edges', 'MAXIMUM', edge, lower)
        lash = maths('lash', 'MULTIPLY', edge, maths('in2', 'MULTIPLY', maths('in3', 'MULTIPLY', patch, tips), on))
        for name, amount_, colour in (('lid', lidded, (float(fur[0]), float(fur[1]), float(fur[2]), 1.0)), ('line', lash, LINE)):
            mix = node('ShaderNodeMix', name, data_type='RGBA')
            nt.links.new(amount_, mix.inputs[0])
            nt.links.new(paint_in, mix.inputs[6])
            mix.inputs[7].default_value = colour
            if name == 'lid':                              # the fur's own colour at each point of the lid
                nt.links.new(node('ShaderNodeAttribute', 'lidc', attribute_name='j5_lidc').outputs['Color'], mix.inputs[7])
            paint_in = mix.outputs[2]
        nt.links.new(paint_in, bsdf.inputs['Base Color'])
    return found


def say(amount=1.0):
    """Open his mouth: 0 = shut, 1 = wide. With a jaw it turns the jaw; without, it sets the number his mouth follows."""
    a = arm()
    if JAW:
        turn('jaw_01', x=SHUT * (1.0 - amount))
    elif TALK in a:
        a[TALK] = float(amount)
        a.update_tag()


def mouth():
    """Give a closed-mouth model a mouth that opens: a hollow that deepens under his mouth line, painted as a mouth.

    He was modelled with his mouth shut, which is why his shut face is neat, so there is no inside to show. Opening
    is a shape ('open') that sinks a mouth-shaped patch of his chin into his head and drops his chin a little, with
    the patch painted dark and a pink tongue in it. How open is a number on his bones called 'talk'."""
    a, ob, me = arm(), skin(), skin().data
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
    mx, half, front, low, high = MOUTH
    face = (np.abs(co[:, 0] - mx) < half) & (co[:, 1] < front) & (co[:, 2] > low) & (co[:, 2] < high)
    nose = face & ((hue > 0.88) | (hue < 0.03)) & (sat > 0.25) & (hi > 0.45) & (np.abs(co[:, 0] - mx) < 0.06)      # the pink of his nose
    nose &= co[:, 2] > 0.10                                # not his collar
    top = float(co[nose, 2].min()) - OPEN[4]               # his mouth line sits a little under his nose
    wide, tall, deep, drop = OPEN[:4]
    x = (co[:, 0] - mx) / wide
    fur = (sat < 0.45) | (hue > 0.6)                        # not his headset's mic
    # The top edge of the opening is his own mouth line, so that an open mouth is the shape of his mouth. The line is
    # paint: across his muzzle, strip by strip, it is the darkest points near the height it should be at. It sits
    # 0.02 to 0.03 under where a level edge was put at first (which left the opening above his smile, not in it),
    # lower on his right than his left, and curls up toward the corners.
    rise = co[:, 2] - top
    about = face & fur & (co[:, 1] < co[nose, 1].max() + 0.10) & (rise > -0.04) & (rise < 0.06)
    found_at, found_up = [], []
    for x0 in np.arange(-0.13, 0.131, 0.01):
        strip = about & (np.abs(co[:, 0] - mx - x0) < 0.005)
        if strip.sum() < 8:
            continue
        darkest = strip.copy()
        darkest[strip] = hi[strip] < np.median(hi[strip]) * 0.72
        if darkest.sum() >= 3 and np.ptp(rise[darkest]) < 0.022:
            found_at.append(float(x0))
            found_up.append(float(np.median(rise[darkest])))
    line = top + np.interp(co[:, 0] - mx, found_at, found_up) if len(found_at) >= 4 else np.full(n, top)
    down = (line - co[:, 2]) / tall                         # 0 at the mouth line, 1 at the bottom of a wide open mouth
    inside = face & fur & (np.abs(x) < 1.0) & (down > 0.0) & (down < 1.4) & (co[:, 1] < co[nose, 1].max() + 0.10)
    shape = np.sqrt(np.clip(1.0 - x * x, 0.0, 1.0))         # a half ellipse hanging from the line
    # a point is in the mouth when this is under how open he is. A part-open mouth is narrower as well as shallower
    # (the same half ellipse, smaller): one that was only shallower was a thin red smear right across his muzzle
    coord = np.full(n, 9.0)
    for part in np.linspace(1.0, 0.04, 49):
        span = part ** 0.4
        coord[inside & (np.abs(x) < span) & (down < part * np.sqrt(np.clip(1.0 - (x / span) ** 2, 0.0, 1.0)))] = part
    tongue = np.where(inside, np.clip(1.6 - np.sqrt((x / 0.62) ** 2 + ((down - 0.80) / 0.42) ** 2) * 1.6, 0.0, 1.0), 0.0)
    x_o, down_o = (co[:, 0] - mx) / 0.085, (top - co[:, 2]) / 0.075     # (the 'o' keeps its own place and size, measured from under his nose)
    round_ = np.where(face & fur & (np.abs(x_o) < 1.0) & (down_o > 0.0) & (down_o < 1.4) & (co[:, 1] < co[nose, 1].max() + 0.10), np.sqrt((x_o / 0.36) ** 2 + ((down_o - 0.44) / 0.44) ** 2), 9.0)   # and for a small round 'o' under the middle of his mouth line: a point is in it when this is under how round his mouth is
    if not me.shape_keys:
        ob.shape_key_add(name='Basis')
    old = me.shape_keys.key_blocks.get('open')
    if old:
        old.driver_remove('value')
        ob.shape_key_remove(old)
    key = ob.shape_key_add(name='open', from_mix=False)
    full = co.copy()
    hollow = np.clip(1.15 - coord, 0.0, 1.0) * inside       # deepest in the middle, nothing at the rim
    full[:, 1] += deep * np.sqrt(hollow)
    under = face & fur & (down > 0.2) & (np.abs(x) < 1.6)   # his chin, below the opening, drops and fades out to the sides
    fall = np.clip((down - 0.2) / 0.5, 0.0, 1.0) * np.clip(1.6 - np.abs(x), 0.0, 1.0) * np.clip((2.6 - down) / 0.8, 0.0, 1.0)
    full[under, 2] -= (drop * fall)[under]
    key.data.foreach_set('co', full.astype('f4').ravel())
    if TALK not in a:
        a[TALK] = 0.0
    a.id_properties_ui(TALK).update(min=0.0, max=1.0)
    drv = key.driver_add('value').driver
    drv.type = 'SUM'
    var = drv.variables.new()
    var.type = 'SINGLE_PROP'
    var.targets[0].id = a
    var.targets[0].data_path = '["%s"]' % TALK
    pit = full.copy() * 0.0                                # the 'o': a smaller, rounder hollow, and his chin drops a little
    pit[:, 1] = deep * 0.7 * np.sqrt(np.clip(1.1 - round_, 0.0, 1.0) * inside)
    pit[under, 2] -= (drop * 0.35 * fall)[under]
    driven(shaped('oh', co, pit), 'o', {'o': OH})
    for name, values in (('j5_mouth', coord), ('j5_tongue', tongue), ('j5_oh', round_)):
        att = me.attributes.get(name) or me.attributes.new(name, 'FLOAT', 'POINT')
        att.data.foreach_set('value', values.astype('f4'))
    for mat in me.materials:
        nt = mat.node_tree
        bsdf = next(k for k in nt.nodes if k.type == 'BSDF_PRINCIPLED')
        source = bsdf.inputs['Base Color'].links[0].from_socket
        if nt.nodes.get('jm_dark'):                        # put things back as they were first
            source = nt.nodes['jm_dark'].inputs[6].links[0].from_socket
        for k in [k for k in nt.nodes if k.name.startswith('jm_')]:
            nt.nodes.remove(k)

        def node(kind, name, **settings):
            k = nt.nodes.new(kind)
            k.name = 'jm_' + name
            for prop, value in settings.items():
                setattr(k, prop, value)
            return k

        amount = node('ShaderNodeValue', 'talk')
        d2 = amount.outputs[0].driver_add('default_value').driver
        d2.type = 'SUM'
        v2 = d2.variables.new()
        v2.type = 'SINGLE_PROP'
        v2.targets[0].id = a
        v2.targets[0].data_path = '["%s"]' % TALK
        where = node('ShaderNodeAttribute', 'where', attribute_name='j5_mouth')
        gap = node('ShaderNodeMath', 'gap', operation='SUBTRACT')
        nt.links.new(amount.outputs[0], gap.inputs[0])
        nt.links.new(where.outputs['Fac'], gap.inputs[1])
        hole = node('ShaderNodeMapRange', 'hole')           # 1 inside the opening, with a soft rim
        nt.links.new(gap.outputs[0], hole.inputs[0])
        for i, value in zip((1, 2, 3, 4), (0.0, 0.06, 0.0, 1.0)):
            hole.inputs[i].default_value = value
        said = node('ShaderNodeValue', 'oh')               # and the same for the 'o'
        d3 = said.outputs[0].driver_add('default_value').driver
        d3.type = 'SUM'
        v3 = d3.variables.new()
        v3.type = 'SINGLE_PROP'
        v3.targets[0].id = a
        v3.targets[0].data_path = '["%s"]' % OH
        gap2 = node('ShaderNodeMath', 'gap2', operation='SUBTRACT')
        nt.links.new(said.outputs[0], gap2.inputs[0])
        nt.links.new(node('ShaderNodeAttribute', 'where2', attribute_name='j5_oh').outputs['Fac'], gap2.inputs[1])
        hole2 = node('ShaderNodeMapRange', 'hole2')
        nt.links.new(gap2.outputs[0], hole2.inputs[0])
        for i, value in zip((1, 2, 3, 4), (0.0, 0.06, 0.0, 1.0)):
            hole2.inputs[i].default_value = value
        either = node('ShaderNodeMath', 'either', operation='MAXIMUM')
        nt.links.new(hole.outputs[0], either.inputs[0])
        nt.links.new(hole2.outputs[0], either.inputs[1])
        dark = node('ShaderNodeMix', 'dark', data_type='RGBA')
        nt.links.new(either.outputs[0], dark.inputs[0])
        nt.links.new(source, dark.inputs[6])
        dark.inputs[7].default_value = (0.045, 0.008, 0.012, 1.0)    # dark, and only a little red: a brighter red showed as a smear at the rim
        lick = node('ShaderNodeMath', 'lick', operation='MULTIPLY')
        nt.links.new(node('ShaderNodeAttribute', 'tongue', attribute_name='j5_tongue').outputs['Fac'], lick.inputs[0])
        nt.links.new(hole.outputs[0], lick.inputs[1])
        pink = node('ShaderNodeMix', 'pink', data_type='RGBA')
        nt.links.new(lick.outputs[0], pink.inputs[0])
        nt.links.new(dark.outputs[2], pink.inputs[6])
        pink.inputs[7].default_value = (0.85, 0.30, 0.36, 1.0)
        deep = node('ShaderNodeMix', 'deep', data_type='RGBA')   # the 'o' is too small to show a tongue: it is dark inside
        nt.links.new(hole2.outputs[0], deep.inputs[0])
        nt.links.new(pink.outputs[2], deep.inputs[6])
        deep.inputs[7].default_value = (0.022, 0.004, 0.008, 1.0)
        nt.links.new(deep.outputs[2], bsdf.inputs['Base Color'])
    return {'points': int(inside.sum()), 'line_at': round(top, 3)}


def grump():
    """Let a frown turn his mouth down as well as slant his lids.

    His mouth is a smile, painted on and modelled in. A shape ('frown') slides the skin at its corners down, taking
    the painted line with it, so the smile flattens and then turns down. It follows the same number on his bones as
    the lids do ('frown')."""
    a, ob, me = arm(), skin(), skin().data
    n = len(me.vertices)
    co = np.zeros(n * 3, 'f4')
    me.shape_keys.key_blocks['Basis'].data.foreach_get('co', co)
    co = co.reshape(-1, 3).astype('f8')
    paint = paint_at_points().astype('f8')
    hi, lo = paint.max(1), paint.min(1)
    d = np.maximum(hi - lo, 1e-6)
    r, g, b = paint[:, 0], paint[:, 1], paint[:, 2]
    hue = np.where(hi == r, ((g - b) / d) % 6, np.where(hi == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    sat = (hi - lo) / np.maximum(hi, 1e-6)
    mx, half, front, low, high = MOUTH
    face = (np.abs(co[:, 0] - mx) < half) & (co[:, 1] < front) & (co[:, 2] > low) & (co[:, 2] < high)
    nose = face & ((hue > 0.88) | (hue < 0.03)) & (sat > 0.25) & (hi > 0.45) & (np.abs(co[:, 0] - mx) < 0.06) & (co[:, 2] > 0.10)
    line = float(co[nose, 2].min()) - OPEN[4]              # the middle of his mouth line, as in mouth()
    corner, drop, his_right, his_left = GRUMP
    out = np.abs(co[:, 0] - mx)
    across = np.clip((out - 0.02) / (corner - 0.02), 0.0, 1.0)
    across = across * across * (3 - 2 * across) * np.clip((corner * 2.5 - out) / (corner * 1.1), 0.0, 1.0)   # full at the corners, none in the middle or out on his cheeks
    up = co[:, 2] - line
    band = np.clip((up + 0.08) / 0.04, 0.0, 1.0) * np.clip((0.065 - up) / 0.03, 0.0, 1.0)                   # from his chin to well short of his eyes
    near = np.clip((co[nose, 1].max() + 0.14 - co[:, 1]) / 0.05, 0.0, 1.0)                                  # the front of his face only
    fur = ~headset_paint(co, me, 6)                         # not his headset's microphone, which sits by one corner of his mouth (well round it: its shaded underside is not teal)
    old = me.shape_keys.key_blocks.get('frown')
    if old:
        old.driver_remove('value')
        ob.shape_key_remove(old)
    key = ob.shape_key_add(name='frown', from_mix=False)
    full = co.copy()
    full[:, 2] -= drop * np.where(co[:, 0] > mx, his_left, his_right) * across * band * near * (face & fur)   # his smile is lopsided: the corner on his right starts lower, so it drops less, and they end level
    key.data.foreach_set('co', full.astype('f4').ravel())
    if FROWN not in a:
        a[FROWN] = 0.0
    driven(key, 'max(f, d, 0.4 * o)', {'f': FROWN, 'd': SAD, 'o': OH})   # a frown turns his mouth down and slants his lids; 'sad' turns his mouth down alone; an 'o' takes some of the smile off
    if LID:                                                # a wink lifts the corner of his mouth on that side
        # The whole of that side of his mouth goes with its corner: the corner most, its whisker pad with it, less
        # toward the middle and out on his cheek. Lifting the corner alone bent his smile into a kink. And it eases
        # off toward his microphone, where a hard stop left a step in his skin.
        from mathutils import kdtree
        smooth = lambda t: np.clip(t, 0.0, 1.0) ** 2 * (3 - 2 * np.clip(t, 0.0, 1.0))
        kit = np.where(headset_paint(co, me, 0))[0]
        tree = kdtree.KDTree(len(kit))
        for k, i in enumerate(kit):
            tree.insert(co[i], k)
        tree.balance()
        reach_ = face & (co[:, 2] > line - 0.2) & (co[:, 2] < line + 0.2)
        clear = np.zeros(n)
        clear[reach_] = [tree.find(co[i])[2] for i in np.where(reach_)[0]]
        wide_ = smooth((out - 0.055) / (corner - 0.055)) * smooth((corner * 3.2 - out) / (corner * 1.6))   # none under his nose
        tall = np.exp(-(up / np.where(up > 0, 0.035, 0.06)) ** 2)                                       # and little above his mouth line, toward his nose
        share = wide_ * tall * near * reach_ * smooth((clear - 0.012) / 0.05)
        lift_, draw = LID['smirk']
        for name, prop, other, mine in (('smirk_l', LIDS[1], LIDS[2], co[:, 0] > mx), ('smirk_r', LIDS[2], LIDS[1], co[:, 0] < mx)):
            lifted = np.zeros((n, 3))
            lifted[:, 2] = lift_ * share * mine
            lifted[:, 0] = draw * share * mine * np.sign(co[:, 0] - mx)
            # only when that eye is shut and the other is not: the flat rig shuts both eyes for a blink one eye at a time
            driven(shaped(name, co, lifted), 'max(s - max(b, o), 0) * (0.5 + 0.5 * j)', {'s': prop, 'o': other, 'b': LIDS[0], 'j': JOY})
        # an 'o' (mouth()) draws the corners of his mouth in toward the middle a little
        drawn = np.zeros((n, 3))
        drawn[:, 0] = -LID['oh'] * share * np.sign(co[:, 0] - mx)
        driven(shaped('pucker', co, drawn), 'o', {'o': OH})
    took = across * band * near * (face & fur) > 0.05
    return {'line_at': round(line, 3), 'his_left': int((took & (co[:, 0] > mx)).sum()), 'his_right': int((took & (co[:, 0] < mx)).sum())}


def browed():
    """Let his brows move. They are painted on, so each move is a shape that slides the patch of skin a brow is on.

    Four shapes: his left and his right brow lifting, and each tilting about its middle (inner end down). Each
    follows a number on his bones (BROWS) that runs from -1 to 1, so a brow can drop or tilt the other way too.
    A brow is found by its paint: the dark mark above each eye. Needs eyes() to have run."""
    a, ob, me = arm(), skin(), skin().data
    n = len(me.vertices)
    co = np.zeros(n * 3, 'f4')
    me.shape_keys.key_blocks['Basis'].data.foreach_get('co', co)
    co = co.reshape(-1, 3).astype('f8')

    def spot(name):
        v = np.zeros(n, 'f4')
        me.attributes[name].data.foreach_get('value', v)
        return v.astype('f8')

    height, far, side = spot('j5_eye'), spot('j5_eyem'), spot('j5_side')
    dark = paint_at_points().astype('f8').max(1) < 0.13
    lift, tilt = BROW
    found = {}
    for which, mine, inner in (('l', side > 0.5, -1.0), ('r', side < 0.5, 1.0)):       # his left eye is the one on +x
        mark = (far < 2.0) & mine & dark & (height > 1.0) & (height < 2.4)             # above the eye, not in it
        c = (co[mark].min(0) + co[mark].max(0)) / 2
        half = (co[mark, 0].max() - co[mark, 0].min()) / 2
        dx, dz = co[:, 0] - c[0], co[:, 2] - c[2]
        patch = (np.clip((half + 0.05 - np.abs(dx)) / 0.04, 0.0, 1.0)                  # the brow and a margin, fading out
                 * np.clip((dz + 0.065) / 0.035, 0.0, 1.0) * np.clip((0.09 - dz) / 0.06, 0.0, 1.0)   # stops short of the eye below
                 * np.clip((c[1] + 0.12 - co[:, 1]) / 0.05, 0.0, 1.0) * mine)           # the front of his head, his side
        patch = patch * patch * (3 - 2 * patch)
        turn_ = math.tan(math.radians(tilt)) * inner         # inner end down: toward his nose is -x on his left, +x on his right
        moves_ = (('brow_' + which, np.stack([np.zeros(n), np.zeros(n), lift * patch], 1)),
                  ('tilt_' + which, np.stack([turn_ * dz * patch, np.zeros(n), -turn_ * dx * patch], 1)))     # a turn about its middle
        for name, by in moves_:
            old = me.shape_keys.key_blocks.get(name)
            if old:
                old.driver_remove('value')
                ob.shape_key_remove(old)
            key = ob.shape_key_add(name=name, from_mix=False)
            key.slider_min, key.slider_max = -1.5, 1.5
            key.data.foreach_set('co', (co + by).astype('f4').ravel())
            if name not in a:
                a[name] = 0.0
            a.id_properties_ui(name).update(min=-1.5, max=1.5)
            drv = key.driver_add('value').driver
            drv.type = 'SUM'
            var = drv.variables.new()
            var.type = 'SINGLE_PROP'
            var.targets[0].id = a
            var.targets[0].data_path = '["%s"]' % name
        found[which] = {'points': int(mark.sum()), 'middle': [round(float(v), 3) for v in c], 'half_length': round(float(half), 3)}
    return found


def brows(left=0.0, right=0.0, tilt_left=0.0, tilt_right=0.0):
    """Set his brows: each up (1) or down (-1), and each tilted, inner end down (1) or up (-1)."""
    a = arm()
    for prop, value in zip(BROWS, (left, right, tilt_left, tilt_right)):
        if prop in a:
            a[prop] = float(value)
    a.update_tag()


def eyeballs():
    """Round his eyeballs out. Meshy models his eyes as paint on a nearly flat patch, so side on they vanish.

    A shape ('eyeball', left full on) domes each eye out along the way it faces, highest in the middle and back to
    nothing at its rim. The paint and the eyelids ride on it. Needs eyes() to have run."""
    ob, me = skin(), skin().data
    n = len(me.vertices)
    co = np.zeros(n * 3, 'f4')
    me.shape_keys.key_blocks['Basis'].data.foreach_get('co', co)
    co = co.reshape(-1, 3).astype('f8')
    no = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('normal', no)
    no = no.reshape(-1, 3).astype('f8')

    def spot(name):
        v = np.zeros(n, 'f4')
        me.attributes[name].data.foreach_get('value', v)
        return v.astype('f8')

    height, wide, far, side = spot('j5_eye'), spot('j5_eyea'), spot('j5_eyem'), spot('j5_side')
    out = np.sqrt(wide + (2 * height - 1) ** 2)             # 0 in the middle of an eye, 1 at its edge
    by = np.zeros((n, 3))
    for mine in (side > 0.5, side < 0.5):
        eye = (far < 2.0) & mine
        facing_ = no[eye & (out < 0.6)].mean(0)
        facing_ /= np.linalg.norm(facing_)
        dome = np.where(eye & (out < 1.0), (1.0 - np.minimum(out, 1.0) ** 2.2) ** 0.9, 0.0)
        by += np.outer(BULGE * dome, facing_)
    old = me.shape_keys.key_blocks.get('eyeball')
    if old:
        ob.shape_key_remove(old)
    key = ob.shape_key_add(name='eyeball', from_mix=False)
    key.data.foreach_set('co', (co + by).astype('f4').ravel())
    key.value = 1.0
    return {'points': int((np.abs(by).sum(1) > 1e-4).sum())}


def spread(n, edges, inside, values):
    """Fill the points marked `inside` with the smoothest values that join the values of the points round them, like
    a sheet stretched over a hole (each ends as the mean of the points it is joined to). `edges` are his skin's
    edges between places (welded()); `values` is one row a point. Solved by conjugate gradients, a column at a time."""
    place = np.where(inside)[0]
    m = len(place)
    at = np.full(n, -1)
    at[place] = np.arange(m)
    u, v = edges[:, 0], edges[:, 1]
    both = inside[u] & inside[v]
    iu, iv = at[u[both]], at[v[both]]
    links = np.bincount(at[u[inside[u]]], minlength=m) + np.bincount(at[v[inside[v]]], minlength=m)
    fixed = np.zeros((m, values.shape[1]))                 # what is round the hole stays as it is
    for near_, off_ in ((u, v), (v, u)):
        edge = inside[near_] & ~inside[off_]
        np.add.at(fixed, at[near_[edge]], values[off_[edge]])

    def over(x):                                           # each point less the mean of the points it is joined to, times its joins
        return links * x - (np.bincount(iu, x[iv], minlength=m) + np.bincount(iv, x[iu], minlength=m))

    out = values.copy()
    for c in range(values.shape[1]):
        x = values[place, c].copy()
        r = fixed[:, c] - over(x)
        d_, rr = r.copy(), float(r @ r)
        for _ in range(6000):
            if rr < 1e-18:
                break
            Ad = over(d_)
            step = rr / float(d_ @ Ad)
            x += step * d_
            r -= step * Ad
            rr, was = float(r @ r), rr
            d_ = r + (rr / was) * d_
        out[place, c] = x
    return out


def shaped(name, base, move):
    """A shape on his skin, by name: his points as modelled (`base`) moved by `move`. One of that name already there is replaced."""
    ob = skin()
    old = ob.data.shape_keys.key_blocks.get(name)
    if old:
        old.driver_remove('value')
        ob.shape_key_remove(old)
    key = ob.shape_key_add(name=name, from_mix=False)
    key.data.foreach_set('co', (base + move).astype('f4').ravel())
    return key


def driven(key, expression, props):
    """Have a shape follow numbers on his bones: `props` is {letter: number's name}, and `expression` a sum in those letters."""
    a = arm()
    key.driver_remove('value')
    drv = key.driver_add('value').driver
    drv.type = 'SCRIPTED'
    for letter, prop in props.items():
        if prop not in a:
            a[prop] = CHEEK_AS if prop == CHEEK else 0.0
            a.id_properties_ui(prop).update(min=0.0, max=1.0)
        var = drv.variables.new()
        var.name = letter
        var.type = 'SINGLE_PROP'
        var.targets[0].id = a
        var.targets[0].data_path = '["%s"]' % prop
    drv.expression = expression


def lids():
    """Make a shut eye a lid over an eyeball, and have his face shut with it.

    His eyes are paint on his skin, with a rim modelled round each one. Painting fur over an eye left that rim
    standing: a ring round every shut eye. For each eye (l: his left, r: his right) these shapes are made, each
    following the numbers the paint follows ('blink', 'blink_l', 'blink_r', 'joy', 'frown'):
      lid_    the skin of the eye and its rim, and no further, is laid afresh as the smoothest skin that joins the
              face round it, then domed out as far as the eyeball under it and a lid's thickness more. The skin
              above, brow and all, comes down a hair with it.
      fold_   a shallow crease where the two lids meet, under the painted lash line, so that the line is a fold in
              his skin from the side and not only paint. arch_ is the same for the happy arch.
      cheek_  the cheek under the eye lifts and pushes out, bunching under the lower lid, as a face does when an eye
              shuts. How far is a number on his bones ('cheek'): a little for a blink, hardly at all for sleepy
              lids; joy lifts it all the way, and a frown squeezes it up under a narrowed eye.
    They are worked out on his eyes as domed by eyeballs(), so run after it."""
    a, ob, me = arm(), skin(), skin().data
    n = len(me.vertices)
    keys = me.shape_keys.key_blocks
    base = np.zeros(n * 3, 'f4')
    keys['Basis'].data.foreach_get('co', base)
    base = base.reshape(-1, 3).astype('f8')
    now = base.copy()
    if 'eyeball' in keys:
        ball = np.zeros(n * 3, 'f4')
        keys['eyeball'].data.foreach_get('co', ball)
        now = ball.reshape(-1, 3).astype('f8')
    no = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('normal', no)
    no = no.reshape(-1, 3).astype('f8')

    def spot(name):
        v = np.zeros(n, 'f4')
        me.attributes[name].data.foreach_get('value', v)
        return v.astype('f8')

    far, side, high_, wide, lean = spot('j5_eyem'), spot('j5_side'), spot('j5_eye'), spot('j5_eyea'), spot('j5_eyeo')
    one, edges = welded(base, me)
    whole, out_, swell = LID['whole'], LID['out'], LID['swell']
    arch_, low, dip, high, bow = EYE[1], EYE[2], EYE[3], EYE[5], EYE[6]
    kit = headset_paint(base, me, 2)                       # his headset and its microphone, which runs close by his right eye, stay as they are
    kit[one[kit]] = True
    dark = linear(paint_at_points().astype('f8')).max(1) < 0.02      # paint as dark as his brows
    from mathutils import kdtree
    teal = np.where(headset_paint(base, me, 0))[0]
    kit_at = kdtree.KDTree(len(teal))                       # for how far a point is from his headset
    for k, i in enumerate(teal):
        kit_at.insert(base[i], k)
    kit_at.balance()
    smooth = lambda t: np.clip(t, 0.0, 1.0) ** 2 * (3 - 2 * np.clip(t, 0.0, 1.0))
    shut = 'max(b, s)'
    made = {}
    for tag_, prop, mine in (('l', LIDS[1], side > 0.5), ('r', LIDS[2], side < 0.5)):
        mine = mine & (far < 2.2)
        # ---- the lid
        # Below and beside the eye only the eye and its rim are laid afresh. Above it the new skin reaches up to
        # his brow: Meshy modelled a hollow there, which is in shadow, and a dark dome over a shut eye made no sense.
        upper = smooth((2 * high_ - 1) / 0.6 + 0.5)        # 0 below the eye's middle, 1 above it
        whole_ = whole + (LID['above'][0] - whole) * upper
        out__ = out_ + (LID['above'][1] - out_) * upper
        # where this eye's points are in its own measure, read back from them: U is 0 at the eye's middle, 1 at its
        # top, -1 at its bottom; X is 0 at its middle, 1 at its outer corner
        fit = mine & (far < 1.5)
        rz, cz = np.linalg.lstsq(np.c_[2 * high_[fit] - 1, np.ones(fit.sum())], now[fit, 2], rcond=None)[0]
        rx, cx = np.linalg.lstsq(np.c_[lean[fit], np.ones(fit.sum())], now[fit, 0], rcond=None)[0]
        U, X = (now[:, 2] - cz) / rz, (now[:, 0] - cx) / rx
        brow_ = dark & (U > 1.4) & (U < 3.4) & (np.abs(X) < 2.4)   # his painted brow, and three rings of skin round it, stay as they are: laying the skin under a brow afresh pulled it out of shape
        brow_[one[brow_]] = True
        for _ in range(3):
            grown = brow_.copy()
            grown[edges[brow_[edges[:, 1]], 0]] = True
            grown[edges[brow_[edges[:, 0]], 1]] = True
            brow_ = grown
        brow_ = brow_[one]
        inside = np.zeros(n, bool)
        inside[one[mine & (far < out__)]] = True
        inside[kit | brow_] = False
        m = int(inside.sum())
        new = spread(n, edges, inside, now)[one]
        facing_ = no[mine & (far < 0.6)].mean(0)
        facing_ /= np.linalg.norm(facing_)
        fade = (1.0 - smooth((far - whole_) / (out__ - whole_))) * mine * ~kit * ~brow_
        bump = np.where(mine & (far < 1.0) & ~kit, (1.0 - np.minimum(far, 1.0) ** 2) ** 1.5, 0.0)
        front = smooth(((now - now[mine & (far < 0.5)].mean(0)) @ facing_ + 0.30) / 0.12)   # the front of his head, by how far back a point lies: going by which way each point faces moved his grooved fur in stripes
        # The cheek lift keeps to his cheek. It used to reach in to his nose, down to his mouth and on to his
        # microphone, and pulled all three out of shape: plain in a wink, where only one side goes.
        span = np.exp(-(X / 1.5) ** 2) * smooth((2.6 - np.abs(X)) / 0.8) * smooth((X + 1.15) / 0.5) * front   # across the eye and a way past it; none by his nose
        nearby = np.where(span > 0.01)[0]
        clear = np.ones(n)
        clear[nearby] = [kit_at.find(base[i])[2] for i in nearby]
        span = span * smooth((clear - 0.012) / 0.05)        # easing off toward his headset and its microphone, which stay still
        move = (new - now) * fade[:, None] + np.outer(swell * bump, facing_)
        driven(shaped('lid_' + tag_, base, move), 'smoothstep(0.45, 0.95, %s)' % shut, {'b': LIDS[0], 's': prop})
        brow = me.shape_keys.key_blocks.get('brow_' + tag_)  # his brow comes down a little with the lid, on its own shape (browed()); a happy arch lifts instead
        if brow:
            driven(brow, 'v - %g * smoothstep(0.45, 0.95, %s) * (1 - j)' % (LID['dip'], shut), {'v': 'brow_' + tag_, 'b': LIDS[0], 's': prop, 'j': JOY})
        made['lid_' + tag_] = {'points': m, 'most': round(float(np.linalg.norm(move, axis=1).max()), 4)}
        # ---- the crease where the lids meet, under the lash line: for a sleepy dip, and for a happy arch
        width, depth = LID['fold']
        on_lid = smooth((1.0 - far) / 0.12) * smooth((0.95 - wide) / 0.3) * mine * ~kit
        for name, line, expression in (('fold_', high_ - dip * wide - low, 'smoothstep(0.75, 1.0, %s) * (1 - j)' % shut),
                                       ('arch_', high_ + bow * wide - high, 'smoothstep(0.75, 1.0, %s) * j' % shut)):
            groove = np.exp(-(line / width) ** 2) * on_lid
            driven(shaped(name + tag_, base, np.outer(-depth * groove, facing_)), expression, {'b': LIDS[0], 's': prop, 'j': JOY})
        # ---- the cheek
        lift, push = LID['cheek']
        under = smooth((-0.35 - U) / 0.8) * smooth((U + 2.3) / 1.2) * span       # from just under the lash line down to the top of his whisker pad
        fuller = smooth((-0.6 - U) / 0.8) * smooth((U + 2.3) / 1.0) * span
        move = np.outer(push * fuller, facing_)
        move[:, 2] += lift * under
        driven(shaped('cheek_' + tag_, base, move),
               'min(1, smoothstep(0.3, 1.0, %s) * min(1, g + max(j, 0.7 * f)) + 0.45 * f * smoothstep(0.15, 0.5, %s))' % (shut, shut),
               {'b': LIDS[0], 's': prop, 'j': JOY, 'f': FROWN, 'g': CHEEK})
        made['cheek_' + tag_] = {'points': int((under > 0.05).sum()), 'eye': [round(float(v), 3) for v in (cx, cz, rx, rz)]}
    return made


def blink(amount=1.0, left=0.0, right=0.0, joy=0.0, frown=0.0):
    """0 = eyes open, 1 = shut. `left` and `right` shut one of his eyes on its own, for a wink.

    joy=1 makes a shut eye a happy arch instead of a sleepy dip. frown=1 slants the lids down toward his nose, so
    half-shut eyes look narrowed and cross."""
    a = arm()
    a['blink'], a['blink_l'], a['blink_r'], a[JOY], a[FROWN] = float(amount), float(left), float(right), float(joy), float(frown)
    a.update_tag()


def gold_eyes(co, paint, mid):
    """The middles of his two eyes (his right, his left), from the yellow in them."""
    hi, lo = paint.max(1), paint.min(1)
    d = np.maximum(hi - lo, 1e-6)
    r, g, b = paint[:, 0], paint[:, 1], paint[:, 2]
    hue = np.where(hi == r, ((g - b) / d) % 6, np.where(hi == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    gold = (hue > EYEHUE[0]) & (hue < EYEHUE[1]) & ((hi - lo) / np.maximum(hi, 1e-6) > 0.3) & (hi > 0.6) & (co[:, 2] > FACE[2]) & (co[:, 1] < FACE[1])
    out = []
    for mine in (gold & (co[:, 0] < mid), gold & (co[:, 0] >= mid)):
        e = co[mine]
        e = e[np.linalg.norm(e - np.median(e, axis=0), axis=1) < 0.14]
        out.append((e.min(0) + e.max(0)) / 2)
    return out


def like(pick, M, inside, top):
    """Match his shut face to Meshy's own model of the same cat with its mouth closed.

    A jaw made for an open mouth never shuts into a neat face: the chin is too big and the lips fold. So once seal()
    has shut him, the fur round his mouth is pulled on to the closed model's surface (it is lined up by the eyes), and
    takes that model's paint. All of it rides on the 'shut' shape and the jaw's driver, like the rest."""
    from mathutils import bvhtree, kdtree
    ob, me = skin(), skin().data
    before = set(bpy.data.objects)
    had = set(bpy.data.actions)
    bpy.ops.import_scene.gltf(filepath=os.path.normpath(os.path.join(HERE, os.path.dirname(GLB), CLOSED)))
    new = [o for o in bpy.data.objects if o not in before]
    other = max((o for o in new if o.type == 'MESH'), key=lambda o: len(o.data.vertices))
    cn = len(other.data.vertices)
    cc = np.zeros(cn * 3, 'f4')
    other.data.vertices.foreach_get('co', cc)
    cc = cc.reshape(-1, 3).astype('f8')
    W = np.array(other.matrix_world)
    cc = cc @ W[:3, :3].T + W[:3, 3]
    cpaint = paint_at_points(other).astype('f8')
    polys = [tuple(p.vertices) for p in other.data.polygons]
    for o in new:
        bpy.data.objects.remove(o, do_unlink=True)
    for act in [act for act in bpy.data.actions if act not in had]:
        bpy.data.actions.remove(act)
    # his own points, at rest
    n = len(me.vertices)
    co = np.zeros(n * 3, 'f4')
    me.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3).astype('f8')
    paint = paint_at_points().astype('f8')
    # line the closed model up: first by its size, then by its eyes
    cc = (cc - (cc.min(0) + cc.max(0)) / 2) * ((co[:, 2].max() - co[:, 2].min()) / (cc[:, 2].max() - cc[:, 2].min())) + (co.min(0) + co.max(0)) / 2
    mid = float(arm().data.bones['head'].head_local.x)
    mine, theirs = gold_eyes(co, paint, mid), gold_eyes(cc, cpaint, float((cc[:, 0].min() + cc[:, 0].max()) / 2))
    scale = np.linalg.norm(mine[1] - mine[0]) / np.linalg.norm(theirs[1] - theirs[0])
    cc = (cc - (theirs[0] + theirs[1]) / 2) * scale + (mine[0] + mine[1]) / 2
    tree = bvhtree.BVHTree.FromPolygons([tuple(v) for v in cc.tolist()], polys)
    near = kdtree.KDTree(cn)
    for i, v in enumerate(cc.tolist()):
        near.insert(v, i)
    near.balance()
    # him, shut, as seal() has left him
    rest()
    jaw(1.0)
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    ev = ob.evaluated_get(dg)
    posed = ev.to_mesh()
    shut = np.zeros(n * 3, 'f4')
    posed.vertices.foreach_get('co', shut)
    ev.to_mesh_clear()
    shut = shut.reshape(-1, 3).astype('f8')[pick]
    hi, lo = paint[pick].max(1), paint[pick].min(1)
    d = np.maximum(hi - lo, 1e-6)
    r, g, b = paint[pick][:, 0], paint[pick][:, 1], paint[pick][:, 2]
    hue = np.where(hi == r, ((g - b) / d) % 6, np.where(hi == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    fur = ((hue > 0.6) | (hue < 0.05) | ((hi - lo) / np.maximum(hi, 1e-6) < 0.3)) & ~inside       # not his headset's mic, not the inside of his mouth
    lip = float(shut[top][:, 2].mean())
    mouth = np.array((MOUTH[0], float(shut[top][:, 1].min()), lip))
    far = np.linalg.norm((shut - mouth) * np.array((1.0, 0.6, 1.0)), axis=1)
    w = (1.0 - np.array([ease((f - LIKE[0]) / (LIKE[1] - LIKE[0])) for f in far])) * fur
    w *= 1.0 - np.array([ease((z - (lip + LIKE[2])) / 0.04) for z in shut[:, 2]])              # leave his nose alone
    move = np.zeros_like(shut)
    colour = np.zeros((n, 4), 'f4')
    weight = np.zeros(n, 'f4')
    for k in np.where(w > 0.001)[0]:
        hit = tree.find_nearest(shut[k].tolist())
        if hit[0] is None:
            continue
        move[k] = (np.array(hit[0]) - shut[k]) * w[k]
        colour[pick[k], :3] = linear(cpaint[near.find(tuple(hit[0]))[1]])
        colour[pick[k], 3] = 1.0
        weight[pick[k]] = w[k]
    nb = mouth5.neighbours(shut, 0.012)
    for _ in range(2):                                     # even the pull out, so nothing creases
        move = np.array([(move[i] * 2 + move[j].sum(0)) / (2 + len(j)) if w[i] > 0.001 else move[i] for i, j in enumerate(nb)])
    back = np.linalg.solve(M[:, :3, :3], move[:, :, None])[:, :, 0]
    key = me.shape_keys.key_blocks['shut']
    data = np.zeros(n * 3, 'f4')
    key.data.foreach_get('co', data)
    data = data.reshape(-1, 3)
    data[pick] += back.astype('f4')
    key.data.foreach_set('co', data.ravel())
    for name, kind, values in (('j5_like', 'FLOAT_COLOR', colour.ravel()), ('j5_likew', 'FLOAT', weight)):
        old = me.attributes.get(name)
        if old:
            me.attributes.remove(old)
        att = me.attributes.new(name, kind, 'POINT')
        att.data.foreach_set('color' if kind == 'FLOAT_COLOR' else 'value', values)
    for mat in me.materials:                               # his paint turns to the closed model's as he shuts
        nt = mat.node_tree
        bsdf = next(k for k in nt.nodes if k.type == 'BSDF_PRINCIPLED')
        for k in [k for k in nt.nodes if k.name.startswith('jl_')]:
            nt.nodes.remove(k)
        source = nt.nodes['j5_mix2'].outputs[2]
        mix = nt.nodes.new('ShaderNodeMix')
        mix.name, mix.data_type = 'jl_mix', 'RGBA'
        where = nt.nodes.new('ShaderNodeAttribute')
        where.name, where.attribute_name = 'jl_where', 'j5_likew'
        what = nt.nodes.new('ShaderNodeAttribute')
        what.name, what.attribute_name = 'jl_what', 'j5_like'
        times = nt.nodes.new('ShaderNodeMath')
        times.name, times.operation = 'jl_times', 'MULTIPLY'
        nt.links.new(where.outputs['Fac'], times.inputs[0])
        nt.links.new(nt.nodes['j5_shut'].outputs[0], times.inputs[1])
        nt.links.new(times.outputs[0], mix.inputs[0])
        nt.links.new(source, mix.inputs[6])
        nt.links.new(what.outputs['Color'], mix.inputs[7])
        nt.links.new(mix.outputs[2], bsdf.inputs['Base Color'])
    rest()
    return {'points': int((w > 0.001).sum()), 'eyes_scale': round(float(scale), 3)}


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
        on_edge = np.clip(1.0 - found['lipdist'] / THICK, 0.0, 1.0) ** 0.5
    else:
        on_edge = np.where(found['top'], EDGE[0], np.where(found['bottom'], EDGE[1], 0.0))
    line[pick] = middle * np.where(found['inside'], 1.0, on_edge)
    cover[pick] = (1.0 - middle) * np.where(found['inside'], 1.0, on_lips)
    ring = found['fur'] & (found['lipdist'] > 0.03) & (found['lipdist'] < 0.06)      # the fur just outside his lips
    fur = np.where(paint[ring] <= 0.04045, paint[ring] / 12.92, ((paint[ring] + 0.055) / 1.055) ** 2.4).mean(0)
    mouth_line(line, cover, (float(fur[0]), float(fur[1]), float(fur[2]), 1.0))
    if CLOSED:
        like(pick, M, found['inside'], found['top'])
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


def lit(face=0.0):
    """Light him. With LIGHT, by a main lamp to one side of the way his face points (`face`, as facing() gives it)
    and a softer one from the other side, so that both sides of his face are lit alike. One lamp fixed in the room
    put the brow's shadow over one eye and not the other: that eye, and its lid, looked darker."""
    sc = bpy.context.scene
    sun, fill = bpy.data.objects.get('JimmySun'), bpy.data.objects.get('JimmyFill')
    if sun is None:
        return
    if fill is None:
        fill = bpy.data.objects.new('JimmyFill', bpy.data.lights.new('JimmyFill', 'SUN'))
        sc.collection.objects.link(fill)
    fill.data.angle = math.radians(40)
    if not LIGHT:
        sun.rotation_euler, sun.data.energy, fill.data.energy = (math.radians(55), 0, math.radians(-35)), 2.6, 0.0
        return
    for lamp, (lean, round_, bright) in zip((sun, fill), LIGHT):
        lamp.rotation_euler = (math.radians(lean), 0, math.radians(-(face + round_)))
        lamp.data.energy = bright


def shot(path, face=False, yaw=0.0, pitch=0.0, size=600, front=False, centre=None, light=True):
    """Render him as he is posed now: face=True for a close-up of his face, else all of him.

    front=True measures yaw from straight in front of his face instead of from the room. Returns (centre, yaw) used,
    so that the frames of a clip can all be taken from one place. light=False leaves the lamps where they are."""
    bpy.context.view_layer.update()
    if light:
        lit(facing())
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
    lit(facing())                                          # the lamps are set once, on that pose, and do not follow his head about
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
        shot(path, face=face, yaw=turn_, size=size, centre=centre, light=False)
        out.append(path)
    rest()
    return out


def paint_at_points(ob=None):
    """The colour of Meshy's paint at each point of his skin (0 to 1, as stored), for finding teeth and tongue."""
    me = (ob or skin()).data
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
