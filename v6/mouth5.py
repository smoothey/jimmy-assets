"""Works out how to seal 3D Jimmy's mouth when his jaw is shut (version 5).

Plain numpy, no Blender, so it can be tested away from Blender. jimmy5.py feeds it the points
round his mouth and turns the answer into a shape key called 'shut'.

His Meshy model was made with his mouth open, so there is a real mouth inside. Turning the jaw
bone alone leaves a gap with teeth and red showing. This lifts his bottom lip to sit just behind
his top lip, and tucks his teeth, tongue and the inside of his mouth away behind it.
"""
import colorsys
import numpy as np

NEAR = 0.008      # points this close count as neighbours
TUCK = (0.010, 0.007)        # bottom lip ends this far inside and above the edge of the top lip
CHIN = 0.085      # how far down the chin the lift fades out
BIN = 0.008       # width of the slices across his mouth
TIDY = 3          # rounds of smoothing on his lifted bottom lip
WHITE = (0.25, 0.6)          # teeth: paint with less colour than this, and brighter than this
FANGS = False                # also look for teeth cut off from the rest of the inside of his mouth
SAFE = (0.06, 0.07)          # a spot deep in his head, this far behind and above the middle of his mouth...
PACK = 0.3                   # ...where the inside of his mouth is packed away, shrunk to this much of its size


def kinds(colour):
    """True where a point is fur; the rest is teeth, tongue, gums or other bright paint."""
    hsv = np.array([colorsys.rgb_to_hsv(*c) for c in colour])
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    white = (s < WHITE[0]) & (v > WHITE[1])
    red = (s > 0.45) & (v > 0.25) & ((h < 0.08) | (h > 0.9))
    pink = (s > 0.2) & (v > 0.55) & ~white & ~red
    return ~(white | red | pink), red


def neighbours(points, near=NEAR):
    """For each point, the points within `near` of it."""
    cells = {}
    keys = np.floor(points / near).astype(int)
    for i, k in enumerate(map(tuple, keys)):
        cells.setdefault(k, []).append(i)
    out = []
    for i, k in enumerate(map(tuple, keys)):
        cand = [j for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
                for j in cells.get((k[0] + a, k[1] + b, k[2] + c), ())]
        cand = np.array(cand)
        d = np.linalg.norm(points[cand] - points[i], axis=1)
        out.append(cand[(d < near) & (cand != i)])
    return out


def parts(rest, colour, jaw):
    """Find the inside of his mouth and the two lip edges."""
    fur, red = kinds(colour)
    nb = neighbours(rest)
    inside = np.zeros(len(rest), bool)
    todo = list(np.where(red & (jaw > 0.8))[0])        # his tongue: red and fully on the jaw
    inside[todo] = True
    while todo:
        i = todo.pop()
        for j in nb[i]:
            if not inside[j] and not fur[j]:
                inside[j] = True
                todo.append(j)
    for _ in range(3):                                   # dark paint in the middle of his mouth is inside too
        for i in np.where(~inside)[0]:
            if len(nb[i]) and inside[nb[i]].mean() > 0.6:
                inside[i] = True
    # dark patches between his teeth are inside too: anything that can't be reached from his face without crossing the inside
    middle = rest[inside].mean(0)
    todo = list(np.where(~inside & (np.linalg.norm(rest - middle, axis=1) > 0.16))[0])
    outside = np.zeros(len(rest), bool)
    outside[todo] = True
    while todo:
        i = todo.pop()
        for j in nb[i]:
            if not outside[j] and not inside[j]:
                outside[j] = True
                todo.append(j)
    lo, hi = rest[inside].min(0), rest[inside].max(0)
    inside = inside | (~outside & ((rest >= lo) & (rest <= hi)).all(1) & np.array([len(n) > 0 for n in nb]))
    edge = np.array([not inside[i] and inside[nb[i]].any() for i in range(len(rest))])
    if FANGS:
        # Fangs that hang in front of the dark of his mouth are cut off from the rest of the inside by paint too dark
        # to tell from fur. Any tooth-white point in the opening between his lips is inside too.
        e = rest[edge]
        white = ~fur & ~red & (np.array([colorsys.rgb_to_hsv(*c)[1] for c in colour]) < WHITE[0])
        for i in np.where(white & ~inside)[0]:
            near = e[np.abs(e[:, 0] - rest[i, 0]) < BIN * 2]
            if len(near) and near[:, 2].min() - 0.01 < rest[i, 2] < near[:, 2].max() + 0.004 and rest[i, 1] < e[:, 1].min() + 0.09:
                inside[i] = True
        todo = list(np.where(white & inside)[0])
        while todo:                                      # and the rest of each such tooth
            i = todo.pop()
            for j in nb[i]:
                if not inside[j] and white[j]:
                    inside[j] = True
                    todo.append(j)
        edge = np.array([not inside[i] and inside[nb[i]].any() for i in range(len(rest))])
    # top lip or bottom lip: above or below the middle of the open mouth at that place across it.
    # (His jaw's pull fades out toward the corners of his mouth, so the jaw alone can't say.)
    e = rest[edge]
    xs = np.arange(e[:, 0].min(), e[:, 0].max() + BIN, BIN)
    mids = []
    for x in xs:
        m = np.abs(e[:, 0] - x) < BIN * 1.5
        mids.append((e[m, 2].max() + e[m, 2].min()) / 2 if m.any() else np.nan)
    mids = np.array(mids)
    ok = ~np.isnan(mids)
    below = rest[:, 2] < np.interp(rest[:, 0], xs[ok], mids[ok])
    lips = rest[edge]
    lipdist = nearest(rest, lips)[1]                     # how far each point is from the lips, mouth open
    mid, half = (lips[:, 0].max() + lips[:, 0].min()) / 2, (lips[:, 0].max() - lips[:, 0].min()) / 2
    across = np.abs(rest[:, 0] - mid) / half             # 0 in the middle of his mouth, 1 at its corners
    return {'inside': inside, 'top': edge & ~below, 'bottom': edge & below, 'below': below, 'nb': nb, 'fur': ~inside,
            'lipdist': lipdist, 'across': across}


def nearest(points, to):
    """For each of `points`, which of `to` is nearest and how far away it is."""
    which, far = np.zeros(len(points), int), np.zeros(len(points))
    for i, q in enumerate(points):
        d = np.linalg.norm(to - q, axis=1)
        which[i] = d.argmin()
        far[i] = d[which[i]]
    return which, far


def smaller(shut, jaw, p, small):
    """Moves that shrink his lower jaw once it is shut: narrower, tucked back, and shorter from lip to chin.

    He was modelled with a big open smile, so the jaw that swings shut is too big for a shut face."""
    top = shut[p['top']]
    pivot = np.array(((top[:, 0].min() + top[:, 0].max()) / 2, shut[p['inside']][:, 1].mean(), top[:, 2].mean()))
    w = jaw * jaw * (3 - 2 * jaw)
    w = w * np.clip(1.0 - (shut[:, 2] - (pivot[2] + 0.01)) / 0.03, 0.0, 1.0)     # nothing above his top lip moves
    d = shut - pivot
    new = d.copy()
    new[:, 0] *= small[0]
    new[:, 1] = np.where(d[:, 1] < 0, d[:, 1] * small[1], d[:, 1])               # only what is in front comes back
    new[:, 2] = np.where(d[:, 2] < 0, d[:, 2] * small[2], d[:, 2])               # only what is below comes up
    return w[:, None] * (new - d)


def seal(rest, shut, colour, jaw, smile=0.0, small=(1.0, 1.0, 1.0)):
    """How far each point should move, in the shut pose, to seal his mouth. Returns (moves, parts)."""
    p = parts(rest, colour, jaw)
    ins, below = p['inside'], p['below']
    shrink = smaller(shut, jaw, p, small)
    shut = shut + shrink
    T, B = shut[p['top']], shut[p['bottom']]
    centre = shut[ins].mean(0)
    # each point on his bottom lip goes to the nearest point on his top lip, then a touch further in and up
    j, _ = nearest(B, T)
    inward = centre - T[j]
    inward[:, 2] = 0.0
    inward /= np.linalg.norm(inward, axis=1)[:, None]
    lip = T[j] + inward * TUCK[0] + np.array((0.0, 0.0, TUCK[1])) - B
    lip = np.array([(lip * w[:, None]).sum(0) / w.sum()                 # evened out along the lip
                    for w in (np.exp(-(np.linalg.norm(B - q, axis=1) / 0.012) ** 2) for q in B)])
    k, far = nearest(shut, B)
    fade = np.clip(1.0 - far / CHIN, 0.0, 1.0)
    fade = fade * fade * (3 - 2 * fade)
    moves = np.zeros_like(shut)
    fur_low = p['fur'] & below
    moves[fur_low] = (lip[k] * fade[:, None])[fur_low]
    # the inside of his mouth: right at his bottom lip it follows the lip; the rest (roof, tongue, teeth) is packed
    # away small, deep in his head, where no change to his jaw or chin can make it poke out.
    # How deep a point is, is measured with his mouth open, so a top fang that reaches his bottom lip still counts.
    _, deep_low = nearest(rest, rest[p['bottom']])
    low = jaw >= 0.5
    deep = np.where(low, np.clip(deep_low / 0.012, 0.0, 1.0), 1.0)    # the top half goes whole, so his bottom lip has room behind his top lip
    teeth = ~kinds(colour)[0] & ~kinds(colour)[1] & (np.array([colorsys.rgb_to_hsv(*c)[1] for c in colour]) < 0.25)
    deep[teeth] = 1.0
    follow = np.where(below, 1.0 - np.clip(far / 0.015, 0.0, 1.0), 0.0)
    safe = centre + np.array((0.0, SAFE[0], SAFE[1]))
    pack = (PACK - 1.0) * (shut - safe)
    moves[ins] = (lip[k] * follow[:, None] + pack * deep[:, None])[ins]
    if smile:                                            # lift the corners of his mouth, both lips together
        mid, half = (B[:, 0].max() + B[:, 0].min()) / 2, (B[:, 0].max() - B[:, 0].min()) / 2
        corner = np.clip((np.abs(shut[:, 0] - mid) - half * 0.35) / (half * 0.65), 0.0, 1.0)
        corner = corner * corner * (3 - 2 * corner)
        close = np.clip(1.0 - np.minimum(far, nearest(shut, T)[1]) / 0.07, 0.0, 1.0)
        moves[:, 2] += smile * corner * close * close * (3 - 2 * close)
    for _ in range(2):                                   # even the moves out over his chin so nothing creases
        smooth = moves.copy()
        for i in np.where(fur_low)[0]:
            n = p['nb'][i][fur_low[p['nb'][i]]]
            smooth[i] = (moves[i] + moves[n].sum(0)) / (1 + len(n))
        moves = smooth
    lipzone = np.where(fur_low & (far < 0.03))[0]
    for _ in range(TIDY):                                # and smooth the lifted lip itself
        at = shut + moves
        for i in lipzone:
            n = p['nb'][i][fur_low[p['nb'][i]]]
            if len(n):
                moves[i] += 0.5 * (at[n].mean(0) - at[i])
    return moves + shrink, p
