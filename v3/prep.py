"""Jimmy from Meshy (JM-491): unpack Meshy's model and bring its fur to the colours of Ant's drawing.

Writes model.npz (points, where each falls on the paint, and the triangles; in Blender's axes, Z up, his face to -Y)
and meshy-fur.png (the paint, recoloured). Plain Python (PIL, numpy).

Meshy's repaint has the brush strokes we want but its own browns. Every fur pixel keeps its place in Meshy's range
from dark to light and takes the colour the drawing's fur has at that same place in its own range (a gradient map),
so the strokes stay and the palette becomes Ant's. Pixels that aren't fur (headset, collar, tag, ears, eyes) are left.

    python3 prep.py <model.glb> <version 2 front view, with alpha> <out folder>
"""
import io, json, struct, sys
import numpy as np
from PIL import Image

def glb_images(path):
    f = open(path, 'rb'); f.read(12)
    n, _ = struct.unpack('<I4s', f.read(8)); j = json.loads(f.read(n))
    n, _ = struct.unpack('<I4s', f.read(8)); blob = f.read(n)
    def view(i):
        v = j['bufferViews'][i]
        return blob[v.get('byteOffset', 0):v.get('byteOffset', 0) + v['byteLength']]
    def acc(i):
        a = j['accessors'][i]
        n = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3}[a['type']]
        dt = {5123: '<u2', 5125: '<u4', 5126: '<f4'}[a['componentType']]
        v = j['bufferViews'][a['bufferView']]
        assert not v.get('byteStride')
        start = v.get('byteOffset', 0) + a.get('byteOffset', 0)
        return np.frombuffer(blob, dt, a['count'] * n, start).reshape(-1, n)
    prim = j['meshes'][0]['primitives'][0]
    assert len(j['meshes']) == 1 and len(j['meshes'][0]['primitives']) == 1 and prim.get('mode', 4) == 4
    co, uv, tri = acc(prim['attributes']['POSITION']), acc(prim['attributes']['TEXCOORD_0']), acc(prim['indices']).reshape(-1, 3)
    base = j['textures'][j['materials'][prim['material']]['pbrMetallicRoughness']['baseColorTexture']['index']]['source']
    paint = Image.open(io.BytesIO(view(j['images'][base]['bufferView'])))
    geo = dict(co=np.stack([co[:, 0], -co[:, 2], co[:, 1]], 1).astype('f4'), uv=np.stack([uv[:, 0], 1 - uv[:, 1]], 1).astype('f4'), tri=tri.astype('i4'))
    return geo, paint

def hsv(a):
    r, g, b = a[..., 0] / 255.0, a[..., 1] / 255.0, a[..., 2] / 255.0
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b); d = mx - mn
    h = np.zeros_like(mx); nz = d > 1e-6
    h = np.where(nz & (mx == r), ((g - b) / np.where(nz, d, 1)) % 6, h)
    h = np.where(nz & (mx == g), (b - r) / np.where(nz, d, 1) + 2, h)
    h = np.where(nz & (mx == b), (r - g) / np.where(nz, d, 1) + 4, h)
    return h * 60, np.where(mx > 0, d / np.where(mx > 0, mx, 1), 0), mx

def main(glb, front, out):
    out = out.rstrip('/') + '/'
    geo, paint = glb_images(glb)
    np.savez_compressed(out + 'model.npz', **geo)
    print('points', len(geo['co']), 'triangles', len(geo['tri']), 'paint', paint.size, 'box', geo['co'].min(0).round(3), geo['co'].max(0).round(3))
    out = out + 'meshy-fur.png'
    tex = np.asarray(paint.convert('RGB')).astype(float)
    h, s, v = hsv(tex)
    fur = ((h < 50) | (h > 250)) & (v < 0.97)                                      # browns, creams, and the plum of his ear backs
    fur |= (v < 0.16)                                                              # Meshy's darkest lines
    fur &= ~((s > 0.45) & (v > 0.5))                                               # not the pink of his ears or the amber of his eyes
    lum = tex @ np.array([0.299, 0.587, 0.114])
    art = np.asarray(Image.open(front).convert('RGBA')).astype(float)
    ah, as_, av = hsv(art[..., :3])
    r, g, b = art[..., 0], art[..., 1], art[..., 2]
    afur = (art[..., 3] > 250) & (r < 170) & (g < 120) & (b < 125) & (r >= b - 25) & (r - g < 70)   # the drawing's fur and its lines; not eyes, ears, collar, tag, headset
    alum = art[..., :3] @ np.array([0.299, 0.587, 0.114])
    order = np.argsort(alum[afur]); cols = art[afur][:, :3][order]
    Q = 256
    ramp = np.stack([cols[int(i * (len(cols) - 1) / (Q - 1)) - 0:][:max(1, len(cols) // Q)].mean(0) for i in range(Q)])
    for c in range(3):                                                             # keep the ramp smooth
        ramp[:, c] = np.convolve(np.pad(ramp[:, c], 8, mode='edge'), np.ones(17) / 17, mode='valid')
    ml = lum[fur]; qs = np.quantile(ml, np.linspace(0, 1, Q))
    q = np.interp(lum, qs, np.linspace(0, Q - 1, Q))
    mapped = np.stack([np.interp(q, np.arange(Q), ramp[:, c]) for c in range(3)], -1)
    res = np.where(fur[..., None], mapped, tex)
    Image.fromarray(np.clip(res, 0, 255).astype('uint8')).save(out)
    print('fur share of the picture: %.2f' % fur.mean(), 'Meshy fur median', np.median(tex[fur], 0).round(), 'becomes', np.median(mapped[fur], 0).round(), 'ramp ends', ramp[0].round(), ramp[-1].round())
    Image.fromarray(np.clip(np.concatenate([tex, res], 1), 0, 255).astype('uint8')).resize((1024, 512)).save(out.replace('.png', '-compare.jpg'))

main(*sys.argv[1:4])
