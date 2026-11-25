"""Jimmy from Meshy (JM-491): make a Meshy .glb small enough to send to the laptop.

Keeps the model, its bones and its colour picture (as a JPEG); the two lighting pictures (normal, metal/rough) are
swapped for one-pixel blanks. Blender doesn't need them for the looks we use.

    python3 slim.py <in.glb> <out.glb>
"""
import io, json, struct, sys
from PIL import Image

def main(src, dst):
    f = open(src, 'rb'); f.read(12)
    n, _ = struct.unpack('<I4s', f.read(8)); j = json.loads(f.read(n))
    n, _ = struct.unpack('<I4s', f.read(8)); blob = f.read(n)
    base = {j['textures'][m['pbrMetallicRoughness']['baseColorTexture']['index']]['source'] for m in j['materials']}
    swap = {}
    for i, im in enumerate(j.get('images', [])):
        v = j['bufferViews'][im['bufferView']]
        data = blob[v.get('byteOffset', 0):v.get('byteOffset', 0) + v['byteLength']]
        out = io.BytesIO()
        if i in base:
            Image.open(io.BytesIO(data)).convert('RGB').save(out, 'JPEG', quality=93, subsampling=0)
        else:
            Image.new('RGB', (1, 1), (128, 128, 255)).save(out, 'JPEG')
        swap[im['bufferView']] = out.getvalue(); im['mimeType'] = 'image/jpeg'
    new = bytearray()
    for i, v in enumerate(j['bufferViews']):
        data = swap.get(i) or blob[v.get('byteOffset', 0):v.get('byteOffset', 0) + v['byteLength']]
        while len(new) % 4: new.append(0)
        v['byteOffset'], v['byteLength'] = len(new), len(data)
        new += data
    while len(new) % 4: new.append(0)
    j['buffers'] = [{'byteLength': len(new)}]
    js = json.dumps(j, separators=(',', ':')).encode()
    js += b' ' * (-len(js) % 4)
    with open(dst, 'wb') as o:
        o.write(struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(js) + 8 + len(new)))
        o.write(struct.pack('<I4s', len(js), b'JSON')); o.write(js)
        o.write(struct.pack('<I4s', len(new), b'BIN\0')); o.write(bytes(new))
    print('from', len(blob) // 1024, 'KB to', len(new) // 1024, 'KB')

main(sys.argv[1], sys.argv[2])
