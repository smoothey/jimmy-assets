"""3D Jimmy: materials in the drawing's style (JM-491).

The drawing is flat colour in soft bands with a warm light edge and a dark
plum line round everything. Here: a toon shader (light turned into three
bands), a rim on the lit side, and the line as a shell drawn inside out.
"""
import bpy

from j3d_common import PAL, rgb


def _new(name):
    m = bpy.data.materials.get('J_' + name)
    if m:
        bpy.data.materials.remove(m)
    m = bpy.data.materials.new('J_' + name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    return m, nt


def _out(nt, colour_socket):
    em = nt.nodes.new('ShaderNodeEmission')
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(colour_socket, em.inputs['Color'])
    nt.links.new(em.outputs[0], out.inputs['Surface'])


def toon(name, shadow, mid, light, rim=None, strokes=0.0, soft=0.05):
    """Three bands of colour from the light, an optional warm rim, optional fur strokes."""
    m, nt = _new(name)
    N, L = nt.nodes.new, nt.links.new
    diff = N('ShaderNodeBsdfDiffuse')
    diff.inputs['Color'].default_value = (1, 1, 1, 1)
    s2r = N('ShaderNodeShaderToRGB')
    L(diff.outputs[0], s2r.inputs[0])
    fac = s2r.outputs['Color']
    if strokes:
        tc = N('ShaderNodeTexCoord')
        mp = N('ShaderNodeMapping')
        mp.inputs['Scale'].default_value = (55, 55, 7)
        noise = N('ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = 1.0
        noise.inputs['Detail'].default_value = 1.5
        L(tc.outputs['Object'], mp.inputs['Vector'])
        L(mp.outputs[0], noise.inputs['Vector'])
        bw = N('ShaderNodeRGBToBW')
        L(fac, bw.inputs[0])
        ma = N('ShaderNodeMath')
        ma.operation = 'MULTIPLY_ADD'      # (noise - 0.5) * strokes + light
        sub = N('ShaderNodeMath')
        sub.operation = 'SUBTRACT'
        L(noise.outputs['Fac'], sub.inputs[0])
        sub.inputs[1].default_value = 0.5
        L(sub.outputs[0], ma.inputs[0])
        ma.inputs[1].default_value = strokes
        L(bw.outputs[0], ma.inputs[2])
        fac = ma.outputs[0]
    ramp = N('ShaderNodeValToRGB')
    cr = ramp.color_ramp
    cr.interpolation = 'LINEAR'
    stops = [(0.0, shadow), (0.30, shadow), (0.30 + soft, mid), (0.82, mid), (0.82 + soft, light), (1.0, light)]
    while len(cr.elements) < len(stops):
        cr.elements.new(0.5)
    for el, (pos, col) in zip(cr.elements, stops):
        el.position = pos
        el.color = rgb(col)
    L(fac, ramp.inputs['Fac'])
    colour = ramp.outputs['Color']
    if rim:
        lw = N('ShaderNodeLayerWeight')
        lw.inputs['Blend'].default_value = 0.32
        edge = N('ShaderNodeMapRange')
        edge.inputs['From Min'].default_value = 0.62
        edge.inputs['From Max'].default_value = 0.74
        L(lw.outputs['Facing'], edge.inputs['Value'])
        lit = N('ShaderNodeMapRange')
        lit.inputs['From Min'].default_value = 0.30
        lit.inputs['From Max'].default_value = 0.45
        L(fac, lit.inputs['Value'])
        both = N('ShaderNodeMath')
        both.operation = 'MULTIPLY'
        L(edge.outputs[0], both.inputs[0])
        L(lit.outputs[0], both.inputs[1])
        mix = N('ShaderNodeMix')
        mix.data_type = 'RGBA'
        L(both.outputs[0], mix.inputs['Factor'])
        L(colour, mix.inputs['A'])
        mix.inputs['B'].default_value = rgb(rim)
        colour = mix.outputs['Result']
    _out(nt, colour)
    m['base'] = mid
    m.diffuse_color = rgb(mid)
    return m


def flat(name, colour):
    m, nt = _new(name)
    c = nt.nodes.new('ShaderNodeRGB')
    c.outputs[0].default_value = rgb(colour)
    _out(nt, c.outputs[0])
    m['base'] = colour
    m.diffuse_color = rgb(colour)
    return m


def gradient(name, bottom, top, z0, z1, edge=None):
    """Unlit, bottom colour at height z0 to top colour at z1 (the object's own Z), darker at the edge."""
    m, nt = _new(name)
    N, L = nt.nodes.new, nt.links.new
    tc = N('ShaderNodeTexCoord')
    sep = N('ShaderNodeSeparateXYZ')
    L(tc.outputs['Object'], sep.inputs[0])
    mr = N('ShaderNodeMapRange')
    mr.name = 'height'
    mr.inputs['From Min'].default_value = z0
    mr.inputs['From Max'].default_value = z1
    L(sep.outputs['Z'], mr.inputs['Value'])
    mix = N('ShaderNodeMix')
    mix.data_type = 'RGBA'
    L(mr.outputs[0], mix.inputs['Factor'])
    mix.inputs['A'].default_value = rgb(bottom)
    mix.inputs['B'].default_value = rgb(top)
    colour = mix.outputs['Result']
    if edge:
        lw = N('ShaderNodeLayerWeight')
        lw.inputs['Blend'].default_value = 0.5
        e = N('ShaderNodeMapRange')
        e.inputs['From Min'].default_value = 0.45
        e.inputs['From Max'].default_value = 0.9
        L(lw.outputs['Facing'], e.inputs['Value'])
        mx = N('ShaderNodeMix')
        mx.data_type = 'RGBA'
        L(e.outputs[0], mx.inputs['Factor'])
        L(colour, mx.inputs['A'])
        mx.inputs['B'].default_value = rgb(edge)
        colour = mx.outputs['Result']
    _out(nt, colour)
    m['base'] = top
    m.diffuse_color = rgb(top)
    return m


def line():
    """The drawn line: a shell round each part, seen only from inside."""
    m = flat('line', PAL['line'])
    m.use_backface_culling = True
    try:
        m.use_backface_culling_shadow = True
    except AttributeError:
        pass
    return m


def make_all():
    P = PAL
    return {
        'fur': toon('fur', P['fur_shadow'], P['fur_mid'], P['fur_light'], rim=P['fur_rim'], strokes=0.15),
        'ear_in': toon('ear_in', '#995a5e', '#c1746a', '#d98568'),
        'nose': toon('nose', P['nose_dark'], P['nose'], P['nose']),
        'pad': toon('pad', P['nose_dark'], P['pad'], P['pad_light']),
        'collar': toon('collar', P['collar_dark'], P['collar'], P['collar_light']),
        'buckle': toon('buckle', P['buckle_dark'], P['buckle'], P['buckle_light']),
        'tag': toon('tag', P['tag_dark'], P['tag'], P['tag_light']),
        'tag_text': flat('tag_text', P['tag_text']),
        'set': toon('set', P['set_dark'], P['set'], P['set_light']),
        'set_glow': toon('set_glow', P['set'], P['set_glow'], '#c9f1ee'),
        'foam': toon('foam', '#1c1e24', P['foam'], P['foam_light']),
        'mouth': flat('mouth', P['mouth']),
        'tongue': toon('tongue', P['tongue_dark'], P['tongue'], P['tongue']),
        'tooth': flat('tooth', P['tooth']),
        'glint': flat('glint', P['glint']),
        'dark': flat('dark', P['line']),
        'hat_pink': toon('hat_pink', P['hat_dark'], P['hat_pink'], P['hat_pink']),
        'hat_yellow': toon('hat_yellow', '#e0a030', P['hat_yellow'], P['hat_yellow']),
        'hat_cyan': toon('hat_cyan', '#2fa9b8', P['hat_cyan'], P['hat_cyan']),
        'pom': toon('pom', '#d98a2e', P['pom'], P['hat_yellow']),
        'line': line(),
    }


def outline(obj, mat_line, thickness=0.011):
    """Draw the line round obj: a shell, turned inside out, in the line colour."""
    if mat_line.name not in [m.name for m in obj.data.materials if m]:
        obj.data.materials.append(mat_line)
    mod = obj.modifiers.get('line') or obj.modifiers.new('line', 'SOLIDIFY')
    mod.thickness = thickness
    mod.offset = 1.0
    mod.use_flip_normals = True
    mod.use_rim = False
    mod.material_offset = len(obj.data.materials) - 1
    mod.use_quality_normals = True
    return mod


def set_height(name, z0, z1):
    """Move a gradient to a new height (after he is sat down, his parts are somewhere else)."""
    m = bpy.data.materials.get('J_' + name)
    if not m:
        return
    n = m.node_tree.nodes.get('height')
    n.inputs['From Min'].default_value = z0
    n.inputs['From Max'].default_value = z1
