"""The Wall in space: planets for the projects, satellites for the runners (JM-491; plan claude/wall-space-theme.md).

Run in Blender (5.2), from this folder:
    import sys; sys.path.insert(0, r'<this folder>'); import space; space.build()

Everything is made in a scene of its own, "Space", so the file it is run in (Jimmy5.blend) keeps its own scene as it
was. Run again, it clears what it made and makes it afresh. Then:
    space.lineup()     one picture of every planet and satellite, lit from one side, for a look
    space.bake()       each planet's surface into plain pictures (out/<id>-colour.png, out/<id>-glow.png for lava)
    space.export()     out/planet.glb, out/<shape>.glb: small model files for a page

A planet is a ball, painted by Blender's own material sums in its project's colour (the Wall's night colours, from
store/api/src/dashboard/tokens.css), with a thin glowing skin of air round it, a cloud shell for an ocean world,
and a ring for a sprint going. Its kind of surface is fixed per project (SURFACE), the same as the mock-up's.

A satellite is a body with solar panels on hinges (objects called hinge-...), so a page can fold them in when the
computer is drained, and a green light that blinks while it works (light-...). One shape per kind of computer
(SHAPE), so they can be told apart without colour.
"""
import math
import os

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
SCENE = 'Space'

# The Wall's colour for each project (night look: tokens.css, the dark theme), and the spares it picks from by name.
COLOURS = {'charliedoll': '#ee8fb8', 'photo-organiser': '#6cb4ff', 'transcenders': '#f0a35e', 'robroblox': '#72d39c',
           'bidden': '#c79bf2', 'jimmy': '#5fcfcf', 'portfolio': '#b1bac4', 'aaah': '#e3c45a', 'azuredump': '#a5a5f0'}
SPARE = ['#e0a33a', '#d9628f', '#6aa9e9', '#8bc36a', '#b58be6', '#4fc4b8', '#e8805a', '#c9c25a']
SURFACE = {'jimmy': 'ocean', 'charliedoll': 'gas', 'photo-organiser': 'ice', 'robroblox': 'rock', 'transcenders': 'gas',
           'portfolio': 'rock', 'bidden': 'lava', 'azuredump': 'ice', 'aaah': 'gas'}
KINDS = ('rock', 'lava', 'ice', 'gas', 'ocean')
SHAPES = ('small', 'quad', 'station', 'dish', 'cube')

# Satellite colours: a pale metal body, gold foil, deep blue panels with a pale grid, a green working light.
METAL, FOIL, PANEL, GRID, GO = (0.62, 0.66, 0.72), (0.92, 0.66, 0.24), (0.035, 0.09, 0.26), (0.42, 0.55, 0.78), (0.35, 1.0, 0.55)


def hex_rgb(h):
    h = h.lstrip('#')
    srgb = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return tuple((c / 12.92) if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb)   # Blender works in linear light


def colour_of(pid):
    if pid in COLOURS:
        return hex_rgb(COLOURS[pid])
    n = 7
    for ch in pid:
        n = (n * 31 + ord(ch)) % 9973
    return hex_rgb(SPARE[n % len(SPARE)])


def surface_of(pid):
    if pid in SURFACE:
        return SURFACE[pid]
    n = 0
    for ch in pid:
        n = (n * 31 + ord(ch)) % 9973
    return KINDS[n % len(KINDS)]


def shade(c, f):
    """Lighter (f > 0, toward white) or darker (f < 0, toward black)."""
    return tuple(x + (1 - x) * f for x in c) if f > 0 else tuple(x * (1 + f) for x in c)


# ---------------------------------------------------------------- the scene

def scene():
    sc = bpy.data.scenes.get(SCENE) or bpy.data.scenes.new(SCENE)
    world = bpy.data.worlds.get('SpaceWorld') or bpy.data.worlds.new('SpaceWorld')
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    bg.inputs['Color'].default_value = (0.02, 0.024, 0.04, 1)
    bg.inputs['Strength'].default_value = 1.0
    sc.world = world
    sc.render.engine = 'BLENDER_EEVEE'
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    sc.view_settings.view_transform = 'Standard'
    return sc


def clear(sc):
    for o in list(sc.collection.all_objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for c in list(sc.collection.children):
        bpy.data.collections.remove(c)
    for m in [m for m in bpy.data.materials if m.name.startswith('sp-')]:
        bpy.data.materials.remove(m)
    for m in [m for m in bpy.data.meshes if m.users == 0]:
        bpy.data.meshes.remove(m)


def collection(sc, name):
    c = bpy.data.collections.new(name)
    sc.collection.children.link(c)
    return c


def adopt(child, parent):
    """Hold child by parent, keeping it where it is."""
    bpy.context.view_layer.update()
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()
    return child


def link(obj, coll):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)
    return obj


# ---------------------------------------------------------------- material helpers

class Nodes:
    """A small helper for writing material sums: n.add('ShaderNodeNoiseTexture', Scale=4) and n.wire(a, 'Fac', b, 'Fac')."""

    def __init__(self, mat):
        mat.use_nodes = True
        self.nt = mat.node_tree
        for k in list(self.nt.nodes):
            self.nt.nodes.remove(k)
        self.x = 0

    def add(self, kind, **values):
        n = self.nt.nodes.new(kind)
        n.location = (self.x, 0)
        self.x += 200
        for k, v in values.items():
            if k in n.inputs:
                n.inputs[k].default_value = v
            else:
                setattr(n, k, v)
        return n

    def wire(self, a, out, b, inp):
        self.nt.links.new(a.outputs[out] if isinstance(out, str) else a.outputs[out], b.inputs[inp] if isinstance(inp, str) else b.inputs[inp])

    def ramp(self, stops, interp='LINEAR'):
        r = self.add('ShaderNodeValToRGB')
        r.color_ramp.interpolation = interp
        els = r.color_ramp.elements
        while len(els) > 1:
            els.remove(els[-1])
        els[0].position, els[0].color = stops[0][0], (*stops[0][1], 1)
        for p, c in stops[1:]:
            e = els.new(p)
            e.color = (*c, 1)
        return r

    def math(self, op, a=None, b=None, **kw):
        m = self.add('ShaderNodeMath', operation=op, **kw)
        for i, v in enumerate((a, b)):
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            elif v is not None:
                self.wire(v[0], v[1], m, i)
        return m


def solid(name, colour, rough=0.6, metal=0.0, glow=None, strength=0.0):
    mat = bpy.data.materials.new(name)
    n = Nodes(mat)
    out, bsdf = n.add('ShaderNodeOutputMaterial'), n.add('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (*colour, 1)
    bsdf.inputs['Roughness'].default_value = rough
    bsdf.inputs['Metallic'].default_value = metal
    if glow:
        bsdf.inputs['Emission Color'].default_value = (*glow, 1)
        bsdf.inputs['Emission Strength'].default_value = strength
    n.wire(bsdf, 'BSDF', out, 'Surface')
    return mat


# ---------------------------------------------------------------- planets

def surface_material(pid, kind, col):
    """The planet's paint: Blender's own sums over the point on the ball (its object coordinates, -1 to 1)."""
    mat = bpy.data.materials.new('sp-' + pid)
    n = Nodes(mat)
    out, bsdf = n.add('ShaderNodeOutputMaterial'), n.add('ShaderNodeBsdfPrincipled')
    n.wire(bsdf, 'BSDF', out, 'Surface')
    bsdf.inputs['Roughness'].default_value = 0.85
    tc = n.add('ShaderNodeTexCoord')
    seed = sum(ord(ch) for ch in pid) % 97
    colour, height = None, None
    if kind == 'rock':
        noise = n.add('ShaderNodeTexNoise', noise_dimensions='4D', Scale=3.2, Detail=8.0, Roughness=0.6, W=seed)
        n.wire(tc, 'Object', noise, 'Vector')
        ground = n.ramp([(0.3, shade(col, -0.62)), (0.5, shade(col, -0.35)), (0.7, shade(col, -0.12))])
        n.wire(noise, 'Fac', ground, 'Fac')
        cr = n.add('ShaderNodeTexVoronoi', voronoi_dimensions='4D', Scale=5.0, Randomness=0.9, W=seed + 0.5)
        n.wire(tc, 'Object', cr, 'Vector')
        bowl = n.add('ShaderNodeMapRange', **{'From Min': 0.0, 'From Max': 0.32, 'To Min': 1.0, 'To Max': 0.0})
        n.wire(cr, 'Distance', bowl, 'Value')
        rim = n.add('ShaderNodeMapRange', **{'From Min': 0.22, 'From Max': 0.34, 'To Min': 0.0, 'To Max': 1.0})
        n.wire(cr, 'Distance', rim, 'Value')
        dark = n.add('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY')
        dark.inputs[6].default_value = (0.55, 0.55, 0.6, 1)
        n.wire(bowl, 'Result', dark, 0)
        n.wire(ground, 'Color', dark, 6)
        dark.inputs[7].default_value = (0.55, 0.55, 0.6, 1)
        colour = dark
        hmix = n.math('SUBTRACT', (rim, 'Result'), (bowl, 'Result'))
        h2 = n.math('ADD', (hmix, 0), (noise, 'Fac'))
        height = h2
    elif kind == 'ocean':
        noise = n.add('ShaderNodeTexNoise', noise_dimensions='4D', Scale=1.8, Detail=9.0, Roughness=0.58, W=seed)
        n.wire(tc, 'Object', noise, 'Vector')
        land = n.ramp([(0.0, shade(col, -0.72)), (0.5, shade(col, -0.5)), (0.535, shade(col, -0.05)), (0.56, shade(col, 0.18)),
                       (0.68, shade(col, -0.1)), (0.8, shade(col, 0.35))])
        n.wire(noise, 'Fac', land, 'Fac')
        colour = land
        hm = n.add('ShaderNodeMapRange', **{'From Min': 0.53, 'From Max': 0.8, 'To Min': 0.0, 'To Max': 1.0})
        n.wire(noise, 'Fac', hm, 'Value')
        height = hm
        bsdf.inputs['Roughness'].default_value = 0.55
    elif kind == 'ice':
        sep = n.add('ShaderNodeSeparateXYZ')
        n.wire(tc, 'Object', sep, 'Vector')
        az = n.math('ABSOLUTE', (sep, 'Z'))
        wob = n.add('ShaderNodeTexNoise', noise_dimensions='4D', Scale=6.0, Detail=4.0, W=seed)
        n.wire(tc, 'Object', wob, 'Vector')
        az2 = n.math('MULTIPLY_ADD', (wob, 'Fac'), 0.18) ; n.wire(az, 0, az2, 2)
        cap = n.add('ShaderNodeMapRange', **{'From Min': 0.72, 'From Max': 0.8, 'To Min': 0.0, 'To Max': 1.0})
        n.wire(az2, 0, cap, 'Value')
        base = n.add('ShaderNodeTexNoise', noise_dimensions='4D', Scale=2.5, Detail=6.0, W=seed + 3)
        n.wire(tc, 'Object', base, 'Vector')
        field = n.ramp([(0.35, shade(col, -0.25)), (0.65, shade(col, 0.45))])
        n.wire(base, 'Fac', field, 'Fac')
        cracks = n.add('ShaderNodeTexVoronoi', voronoi_dimensions='4D', feature='DISTANCE_TO_EDGE', Scale=4.0, W=seed + 1)
        n.wire(tc, 'Object', cracks, 'Vector')
        line = n.add('ShaderNodeMapRange', **{'From Min': 0.0, 'From Max': 0.025, 'To Min': 1.0, 'To Max': 0.0})
        n.wire(cracks, 'Distance', line, 'Value')
        cracked = n.add('ShaderNodeMix', data_type='RGBA', blend_type='MIX')
        n.wire(line, 'Result', cracked, 0)
        n.wire(field, 'Color', cracked, 6)
        cracked.inputs[7].default_value = (*shade(col, -0.45), 1)
        capped = n.add('ShaderNodeMix', data_type='RGBA', blend_type='MIX')
        n.wire(cap, 'Result', capped, 0)
        n.wire(cracked, 2, capped, 6)
        capped.inputs[7].default_value = (0.93, 0.96, 1.0, 1)
        colour = capped
        height = n.math('SUBTRACT', (cap, 'Result'), (line, 'Result'))
        bsdf.inputs['Roughness'].default_value = 0.35
    elif kind == 'gas':
        warp = n.add('ShaderNodeTexNoise', noise_dimensions='4D', Scale=2.0, Detail=5.0, W=seed)
        n.wire(tc, 'Object', warp, 'Vector')
        sep = n.add('ShaderNodeSeparateXYZ')
        n.wire(tc, 'Object', sep, 'Vector')
        band = n.math('MULTIPLY_ADD', (warp, 'Fac'), 0.32) ; n.wire(sep, 'Z', band, 2)
        wave = n.math('SINE', (n.math('MULTIPLY', (band, 0), 9.0 + seed % 4), 0))
        fine = n.math('SINE', (n.math('MULTIPLY', (band, 0), 23.0), 0))
        mixw = n.math('MULTIPLY_ADD', (fine, 0), 0.3) ; n.wire(wave, 0, mixw, 2)
        tone = n.math('MULTIPLY_ADD', (mixw, 0), 0.35) ; tone.inputs[2].default_value = 0.5
        bands = n.ramp([(0.0, shade(col, -0.45)), (0.35, shade(col, -0.12)), (0.6, shade(col, 0.2)), (1.0, shade(col, 0.5))])
        n.wire(tone, 0, bands, 'Fac')
        # one big oval storm, at a fixed place
        spot = n.add('ShaderNodeVectorMath', operation='DISTANCE')
        n.wire(tc, 'Object', spot, 0)
        spot.inputs[1].default_value = (0.55, -0.65, -0.35)
        sq = n.add('ShaderNodeMapRange', **{'From Min': 0.12, 'From Max': 0.3, 'To Min': 1.0, 'To Max': 0.0})
        n.wire(spot, 'Value', sq, 'Value')
        storm = n.add('ShaderNodeMix', data_type='RGBA', blend_type='MIX')
        n.wire(sq, 'Result', storm, 0)
        n.wire(bands, 'Color', storm, 6)
        storm.inputs[7].default_value = (*shade(col, -0.55), 1)
        colour = storm
        bsdf.inputs['Roughness'].default_value = 0.7
    else:   # lava
        noise = n.add('ShaderNodeTexNoise', noise_dimensions='4D', Scale=3.0, Detail=7.0, W=seed)
        n.wire(tc, 'Object', noise, 'Vector')
        rock = n.ramp([(0.35, (0.012, 0.01, 0.012)), (0.65, shade(col, -0.86))])
        n.wire(noise, 'Fac', rock, 'Fac')
        colour = rock
        cracks = n.add('ShaderNodeTexVoronoi', voronoi_dimensions='4D', feature='DISTANCE_TO_EDGE', Scale=3.2, W=seed + 2)
        warpv = n.add('ShaderNodeVectorMath', operation='MULTIPLY_ADD')
        n.wire(noise, 'Color', warpv, 0)
        warpv.inputs[1].default_value = (0.25, 0.25, 0.25)
        n.wire(tc, 'Object', warpv, 2)
        n.wire(warpv, 'Vector', cracks, 'Vector')
        crack = n.add('ShaderNodeMapRange', **{'From Min': 0.0, 'From Max': 0.022, 'To Min': 1.0, 'To Max': 0.0})
        n.wire(cracks, 'Distance', crack, 'Value')
        where = n.add('ShaderNodeTexNoise', noise_dimensions='4D', Scale=1.6, Detail=2.0, W=seed + 9)   # only some of the cracks glow
        n.wire(tc, 'Object', where, 'Vector')
        some = n.add('ShaderNodeMapRange', **{'From Min': 0.45, 'From Max': 0.62, 'To Min': 0.0, 'To Max': 1.0})
        n.wire(where, 'Fac', some, 'Value')
        hot = n.math('MULTIPLY', (crack, 'Result'), (some, 'Result'))
        pools = n.add('ShaderNodeTexVoronoi', voronoi_dimensions='4D', Scale=4.0, W=seed + 4)            # and a few pools of it
        n.wire(tc, 'Object', pools, 'Vector')
        pool = n.add('ShaderNodeMapRange', **{'From Min': 0.04, 'From Max': 0.13, 'To Min': 1.0, 'To Max': 0.0})
        n.wire(pools, 'Distance', pool, 'Value')
        pool2 = n.math('MULTIPLY', (pool, 'Result'), (some, 'Result'))
        hot = n.math('MAXIMUM', (hot, 0), (pool2, 0))
        glow = n.ramp([(0.0, (0, 0, 0)), (0.5, shade(col, 0.05)), (1.0, shade(col, 0.7))])
        n.wire(hot, 0, glow, 'Fac')
        n.wire(glow, 'Color', bsdf, 'Emission Color')
        bsdf.inputs['Emission Strength'].default_value = 3.0
        height = n.math('SUBTRACT', 1.0, (hot, 0))
    n.wire(colour, 'Color' if 'Color' in colour.outputs else 2, bsdf, 'Base Color')
    if height is not None:
        bump = n.add('ShaderNodeBump', Strength=0.35, Distance=0.02)
        n.wire(height, 0 if 'Value' in height.outputs else 'Result', bump, 'Height')
        n.wire(bump, 'Normal', bsdf, 'Normal')
    return mat


def air_material(pid, col, strength):
    """A thin shell that glows softly at its edge, brightest on the lit side, and is clear face-on. The light it
    takes is measured by a plain white surface under the same lamps, so it follows wherever the light comes from."""
    mat = bpy.data.materials.new('sp-air-' + pid)
    mat.surface_render_method = 'BLENDED'
    n = Nodes(mat)
    out = n.add('ShaderNodeOutputMaterial')
    lw = n.add('ShaderNodeLayerWeight', Blend=0.6)
    edge = n.math('POWER', (lw, 'Facing'), 2.2)
    lit = n.add('ShaderNodeBsdfDiffuse', Color=(1, 1, 1, 1))
    amount = n.add('ShaderNodeShaderToRGB')
    n.wire(lit, 'BSDF', amount, 'Shader')
    bw = n.add('ShaderNodeRGBToBW')
    n.wire(amount, 'Color', bw, 'Color')
    how = n.math('MULTIPLY_ADD', (bw, 'Val'), 1.4) ; how.inputs[2].default_value = 0.35     # the dark side keeps a faint rim
    a = n.math('MULTIPLY', (edge, 0), (how, 0))
    a2 = n.math('MINIMUM', (a, 0), 1.0)
    em = n.add('ShaderNodeEmission', Color=(*shade(col, 0.35), 1), Strength=strength)
    clear_ = n.add('ShaderNodeBsdfTransparent')
    add = n.add('ShaderNodeAddShader')
    n.wire(clear_, 'BSDF', add, 0)
    n.wire(em, 'Emission', add, 1)
    mix = n.add('ShaderNodeMixShader')
    n.wire(a2, 0, mix, 'Fac')
    n.wire(clear_, 'BSDF', mix, 1)
    n.wire(add, 'Shader', mix, 2)
    n.wire(mix, 'Shader', out, 'Surface')
    return mat


def cloud_material(pid):
    mat = bpy.data.materials.new('sp-cloud-' + pid)
    mat.surface_render_method = 'BLENDED'
    n = Nodes(mat)
    out = n.add('ShaderNodeOutputMaterial')
    tc = n.add('ShaderNodeTexCoord')
    st = n.add('ShaderNodeMapping', vector_type='POINT')
    st.inputs['Scale'].default_value = (1.0, 1.0, 2.6)       # stretched along the turning, as clouds are
    n.wire(tc, 'Object', st, 'Vector')
    noise = n.add('ShaderNodeTexNoise', noise_dimensions='4D', Scale=2.6, Detail=6.0, W=sum(ord(c) for c in pid) % 53 + 7)
    n.wire(st, 'Vector', noise, 'Vector')
    a = n.add('ShaderNodeMapRange', **{'From Min': 0.52, 'From Max': 0.68, 'To Min': 0.0, 'To Max': 0.85})
    n.wire(noise, 'Fac', a, 'Value')
    white = n.add('ShaderNodeBsdfDiffuse', Color=(0.95, 0.97, 1.0, 1))
    clear_ = n.add('ShaderNodeBsdfTransparent')
    mix = n.add('ShaderNodeMixShader')
    n.wire(a, 'Result', mix, 'Fac')
    n.wire(clear_, 'BSDF', mix, 1)
    n.wire(white, 'BSDF', mix, 2)
    n.wire(mix, 'Shader', out, 'Surface')
    return mat


def ring_material(pid, col):
    """Bands across the ring, light and dark in the planet's colour, with gaps."""
    mat = bpy.data.materials.new('sp-ring-' + pid)
    mat.surface_render_method = 'BLENDED'
    n = Nodes(mat)
    out = n.add('ShaderNodeOutputMaterial')
    tc = n.add('ShaderNodeTexCoord')
    length = n.add('ShaderNodeVectorMath', operation='LENGTH')
    n.wire(tc, 'Object', length, 0)
    across = n.add('ShaderNodeMapRange', **{'From Min': 1.45, 'From Max': 2.15, 'To Min': 0.0, 'To Max': 1.0})
    n.wire(length, 'Value', across, 'Value')
    bands = n.add('ShaderNodeTexNoise', noise_dimensions='1D', Scale=14.0, Detail=3.0, W=0.0)
    n.wire(across, 'Result', bands, 'W')
    alpha = n.ramp([(0.0, (0, 0, 0)), (0.06, (0.7, 0.7, 0.7)), (0.4, (0.35, 0.35, 0.35)), (0.47, (0.05, 0.05, 0.05)), (0.52, (0.8, 0.8, 0.8)), (0.94, (0.6, 0.6, 0.6)), (1.0, (0, 0, 0))])
    n.wire(across, 'Result', alpha, 'Fac')
    a2 = n.math('MULTIPLY', (alpha, 'Color'), (bands, 'Fac'))
    a3 = n.math('MULTIPLY', (a2, 0), 1.6)
    tint = n.add('ShaderNodeBsdfDiffuse', Color=(*shade(col, 0.3), 1))
    clear_ = n.add('ShaderNodeBsdfTransparent')
    mix = n.add('ShaderNodeMixShader')
    n.wire(a3, 0, mix, 'Fac')
    n.wire(clear_, 'BSDF', mix, 1)
    n.wire(tint, 'BSDF', mix, 2)
    n.wire(mix, 'Shader', out, 'Surface')
    return mat


def planet(pid, coll, at=(0, 0, 0), ring=False):
    """One planet, radius 1, at `at`: its ball (planet-<id>), its air (air-<id>), clouds for an ocean world, and a
    ring if a sprint is going (ring-<id>)."""
    kind, col = surface_of(pid), colour_of(pid)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=1.0, location=at)
    ball = link(bpy.context.active_object, coll)
    ball.name = 'planet-' + pid
    bpy.ops.object.shade_smooth()
    ball.data.materials.append(surface_material(pid, kind, col))
    ball['kind'] = kind
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.09, location=at)
    air = link(bpy.context.active_object, coll)
    air.name = 'air-' + pid
    bpy.ops.object.shade_smooth()
    air.data.materials.append(air_material(pid, col, 1.6))
    adopt(air, ball)
    if kind == 'ocean':
        bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.018, location=at)
        cl = link(bpy.context.active_object, coll)
        cl.name = 'clouds-' + pid
        bpy.ops.object.shade_smooth()
        cl.data.materials.append(cloud_material(pid))
        adopt(cl, ball)
    if ring:
        bpy.ops.mesh.primitive_circle_add(vertices=128, radius=2.15, fill_type='NOTHING', location=at)
        r = link(bpy.context.active_object, coll)
        r.name = 'ring-' + pid
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.extrude_region_shrink_fatten(TRANSFORM_OT_shrink_fatten={'value': 0.0})
        bpy.ops.transform.resize(value=(1.45 / 2.15, 1.45 / 2.15, 1.0))
        bpy.ops.object.mode_set(mode='OBJECT')
        r.data.materials.append(ring_material(pid, col))
        r.rotation_euler = (math.radians(16), math.radians(-18), 0)    # nearly edge-on, slanted
        adopt(r, ball)
    ball.rotation_euler = (math.radians(12), 0, 0)    # a slight tilt to its turning
    return ball


# ---------------------------------------------------------------- satellites

def box(name, size, at, mat, coll, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=at)
    o = link(bpy.context.active_object, coll)
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    o.data.materials.append(mat)
    if parent:
        adopt(o, parent)
    return o


def panel_material():
    mat = bpy.data.materials.get('sp-panel')
    if mat:
        return mat
    mat = bpy.data.materials.new('sp-panel')
    n = Nodes(mat)
    out, bsdf = n.add('ShaderNodeOutputMaterial'), n.add('ShaderNodeBsdfPrincipled')
    n.wire(bsdf, 'BSDF', out, 'Surface')
    tc = n.add('ShaderNodeTexCoord')
    sep = n.add('ShaderNodeSeparateXYZ')                  # the panel lies across X and Z: its grid is laid out on those
    n.wire(tc, 'Object', sep, 'Vector')
    flat = n.add('ShaderNodeCombineXYZ')
    n.wire(sep, 'X', flat, 'X')
    n.wire(sep, 'Z', flat, 'Y')
    grid = n.add('ShaderNodeTexBrick', offset=0.0, squash=1.0, **{'Mortar Size': 0.012, 'Scale': 7.0, 'Brick Width': 0.5, 'Row Height': 0.25,
                                                                   'Color1': (*PANEL, 1), 'Color2': (*shade(PANEL, 0.1), 1), 'Mortar': (*GRID, 1)})
    n.wire(flat, 'Vector', grid, 'Vector')
    n.wire(grid, 'Color', bsdf, 'Base Color')
    bsdf.inputs['Metallic'].default_value = 0.4
    bsdf.inputs['Roughness'].default_value = 0.25
    return mat


def wing(name, hinge_at, side, length, width, mat, coll, body, up=0.0):
    """A solar panel on a hinge at the body's edge. The hinge (an empty) turns about the body's front-to-back line:
    0 is open flat, 90 degrees is folded against the body. side: +1 right, -1 left."""
    hinge = bpy.data.objects.new('hinge-' + name, None)
    coll.objects.link(hinge)
    hinge.location = hinge_at
    adopt(hinge, body)
    strut = box('strut-' + name, (0.12, 0.03, 0.03), (hinge_at[0] + side * 0.06, hinge_at[1], hinge_at[2] + up), solid_metal(), coll, hinge)
    p = box('panel-' + name, (length, 0.025, width), (hinge_at[0] + side * (0.12 + length / 2), hinge_at[1], hinge_at[2] + up), mat, coll, hinge)
    return hinge


def solid_metal():
    return bpy.data.materials.get('sp-metal') or solid('sp-metal', METAL, rough=0.4, metal=0.35)


def foil():
    return bpy.data.materials.get('sp-foil') or solid('sp-foil', FOIL, rough=0.38, metal=0.4)


def go_light():
    return bpy.data.materials.get('sp-go') or solid('sp-go', GO, rough=0.4, glow=GO, strength=1.6)


def light_dot(name, at, coll, body, r=0.07):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=r, location=at)
    o = link(bpy.context.active_object, coll)
    o.name = 'light-' + name
    o.data.materials.append(go_light())
    adopt(o, body)
    return o


def satellite(shape, coll, at=(0, 0, 0)):
    """One satellite, about 1 across its body, at `at`, panels open. Its front faces -Y (toward the camera)."""
    x, y, z = at
    pm = panel_material()
    if shape == 'station':      # GitHub: a ring station
        bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.12, major_segments=64, minor_segments=16, location=at,
                                         rotation=(math.radians(90), 0, 0))
        body = link(bpy.context.active_object, coll)
        body.name = 'sat-station'
        body.data.materials.append(solid_metal())
        bpy.ops.object.shade_smooth()
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.34, location=at)
        hub = link(bpy.context.active_object, coll)
        hub.name = 'hub-station'
        hub.data.materials.append(foil())
        bpy.ops.object.shade_smooth()
        adopt(hub, body)
        for a in (0, 90):
            s = box('spoke-%d' % a, (2.0, 0.05, 0.05), at, solid_metal(), coll, body)
            s.rotation_euler = (0, math.radians(a), 0)
        # one panel on a mast above the hub, hinged at the hub's top
        hinge = bpy.data.objects.new('hinge-station-up', None)
        coll.objects.link(hinge)
        hinge.location = (x, y, z + 0.34)
        adopt(hinge, body)
        box('strut-station-up', (0.03, 0.03, 0.3), (x, y, z + 0.49), solid_metal(), coll, hinge)
        box('panel-station-up', (0.5, 0.025, 0.75), (x, y, z + 1.0), pm, coll, hinge)
        light_dot('station', (x, y - 0.32, z + 0.1), coll, body)
    elif shape == 'dish':       # Azure: a body with a dish
        body = box('sat-dish', (0.8, 0.7, 0.9), at, foil(), coll)
        wing('dish-left', (x - 0.4, y, z), -1, 1.1, 0.42, pm, coll, body)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.55, location=(x + 0.95, y, z))
        dish = link(bpy.context.active_object, coll)
        dish.name = 'dish'
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        for v in dish.data.vertices:
            v.select = v.co.x > -0.18
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.delete(type='VERT')
        bpy.ops.object.mode_set(mode='OBJECT')
        dish.data.materials.append(bpy.data.materials.get('sp-white') or solid('sp-white', (0.9, 0.92, 0.95), rough=0.4))
        mod = dish.modifiers.new('thick', 'SOLIDIFY')
        mod.thickness = 0.04
        bpy.ops.object.shade_smooth()
        adopt(dish, body)
        box('dish-arm', (0.4, 0.06, 0.06), (x + 0.55, y, z), solid_metal(), coll, body)
        light_dot('dish', (x, y - 0.36, z + 0.3), coll, body)
    elif shape == 'quad':       # Balrog and Gandalf: four panels
        body = box('sat-quad', (0.9, 0.9, 1.2), at, foil(), coll)
        for side, sx in ((-1, x - 0.45), (1, x + 0.45)):
            for row, dz in (('a', 0.3), ('b', -0.3)):
                wing('quad-%s%s' % ('left' if side < 0 else 'right', row), (sx, y, z), side, 1.15, 0.42, pm, coll, body, up=dz)
        light_dot('quad', (x, y - 0.46, z + 0.45), coll, body)
        box('quad-band', (0.92, 0.92, 0.12), at, solid_metal(), coll, body)
    elif shape == 'cube':       # a phone: a tiny cube with one panel
        body = box('sat-cube', (0.55, 0.55, 0.55), at, foil(), coll)
        wing('cube-right', (x + 0.275, y, z), 1, 0.7, 0.36, pm, coll, body)
        light_dot('cube', (x, y - 0.28, z + 0.18), coll, body, r=0.05)
    else:                       # msi (and any other computer): small, two panels
        body = box('sat-small', (0.75, 0.75, 0.75), at, foil(), coll)
        for side, sx, nm in ((-1, x - 0.375, 'left'), (1, x + 0.375, 'right')):
            wing('small-' + nm, (sx, y, z), side, 0.95, 0.46, pm, coll, body)
        light_dot('small', (x, y - 0.38, z + 0.25), coll, body)
        box('small-band', (0.77, 0.77, 0.1), at, solid_metal(), coll, body)
    # an aerial on top, for all but the station
    if shape != 'station':
        top = body.dimensions.z / 2
        a = box('aerial-' + shape, (0.025, 0.025, 0.45), (x, y, z + top + 0.22), solid_metal(), coll, body)
    body['shape'] = shape
    return body


def fold(body, amount):
    """Fold its panels: 0 open flat, 1 folded against the body (drained)."""
    for h in [c for c in body.children_recursive if c.name.startswith('hinge-')]:
        side = 1 if (h.matrix_world.translation.x - body.matrix_world.translation.x) > 0 else -1
        h.rotation_euler = (0, -side * math.radians(85) * amount, 0)


# ---------------------------------------------------------------- build, look, bake, export

PROJECTS = ('jimmy', 'charliedoll', 'photo-organiser', 'robroblox', 'transcenders', 'portfolio', 'bidden')


def build(projects=PROJECTS, rings=('charliedoll',)):
    """Make every planet and satellite in the Space scene, laid out in a row each, ready for lineup()."""
    sc = scene()
    win = bpy.context.window
    old = win.scene
    win.scene = sc
    try:
        clear(sc)
        pc, scc = collection(sc, 'Planets'), collection(sc, 'Satellites')
        made = {'planets': [], 'satellites': []}
        for i, pid in enumerate(projects):
            planet(pid, pc, at=(i * 3.2, 0, 0), ring=pid in rings)
            made['planets'].append((pid, surface_of(pid)))
        for i, shape in enumerate(SHAPES):
            body = satellite(shape, scc, at=(i * 3.6, 0, -4.5))
            r = body.rotation_euler
            body.rotation_euler = (r.x + math.radians(14), r.y, r.z + math.radians(-26))    # a three-quarter view for the look
            made['satellites'].append(shape)
        return made
    finally:
        win.scene = old


def lineup(path=None, size=(2400, 1100), light_from=(-1.0, -0.6, 0.35)):
    """One picture of the planets (top row) and satellites (bottom row), lit from one side as Jimmy would light them."""
    sc = bpy.data.scenes[SCENE]
    win = bpy.context.window
    old = win.scene
    win.scene = sc
    try:
        for o in [o for o in sc.objects if o.name in ('SpaceSun', 'SpaceCam', 'SpaceFill')]:
            bpy.data.objects.remove(o, do_unlink=True)
        sun = bpy.data.objects.new('SpaceSun', bpy.data.lights.get('SpaceSun') or bpy.data.lights.new('SpaceSun', 'SUN'))
        sc.collection.objects.link(sun)
        sun.data.energy = 4.5
        sun.data.angle = math.radians(3)
        sun.rotation_euler = Vector(light_from).normalized().to_track_quat('Z', 'Y').to_euler()
        fill = bpy.data.objects.new('SpaceFill', bpy.data.lights.get('SpaceFill') or bpy.data.lights.new('SpaceFill', 'SUN'))
        sc.collection.objects.link(fill)
        fill.data.energy = 0.35                     # the dark side never goes fully black
        fill.rotation_euler = (-Vector(light_from)).normalized().to_track_quat('Z', 'Y').to_euler()
        cam = bpy.data.objects.new('SpaceCam', bpy.data.cameras.get('SpaceCam') or bpy.data.cameras.new('SpaceCam'))
        sc.collection.objects.link(cam)
        sc.camera = cam
        cam.data.type = 'ORTHO'
        xs = [o.matrix_world.translation.x for o in sc.objects if o.name.startswith(('planet-', 'sat-'))]
        cam.data.ortho_scale = max(xs) - min(xs) + 4.5
        cam.location = ((max(xs) + min(xs)) / 2, -30, -2.0)
        cam.rotation_euler = (math.radians(90), 0, 0)
        sc.render.resolution_x, sc.render.resolution_y = size
        sc.render.filepath = path or os.path.join(OUT, 'lineup.png')
        os.makedirs(os.path.dirname(sc.render.filepath), exist_ok=True)
        bpy.ops.render.render(write_still=True, scene=sc.name)
        return sc.render.filepath
    finally:
        win.scene = old


def look(path=None, size=(1920, 1080)):
    """How they might sit on the Wall: the planets on an oval round the middle, lit from a lamp there (where Jimmy
    is), with the satellites parked between them and one circling a planet. For a look only: it moves what build()
    made, and build() puts them back in rows."""
    sc = bpy.data.scenes[SCENE]
    win = bpy.context.window
    old = win.scene
    win.scene = sc
    try:
        for o in [o for o in sc.objects if o.name in ('SpaceSun', 'SpaceCam', 'SpaceFill', 'SpaceJimmy', 'SpaceGlow')]:
            bpy.data.objects.remove(o, do_unlink=True)
        balls = sorted([o for o in sc.objects if o.name.startswith('planet-')], key=lambda o: o.name)
        sats = [o for o in sc.objects if o.name.startswith('sat-')]
        sizes = {'jimmy': 1.25, 'charliedoll': 1.1, 'photo-organiser': 1.0, 'robroblox': 0.75, 'transcenders': 0.9, 'portfolio': 0.6, 'bidden': 0.7}
        n = len(balls)
        for i, b in enumerate(balls):
            a = math.pi / 2 + (i + 0.5) / n * math.pi * 2
            s_ = sizes.get(b.name[7:], 0.8)
            b.location = (math.cos(a) * 9.5, 0, math.sin(a) * 5.0)
            b.scale = (s_, s_, s_)
        spots = [(-4.6, 2.6), (4.8, 2.4), (-5.2, -2.6), (5.4, -2.2), (0.0, -3.4)]
        for b, (sx, sz) in zip(sats, spots):
            b.location = (sx, 1.5, sz)
            b.scale = (0.32, 0.32, 0.32)
        # one at work, circling Snappier
        po = sc.objects.get('planet-photo-organiser')
        if po and sats:
            q = sats[1]
            q.location = po.location + Vector((po.scale.x * 1.7, -1.0, po.scale.x * 0.5))
        lamp = bpy.data.objects.new('SpaceJimmy', bpy.data.lights.get('SpaceJimmy') or bpy.data.lights.new('SpaceJimmy', 'POINT'))
        sc.collection.objects.link(lamp)
        lamp.data.energy = 2600
        lamp.data.shadow_soft_size = 0.8
        lamp.data.color = (1.0, 0.95, 0.88)
        lamp.location = (0, -1.5, 0)
        fill = bpy.data.objects.new('SpaceFill', bpy.data.lights.get('SpaceFill') or bpy.data.lights.new('SpaceFill', 'SUN'))
        sc.collection.objects.link(fill)
        fill.data.energy = 0.25
        fill.rotation_euler = (math.radians(90), 0, 0)
        # (Jimmy in his saucer, out/jimmy-saucer-still.png, is laid in the middle afterwards.)
        cam = bpy.data.objects.new('SpaceCam', bpy.data.cameras.get('SpaceCam') or bpy.data.cameras.new('SpaceCam'))
        sc.collection.objects.link(cam)
        sc.camera = cam
        cam.data.type = 'ORTHO'
        cam.data.ortho_scale = 25
        cam.location = (0, -30, 0)
        cam.rotation_euler = (math.radians(90), 0, 0)
        sc.render.resolution_x, sc.render.resolution_y = size
        sc.render.filepath = path or os.path.join(OUT, 'look.png')
        bpy.ops.render.render(write_still=True, scene=sc.name)
        return sc.render.filepath
    finally:
        win.scene = old


# ---------------------------------------------------------------- for the Wall: plain pictures and model files
#
# Ant (25 Nov 02:31): live 3D on the Wall. So each kind of surface is baked once, in grey, and the page tints it
# with the project's colour. That works because of how surface_material() colours things: every colour is
# shade(c, f) for a project colour c and a step f (-1 black, 0 the colour, 1 white), and with c at 0.5 grey,
# shade gives exactly 0.5 + 0.5 f. So a grey bake holds f itself, per channel, and the page works out
#     f = 2 v - 1;   colour = f > 0 ? c + (1 - c) f : c (1 + f)
# for any project, the spare colours included. Two of each kind (VARIANTS), so two rocky planets differ.

VARIANTS = ('a', 'b')
BAKE = (1024, 512)
GREY = (0.5, 0.5, 0.5)


def _bake_image(name, size, alpha=False, data=False):
    img = bpy.data.images.get(name)
    if img:
        bpy.data.images.remove(img)
    img = bpy.data.images.new(name, size[0], size[1], alpha=alpha, float_buffer=False)
    if data:
        img.colorspace_settings.name = 'Non-Color'
    return img


def _bake(obj, mat, img, kind, pass_filter=None):
    nt = mat.node_tree
    node = nt.nodes.new('ShaderNodeTexImage')
    node.image = img
    for k in nt.nodes:
        k.select = False
    node.select = True
    nt.nodes.active = node
    vl = bpy.context.view_layer
    for o in vl.objects:
        o.select_set(False)
    obj.select_set(True)
    vl.objects.active = obj
    try:
        bpy.ops.object.bake(type=kind, pass_filter=pass_filter or set(), margin=4, use_clear=True)
    finally:
        nt.nodes.remove(node)


def _save(img, path, jpeg=True):
    img.filepath_raw = path
    img.file_format = 'JPEG' if jpeg else 'PNG'
    img.save()
    return path


def bake(kinds=KINDS, variants=VARIANTS):
    """Bake every kind of surface, in grey, onto a ball's own pictures: out/<kind>-<v>-shade.jpg (the steps, see
    above), out/<kind>-<v>-normal.png (the bumps), out/lava-<v>-glow.jpg (where the cracks glow) and
    out/ocean-<v>-clouds.jpg (where the clouds are). Returns the files written."""
    sc = scene()
    win = bpy.context.window
    old = win.scene
    win.scene = sc
    engine = sc.render.engine
    written = []
    try:
        sc.render.engine = 'CYCLES'
        sc.cycles.device = 'CPU'
        sc.cycles.samples = 4
        coll = bpy.data.collections.get('Bake') or collection(sc, 'Bake')
        os.makedirs(OUT, exist_ok=True)
        for kind in kinds:
            for v in variants:
                name = '%s-%s' % (kind, v)
                for o in list(coll.objects):
                    bpy.data.objects.remove(o, do_unlink=True)
                bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=1.0)
                ball = link(bpy.context.active_object, coll)
                bpy.ops.object.shade_smooth()
                mat = surface_material(name, kind, GREY)
                ball.data.materials.append(mat)
                img = _bake_image(name + '-shade', BAKE)
                _bake(ball, mat, img, 'DIFFUSE', {'COLOR'})
                written.append(_save(img, os.path.join(OUT, name + '-shade.jpg')))
                img = _bake_image(name + '-normal', BAKE, data=True)
                _bake(ball, mat, img, 'NORMAL')
                written.append(_save(img, os.path.join(OUT, name + '-normal.png'), jpeg=False))
                if kind == 'lava':
                    img = _bake_image(name + '-glow', BAKE)
                    _bake(ball, mat, img, 'EMIT')
                    written.append(_save(img, os.path.join(OUT, name + '-glow.jpg')))
                if kind == 'ocean':
                    cm = cloud_material(name)
                    # the clouds' cover, as light: a plain emission of the same sum
                    n = cm.node_tree
                    mix = next(k for k in n.nodes if k.bl_idname == 'ShaderNodeMixShader')
                    em = n.nodes.new('ShaderNodeEmission')
                    n.links.new(mix.inputs['Fac'].links[0].from_socket, em.inputs['Strength'])
                    em.inputs['Color'].default_value = (1, 1, 1, 1)
                    n.links.new(em.outputs['Emission'], n.nodes['Material Output'].inputs['Surface'])
                    ball.data.materials[0] = cm
                    img = _bake_image(name + '-clouds', BAKE)
                    _bake(ball, cm, img, 'EMIT')
                    written.append(_save(img, os.path.join(OUT, name + '-clouds.jpg')))
                bpy.data.objects.remove(ball, do_unlink=True)
        return written
    finally:
        sc.render.engine = engine
        win.scene = old


def ring_picture(path=None, width=512, seed=5):
    """The ring's bands, from inside edge (left) to outside (right): light where there is ring, clear in its gaps.
    A strip one pixel high, as a PNG with see-through; the page tints it with the project's colour."""
    import random
    rnd = random.Random(seed)
    img = _bake_image('ring-bands', (width, 1), alpha=True)
    px = []
    noise = [rnd.random() for _ in range(64)]
    for i in range(width):
        t = i / (width - 1)
        edge = min(1.0, t / 0.05, (1 - t) / 0.06)
        gap = 0.08 if 0.47 < t < 0.52 else 1.0                       # one clear gap, as Saturn's
        n = noise[int(t * 63)] * 0.5 + 0.5
        dense = (0.55 + 0.45 * math.sin(t * 34.0 + 1.3) ** 2) * n
        a = max(0.0, edge) * gap * dense
        light = 0.75 + 0.25 * n
        px += [light, light, light, a]
    img.pixels.foreach_set(px)
    img.update()
    path = path or os.path.join(OUT, 'ring.png')
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    return path


def panel_picture(path=None, size=(256, 128)):
    """A solar panel's face: deep blue cells with a pale grid. For the model files (Blender's own brick sum does
    not go into a model file)."""
    w, h = size
    img = _bake_image('panel-cells', size)
    px = []
    for y in range(h):
        for x in range(w):
            line = (x % 32 < 2) or (y % 32 < 2)
            c = GRID if line else (shade(PANEL, 0.08) if ((x // 32 + y // 32) % 2) else PANEL)
            px += [*c, 1.0]
    img.pixels.foreach_set(px)
    img.update()
    path = path or os.path.join(OUT, 'panel.png')
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    return img


def export(path=None):
    """out/space.glb: one ball (planet, radius 1, with the pictures' layout), its cloud shell (clouds), a ring
    (ring: inside radius 1.45, outside 2.15, its pictures across from inside to outside), and the five satellites
    (sat-<shape>, each at the middle, panels open, front toward -Y in Blender, which is +Z for a page). The page
    paints the planet and ring itself (bake(), ring_picture()); the satellites carry their own colours."""
    sc = scene()
    win = bpy.context.window
    old = win.scene
    win.scene = sc
    try:
        clear(sc)                                          # names must be free: build() again for a look afterwards
        coll = collection(sc, 'Export')
        plain = bpy.data.materials.get('sp-plain') or solid('sp-plain', (1, 1, 1))
        bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=1.0)
        ball = link(bpy.context.active_object, coll)
        ball.name = 'planet'
        bpy.ops.object.shade_smooth()
        ball.data.materials.append(plain)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.018)
        cl = link(bpy.context.active_object, coll)
        cl.name = 'clouds'
        bpy.ops.object.shade_smooth()
        cl.data.materials.append(plain)
        # the ring: a flat band with its pictures laid from inside (u 0) to outside (u 1)
        import bmesh
        bm = bmesh.new()
        uv = bm.loops.layers.uv.new()
        seg = 128
        inner = [bm.verts.new((1.45 * math.cos(2 * math.pi * i / seg), 1.45 * math.sin(2 * math.pi * i / seg), 0)) for i in range(seg)]
        outer = [bm.verts.new((2.15 * math.cos(2 * math.pi * i / seg), 2.15 * math.sin(2 * math.pi * i / seg), 0)) for i in range(seg)]
        for i in range(seg):
            j = (i + 1) % seg
            f = bm.faces.new((inner[i], outer[i], outer[j], inner[j]))
            for l, (u, v) in zip(f.loops, ((0, i / seg), (1, i / seg), (1, (i + 1) / seg), (0, (i + 1) / seg))):
                l[uv].uv = (u, v)
        me = bpy.data.meshes.new('ring')
        bm.to_mesh(me)
        bm.free()
        ring = bpy.data.objects.new('ring', me)
        coll.objects.link(ring)
        ring.data.materials.append(plain)
        # satellites: built afresh at the middle, panels open. Their panels' cells are a picture of their own
        # (panel_picture()), given to the material called sp-panel by the page: a picture inside the model file would
        # be fetched by a blob: address, which the Hub does not allow.
        panel_picture()
        for shape in SHAPES:
            body = satellite(shape, coll, at=(0, 0, 0))
            body.name = 'sat-' + shape
        for o in coll.objects:
            for m in o.modifiers:
                bpy.context.view_layer.objects.active = o
                bpy.ops.object.modifier_apply(modifier=m.name)
        # each satellite at the middle: spread them so the file opens tidily, the page puts each where it wants
        vl = bpy.context.view_layer
        for o in vl.objects:
            o.select_set(o.name in coll.all_objects)
        vl.objects.active = ball
        path = path or os.path.join(OUT, 'space.glb')
        bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True, export_yup=True,
                                  export_image_format='AUTO', export_cameras=False, export_lights=False, export_extras=False,
                                  export_animations=False, export_apply=True)
        return {'path': path, 'bytes': os.path.getsize(path), 'nodes': sorted(o.name for o in coll.objects if o.parent is None)}
    finally:
        win.scene = old
