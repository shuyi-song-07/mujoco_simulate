"""A rigid block-letter R with an open bowl and a real diagonal pushing leg."""
import math
import xml.etree.ElementTree as ET

import numpy as np

from simulation.mujoco.dual_arm.task_scenes import numbers
from tasks.adapters.tray import initial_state


def parts(scene):
    """Return nonoverlapping boxes and the lower convex leg in body coordinates."""
    hx, hy, hz = scene['object_half_size']
    stroke = scene['r_shape']['stroke_width_m']
    if (not all(math.isfinite(v) and v > 0 for v in (hx, hy, hz, stroke, scene['object_mass_kg']))
            or stroke >= min(hx, hy) or 2.3*stroke >= 2*hx):
        raise ValueError('R dimensions and mass must be positive and leave an open bowl')
    middle = scene['r_shape']['middle_bar_y_m']
    top_inner = hy - stroke
    middle_top, middle_bottom = middle + stroke/2, middle - stroke/2
    if not -hy + stroke < middle_bottom < middle_top < top_inner:
        raise ValueError('R middle bar must leave room for the bowl and diagonal leg')
    boxes = [((-hx + stroke/2, 0, 0), (stroke/2, hy, hz)),
             ((stroke/2, hy - stroke/2, 0), (hx - stroke/2, stroke/2, hz)),
             ((stroke/2, middle, 0), (hx - stroke/2, stroke/2, hz)),
             ((hx - stroke/2, (middle_top + top_inner)/2, 0),
              (stroke/2, (top_inner - middle_top)/2, hz))]
    # Horizontal top/bottom cuts meet the middle bar and table-facing foot
    # exactly, preserving the bowl and lower concavity in collision geometry.
    leg = [(-hx + stroke, middle_bottom),
           (-hx + 2.1*stroke, middle_bottom),
           (hx, -hy),
           (hx - 1.3*stroke, -hy)]
    return boxes, leg


def _polygon_area(points):
    return abs(sum(a[0]*b[1] - b[0]*a[1]
                   for a, b in zip(points, points[1:] + points[:1])))/2


def configure_xml(root, config):
    scene = config['scene']
    boxes, leg = parts(scene)
    hz = scene['object_half_size'][2]
    color = '0.17 0.20 0.23 1'
    asset = root.find('asset')
    ET.SubElement(asset, 'material', name='paper_matte',
                  specular='0', shininess='0', reflectance='0')
    root.find(".//geom[@name='table_geom']").set('material', 'paper_matte')
    for side in ('left', 'right'):
        x, y, z = scene[f'{side}_base_position']
        pedestal = root.find(f".//body[@name='{side}_pedestal']")
        pedestal.set('pos', numbers((x, y, z/2)))
        geom = pedestal.find('geom')
        size = np.fromstring(geom.get('size'), sep=' ')
        geom.set('size', numbers((size[0], size[1], z/2)))

    vertices = [(x, y, z) for z in (-hz, hz) for x, y in leg]
    ET.SubElement(asset, 'mesh', name='r_diagonal_leg',
                  vertex=numbers([v for point in vertices for v in point]))
    volumes = [8*math.prod(size) for _, size in boxes] + [_polygon_area(leg)*2*hz]
    volume = sum(volumes)
    body = root.find(".//body[@name='cooperative_object']")
    for geom in body.findall('geom'):
        body.remove(geom)
    for index, (position, size) in enumerate(boxes):
        ET.SubElement(body, 'geom', name='object_geom' if index == 0 else f'object_part_{index}',
                      type='box', pos=numbers(position), size=numbers(size),
                      mass=str(scene['object_mass_kg']*volumes[index]/volume),
                      condim='6', priority='1', friction=numbers(scene['object_friction']), rgba=color)
    ET.SubElement(body, 'geom', name='object_diagonal_leg', type='mesh', mesh='r_diagonal_leg',
                  mass=str(scene['object_mass_kg']*volumes[-1]/volume), condim='6', priority='1',
                  friction=numbers(scene['object_friction']), rgba=color)
    for side in ('left', 'right'):
        body.find(f"site[@name='{side}_grasp_site']").set('rgba', '0 0 0 0')

    target = root.find(".//body[@name='target']")
    for geom in target.findall('geom'):
        target.remove(geom)
    hx, hy, _ = scene['object_half_size']
    stroke = scene['r_shape']['stroke_width_m']
    middle = scene['r_shape']['middle_bar_y_m']
    # Trace the outside and the bowl opening with the H-frame's 5 mm outline.
    boundaries = [
        [(-hx, -hy), (-hx, hy), (hx, hy), (hx, middle-stroke/2),
         leg[1], leg[2], leg[3], leg[0], (-hx+stroke, -hy)],
        [(-hx+stroke, middle+stroke/2), (hx-stroke, middle+stroke/2),
         (hx-stroke, hy-stroke), (-hx+stroke, hy-stroke)],
    ]
    index = 0
    for boundary in boundaries:
        for start, end in zip(boundary, boundary[1:] + boundary[:1]):
            dx, dy = end[0]-start[0], end[1]-start[1]
            angle = math.atan2(dy, dx)/2
            ET.SubElement(target, 'geom', name='target_geom' if index == 0 else f'target_part_{index}',
                          type='box', pos=numbers(((start[0]+end[0])/2, (start[1]+end[1])/2, 0)),
                          size=numbers((math.hypot(dx, dy)/2+.0025, .0025, .0004)),
                          quat=numbers((math.cos(angle), 0, 0, math.sin(angle))),
                          rgba='0.14 0.95 0.66 0.8', contype='0', conaffinity='0')
            index += 1
    for side, offset in zip(('left', 'right'), scene['grasp_offsets']):
        for parent, prefix, z, rgba in ((body, 'frame', hz+.0005, '1 .68 .18 1'),
                                       (target, 'goal', .0005, '.25 1 .82 1')):
            ET.SubElement(parent, 'site', name=f'{prefix}_{side}_datum', type='cylinder',
                          pos=numbers((-hx+stroke/2, offset[1], z)),
                          size='0.009 0.0005', rgba=rgba)


def reset(controller, object_position, yaw):
    return {}
