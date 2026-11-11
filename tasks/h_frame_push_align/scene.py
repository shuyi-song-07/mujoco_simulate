"""Three physical beams on the existing Panda table, with visual alignment marks."""
import math
import xml.etree.ElementTree as ET

import numpy as np

from simulation.mujoco.dual_arm.task_scenes import numbers
from tasks.adapters.tray import initial_state


def configure_xml(root, config):
    scene = config['scene']
    half_x, half_y, half_z = scene['object_half_size']
    settings = scene['push_alignment']
    rail = settings['rail_width_m']
    bridge = settings['crossbeam_width_m']
    offset = settings['crossbeam_offset_x_m']
    mass = scene['object_mass_kg']
    if (not all(math.isfinite(v) and v > 0 for v in (half_x, half_y, half_z, rail, bridge, mass))
            or not rail < half_y or not abs(offset) + bridge/2 < half_x):
        raise ValueError('H-frame beams must fit within the frame bounds')

    # Match the high-wall tray's table finish and pedestals. The shared model
    # factory already sets table size, robot bases, actuators and cameras.
    ET.SubElement(root.find('asset'), 'material', name='paper_matte',
                  specular='0', shininess='0', reflectance='0')
    root.find(".//geom[@name='table_geom']").set('material', 'paper_matte')
    for side in ('left', 'right'):
        x, y, z = scene[f'{side}_base_position']
        pedestal = root.find(f".//body[@name='{side}_pedestal']")
        pedestal.set('pos', numbers((x, y, z/2)))
        geom = pedestal.find('geom')
        size = np.fromstring(geom.get('size'), sep=' ')
        geom.set('size', numbers((size[0], size[1], z/2)))

    body = root.find(".//body[@name='cooperative_object']")
    rail_y = half_y - rail/2
    beams = [((0, -rail_y, 0), (half_x, rail/2, half_z)),
             ((0, rail_y, 0), (half_x, rail/2, half_z)),
             ((offset, 0, 0), (bridge/2, half_y-rail, half_z))]
    volume = sum(math.prod(size) for _, size in beams)
    for geom in body.findall('geom'):
        body.remove(geom)
    for index, (position, size) in enumerate(beams):
        ET.SubElement(body, 'geom', name='object_geom' if index == 0 else f'object_part_{index}',
                      type='box', pos=numbers(position), size=numbers(size),
                      mass=str(mass*math.prod(size)/volume), condim='6', priority='1',
                      friction=numbers(scene['object_friction']), rgba='0.17 0.20 0.23 1')
    for side in ('left', 'right'):
        body.find(f"site[@name='{side}_grasp_site']").set('rgba', '0 0 0 0')

    target = root.find(".//body[@name='target']")
    target.remove(target.find('geom'))
    inner_y = half_y - rail
    bridge_min, bridge_max = offset - bridge/2, offset + bridge/2
    boundary = [(-half_x, -half_y), (half_x, -half_y), (half_x, -inner_y),
                (bridge_max, -inner_y), (bridge_max, inner_y), (half_x, inner_y),
                (half_x, half_y), (-half_x, half_y), (-half_x, inner_y),
                (bridge_min, inner_y), (bridge_min, -inner_y), (-half_x, -inner_y)]
    for index, (start, end) in enumerate(zip(boundary, boundary[1:] + boundary[:1])):
        center = [(a+b)/2 for a, b in zip(start, end)]
        half_size = [abs(a-b)/2 + .0025 for a, b in zip(start, end)]
        ET.SubElement(target, 'geom', name='target_geom' if index == 0 else f'target_part_{index}',
                      type='box', pos=numbers((*center, 0)), size=numbers((*half_size, .0004)),
                      rgba='0.14 0.95 0.66 0.8', contype='0', conaffinity='0')
    for side, y in (('left', -rail_y), ('right', rail_y)):
        for parent, prefix, z, color in ((body, 'frame', half_z+.0005, '1 .68 .18 1'),
                                          (target, 'goal', .0005, '.25 1 .82 1')):
            ET.SubElement(parent, 'site', name=f'{prefix}_{side}_datum', type='cylinder',
                          pos=numbers((half_x*.58, y, z)), size='0.009 0.0005', rgba=color)


def reset(controller, object_position, yaw):
    """Shared reset places the free body and target; there are no obstacles."""
    return {}
