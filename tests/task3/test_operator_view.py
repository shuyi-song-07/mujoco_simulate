"""Operator translucency changes previews only; raw training images stay intact."""
from __future__ import annotations

import unittest
from contextlib import contextmanager

import mujoco
import numpy as np

from simulation.mujoco.dual_arm.config import load_config, config_hash
from simulation.mujoco.dual_arm.dual_arm_task_controller import DualArmTaskController, create_model
from simulation.mujoco.dual_arm.dual_arm_render_workers import (
    MultiCameraRenderer, apply_operator_view, robot_geom_ids, sync_operator_view,
)


class FakePassiveViewer:
    """Exercise MuJoCo's real scene update at the synchronous-copy boundary."""
    def __init__(self, controller, *, fail=False):
        self.controller = controller
        self.cam, self.opt = mujoco.MjvCamera(), mujoco.MjvOption()
        self.scene = mujoco.MjvScene(controller.model, maxgeom=1000)
        self.locked = False
        self.fail = fail

    @contextmanager
    def lock(self):
        if self.locked:
            raise AssertionError("viewer lock is not re-entrant")
        self.locked = True
        try:
            yield
        finally:
            self.locked = False

    def sync(self):
        if self.locked:
            raise AssertionError("sync must acquire its own viewer lock")
        c = self.controller
        with self.lock():
            mujoco.mjv_updateScene(c.model, c.data, self.opt, None,
                                  self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            if self.fail:
                raise RuntimeError("simulated viewer synchronization failure")


class OperatorViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.controller = DualArmTaskController()

    def setUp(self):
        self.controller.reset()

    def scene(self, camera="top", option=None):
        c = self.controller
        scene = mujoco.MjvScene(c.model, maxgeom=1000)
        view = mujoco.MjvCamera()
        view.type = mujoco.mjtCamera.mjCAMERA_FIXED
        view.fixedcamid = c.model.camera(camera).id
        mujoco.mjv_updateScene(c.model, c.data, option or mujoco.MjvOption(), None,
                              view, mujoco.mjtCatBit.mjCAT_ALL, scene)
        return scene

    def test_orthographic_top_keeps_world_scale_and_expected_axes(self):
        c = self.controller
        camera = c.model.camera("top").id
        self.assertEqual(c.model.cam_projection[camera], mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC)
        self.assertAlmostEqual(c.model.cam_fovy[camera], 0.62)
        rotation = c.data.cam_xmat[camera].reshape(3, 3)
        np.testing.assert_allclose(rotation[:, 0], [0, 1, 0], atol=1e-8)
        np.testing.assert_allclose(rotation[:, 1], [-1, 0, 0], atol=1e-8)
        np.testing.assert_allclose(rotation[:, 2], [0, 0, 1], atol=1e-8)

    def test_central_front_separates_both_open_fingers_across_pick_and_place_region(self):
        c = self.controller
        camera = c.model.camera("front").id
        self.assertEqual(c.model.cam_projection[camera], mujoco.mjtProjection.mjPROJ_PERSPECTIVE)
        position = c.data.cam_xpos[camera]
        self.assertGreater(position[0], max(c.config["scene"][f"{side}_workspace"][0][1]
                                          for side in ("left", "right")))
        rotation = c.data.cam_xmat[camera].reshape(3, 3)
        focal = 240 / np.tan(np.deg2rad(c.model.cam_fovy[camera]) / 2)
        # Include start/end XY randomization and the open 8 cm finger width.
        # This distinguishes the new central view from the old orthographic
        # +X view, in which the two fingers projected on top of each other.
        for x in (-0.015, 0.165):
            for y in (-0.235, -0.205, 0.205, 0.235):
                for z in (0.36, 0.55):
                    pixels = []
                    for finger_x in (x-0.04, x+0.04):
                        local = rotation.T @ (np.array([finger_x, y, z])-position)
                        self.assertLess(local[2], 0)
                        pixel = np.array([320, 240]) + focal * local[:2] / -local[2] * [1, -1]
                        self.assertTrue(8 < pixel[0] < 632 and 8 < pixel[1] < 472,
                                        f"Finger cropped at world point {(finger_x, y, z)}")
                        pixels.append(pixel)
                    self.assertGreater(abs(pixels[0][0]-pixels[1][0]), 12,
                                       "Both open fingers must be visibly separated horizontally")

    def test_frozen_old_config_keeps_projection_hash_and_opaque_robot(self):
        old = load_config()
        old.pop("operator_view"); old["scene"].pop("table_half_size_xy")
        for cam in old["scene"]["cameras"].values():
            cam.pop("projection"); cam["fovy"] = 5
        before = config_hash(old)
        model = create_model(old)
        self.assertTrue(all(model.cam_projection == mujoco.mjtProjection.mjPROJ_PERSPECTIVE))
        np.testing.assert_allclose(model.cam_fovy, 5)
        self.assertEqual(config_hash(load_config(old)), before)
        self.assertNotIn("operator_view", load_config(old))
        scene = self.scene()
        original = {g.objid: g.rgba.copy() for g in scene.geoms[:scene.ngeom]
                    if g.objtype == mujoco.mjtObj.mjOBJ_GEOM}
        apply_operator_view(scene, self.controller.model, "top", old)
        for geom in scene.geoms[:scene.ngeom]:
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM:
                np.testing.assert_array_equal(geom.rgba, original[geom.objid])

    def test_table_footprint_and_invalid_projection_parameters(self):
        config = load_config({"scene": {"table_half_size_xy": [0.62, 0.6]}})
        model = create_model(config)
        np.testing.assert_allclose(model.geom("table_geom").size, [0.62, 0.6, 0.175])
        for patch in ({"scene": {"table_half_size_xy": [0, 0.5]}},
                      {"scene": {"cameras": {"top": {"projection": "unknown"}}}},
                      {"scene": {"cameras": {"side": {"fovy": 180}}}}):
            with self.assertRaises(ValueError):
                load_config(patch)

    def test_top_transparency_only_affects_robot_and_hides_diagnostic_sites(self):
        c = self.controller
        config = {"operator_view": {"top_robot_alpha": 0.22}}
        roots = {c.model.body(name).id for name in ("left_link0", "right_link0")}
        for camera in ("top", "front", "side"):
            scene = self.scene(camera)
            before = [g.rgba.copy() for g in scene.geoms[:scene.ngeom]]
            apply_operator_view(scene, c.model, camera, config)
            transparent_robot = 0
            hidden_sites = 0
            for index, geom in enumerate(scene.geoms[:scene.ngeom]):
                if geom.objtype == mujoco.mjtObj.mjOBJ_SITE:
                    self.assertEqual(geom.rgba[3], 0)
                    hidden_sites += 1
                elif geom.objtype == mujoco.mjtObj.mjOBJ_GEOM:
                    root = c.model.body_rootid[c.model.geom_bodyid[geom.objid]]
                    if camera == "top" and root in roots:
                        self.assertAlmostEqual(geom.rgba[3], 0.22)
                        self.assertTrue(geom.transparent)
                        transparent_robot += 1
                    else:
                        np.testing.assert_array_equal(geom.rgba, before[index])
            self.assertEqual(hidden_sites, 4)
            if camera == "top":
                self.assertGreater(transparent_robot, 80)
            else:
                self.assertEqual(transparent_robot, 0)

    def test_operator_view_never_reveals_disabled_or_already_hidden_geometry(self):
        c = self.controller
        option = mujoco.MjvOption()
        option.geomgroup[2] = 0
        option.geomgroup[3] = 1
        scene = self.scene(option=option)
        visible_ids = [g.objid for g in scene.geoms[:scene.ngeom]
                       if g.objtype == mujoco.mjtObj.mjOBJ_GEOM]
        self.assertFalse(any(c.model.geom_group[i] == 2 for i in visible_ids))
        robot = next(g for g in scene.geoms[:scene.ngeom]
                     if g.objtype == mujoco.mjtObj.mjOBJ_GEOM
                     and c.model.geom(g.objid).name.startswith("left_geom_"))
        robot.rgba[3] = 0
        apply_operator_view(scene, c.model, "top", {"operator_view": {"top_robot_alpha": 0.22}})
        self.assertEqual(robot.rgba[3], 0)
        self.assertEqual(visible_ids, [g.objid for g in scene.geoms[:scene.ngeom]
                                      if g.objtype == mujoco.mjtObj.mjOBJ_GEOM])

    def test_preview_render_restores_raw_rgb_and_leaves_model_and_physics_untouched(self):
        c = self.controller
        before = {name: getattr(c.model, name).copy()
                  for name in ("geom_rgba", "site_rgba", "mat_rgba", "geom_group")}
        qpos, qvel, ctrl = c.data.qpos.copy(), c.data.qvel.copy(), c.data.ctrl.copy()
        renderer = MultiCameraRenderer(c.model)
        try:
            raw_before = renderer.render(c.data)
            opaque = renderer.render(c.data, operator_view={"operator_view": {"top_robot_alpha": 1}})
            preview = renderer.render(c.data, operator_view={"operator_view": {"top_robot_alpha": 0.22}})
            raw_after = renderer.render(c.data)
        finally:
            renderer.close()
        self.assertGreater(np.count_nonzero(preview["top"] != opaque["top"]), 1000)
        def same_rgb(actual, expected):
            # GPU multisampling can round a handful of channels by one level.
            error = np.abs(actual.astype(int)-expected.astype(int))
            self.assertLessEqual(error.max(), 1)
            self.assertLess(np.count_nonzero(error), 32)
        for camera in ("front", "side"):
            same_rgb(preview[camera], opaque[camera])
        for camera in ("top", "front", "side"):
            same_rgb(raw_after[camera], raw_before[camera])
        for name, values in before.items():
            np.testing.assert_array_equal(getattr(c.model, name), values)
        for values, original in ((c.data.qpos, qpos), (c.data.qvel, qvel), (c.data.ctrl, ctrl)):
            np.testing.assert_array_equal(values, original)

    def test_native_sync_copies_top_alpha_and_restores_model_for_front_and_old_config(self):
        c = self.controller
        viewer = FakePassiveViewer(c)
        original = c.model.geom_rgba.copy()
        material_ids, materials = c.model.geom_matid.copy(), c.model.mat_rgba.copy()
        robot_ids = set(robot_geom_ids(c.model))
        scene = self.scene()
        baseline = {g.objid: g.rgba.copy() for g in scene.geoms[:scene.ngeom]
                    if g.objtype == mujoco.mjtObj.mjOBJ_GEOM}
        for camera, config, alpha in (("top", {"operator_view": {"top_robot_alpha": .22}}, .22),
                                      ("front", {"operator_view": {"top_robot_alpha": .22}}, 1),
                                      ("side", {"operator_view": {"top_robot_alpha": .22}}, 1),
                                      ("top", {}, 1)):
            sync_operator_view(viewer, c.model, camera, config)
            self.assertEqual(viewer.cam.fixedcamid, c.model.camera(camera).id)
            self.assertFalse(any(g.objtype == mujoco.mjtObj.mjOBJ_SITE
                                 for g in viewer.scene.geoms[:viewer.scene.ngeom]))
            seen = 0
            for geom in viewer.scene.geoms[:viewer.scene.ngeom]:
                if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM:
                    continue
                np.testing.assert_array_equal(geom.rgba[:3], baseline[geom.objid][:3])
                if geom.objid in robot_ids:
                    self.assertAlmostEqual(geom.rgba[3], alpha)
                    seen += 1
                else:
                    np.testing.assert_array_equal(geom.rgba, baseline[geom.objid])
            self.assertGreater(seen, 80)
            np.testing.assert_array_equal(c.model.geom_rgba, original)
            np.testing.assert_array_equal(c.model.geom_matid, material_ids)
            np.testing.assert_array_equal(c.model.mat_rgba, materials)

    def test_native_sync_failure_restores_visual_attributes_and_lock(self):
        c = self.controller
        viewer = FakePassiveViewer(c, fail=True)
        original = c.model.geom_rgba.copy()
        with self.assertRaisesRegex(RuntimeError, "synchronization failure"):
            sync_operator_view(viewer, c.model, "top", {"operator_view": {"top_robot_alpha": .22}})
        np.testing.assert_array_equal(c.model.geom_rgba, original)
        self.assertFalse(viewer.locked)

    def test_alpha_config_rejects_invalid_values_and_accepts_endpoints(self):
        for alpha in (-0.01, 1.01, float("nan"), float("inf"), "0.22", True, None):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                load_config({"operator_view": {"top_robot_alpha": alpha}})
        for alpha in (0, .22, 1):
            self.assertEqual(load_config({"operator_view": {"top_robot_alpha": alpha}})
                             ["operator_view"]["top_robot_alpha"], alpha)


if __name__ == "__main__":
    unittest.main()
