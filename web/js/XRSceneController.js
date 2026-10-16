/**
 * XRSceneController: WebXR controller interaction for a splat / hybrid scene.
 *
 * Ported from img3dviewer's XRMeshController, keeping its button layout. Its
 * depth-mesh shader controls (z-scale, z-offset, pan, depth threshold, video)
 * have no equivalent for splats and are left out.
 *
 * MAIN CONTROLLER (right by default; swap in the menu):
 * - Thumbstick Up/Down: Move the scene forward/backward along your gaze.
 * - Index Trigger (0): Reset the scene in front of you, facing you.
 * - Grip (1): Hold to drag the scene in 3D.
 * - Thumbstick Click (3): Hold to attach the scene to the controller and rotate it.
 * - Button A/X (4): [Short press] Next scene; [Hold 1 s] Reload current scene.
 * - Button B/Y (5): Previous scene.
 *
 * SECONDARY CONTROLLER:
 * - Thumbstick Up/Down: Scale the scene.
 * - Button B/Y (5): Open/close the settings menu.
 *
 * SETTINGS MENU (thumbstick to navigate, trigger to select):
 * - Swap main/secondary hand, main on/off, secondary on/off,
 *   delete current scene (press trigger twice to confirm).
 */
import * as THREE from "three";

export class XRSceneController {
  /**
   * @param {THREE.WebGLRenderer} renderer
   * @param {THREE.Camera} camera
   * @param {THREE.Scene} scene
   * @param {object} options
   *   group:   object placed/moved/scaled in the world (the scene root).
   *   content: child of `group` holding the scene; attached to the controller to rotate it.
   *   FontLoader, TextGeometry: three.js addons, for the settings menu text.
   *   onStepScene(delta), onReloadScene(), onDeleteScene(): scene callbacks.
   */
  constructor(renderer, camera, scene, options = {}) {
    this.renderer = renderer;
    this.camera = camera;
    this.scene = scene;
    this.group = options.group;
    this.content = options.content;
    this.FontLoader = options.FontLoader || null;
    this.TextGeometry = options.TextGeometry || null;

    this.onStepScene = options.onStepScene || (() => {});
    this.onReloadScene = options.onReloadScene || (() => {});
    this.onDeleteScene = options.onDeleteScene || (() => {});

    this.minScale = 0.05;
    this.maxScale = 20;

    // Button state per hand ([left, right]) and xr-standard button index.
    this.conts = [null, null];
    this.butts = [0, 1].map(() =>
      Array(12).fill(null).map(() => ({ clicked: false, unclicked: false, pressed: false })));
    this.mEnabled = true;
    this.sEnabled = true;
    this.motionInd = 1;  // main hand: 0 = left, 1 = right

    this.lastGroupPos = null;
    this.lastContPos = null;

    this.menuGroup = new THREE.Group();
    this.menuGroup.visible = false;
    this.scene.add(this.menuGroup);
    this.menuIndex = 0;
    this.confirmDelete = false;
    this._font = null;
    this._stickMoved = false;

    this._initControllers();
  }

  _initControllers() {
    [this.renderer.xr.getController(0), this.renderer.xr.getController(1)].forEach((controller) => {
      controller.addEventListener("connected", (event) => {
        if (event.data.handedness === "right") this.conts[1] = controller;
        else if (event.data.handedness === "left") this.conts[0] = controller;
      });
      controller.addEventListener("disconnected", () => {
        if (controller === this.conts[1]) this.conts[1] = null;
        else if (controller === this.conts[0]) this.conts[0] = null;
      });
      this.scene.add(controller);
    });
  }

  // --- Settings menu ---------------------------------------------------------

  _loadFont() {
    if (this._font || this._fontLoading || !this.FontLoader) return;
    this._fontLoading = true;
    new this.FontLoader().load(
      "https://cdn.jsdelivr.net/npm/three@0.178.0/examples/fonts/helvetiker_regular.typeface.json",
      (font) => { this._font = font; this._buildMenu(); },
      undefined,
      (e) => console.warn("XR menu font failed to load:", e));
  }

  _buildMenu() {
    while (this.menuGroup.children.length) {
      const child = this.menuGroup.children[0];
      this.menuGroup.remove(child);
      child.geometry?.dispose();
      child.material?.dispose();
    }
    this.menuGroup.add(new THREE.Mesh(
      new THREE.PlaneGeometry(0.42, 0.3),
      new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.8 })));
    if (!this._font || !this.TextGeometry) { this._loadFont(); return; }

    const text = (str, y, selected) => {
      const mesh = new THREE.Mesh(
        new this.TextGeometry(str, { font: this._font, size: 0.018, depth: 0.001 }),
        new THREE.MeshBasicMaterial({ color: selected ? 0xffff00 : 0xffffff }));
      mesh.position.set(-0.19, y, 0.01);
      this.menuGroup.add(mesh);
    };
    const mLabel = this.motionInd === 0 ? "Left" : "Right";
    text("--- VR SETTINGS ---", 0.1, false);
    text(`1. Main hand: ${mLabel}`, 0.05, this.menuIndex === 0);
    text(`2. Main: ${this.mEnabled ? "ON" : "OFF"}`, 0.0, this.menuIndex === 1);
    text(`3. Secondary: ${this.sEnabled ? "ON" : "OFF"}`, -0.05, this.menuIndex === 2);
    text(this.confirmDelete ? "4. Delete? Trigger again!" : "4. Delete current scene", -0.10, this.menuIndex === 3);
    text("[Trigger: select | Stick: navigate]", -0.135, false);
  }

  toggleMenu() {
    this.menuGroup.visible = !this.menuGroup.visible;
    this.confirmDelete = false;
    if (this.menuGroup.visible) {
      const cam = this.renderer.xr.getCamera();
      const pos = new THREE.Vector3(), quat = new THREE.Quaternion();
      cam.getWorldPosition(pos);
      cam.getWorldQuaternion(quat);
      this.menuGroup.position.copy(pos).add(new THREE.Vector3(0, 0, -0.5).applyQuaternion(quat));
      this.menuGroup.lookAt(pos);
      this._buildMenu();
    }
  }

  // --- Scene placement ------------------------------------------------------

  /** Place the scene at your head, facing your gaze (yaw only), and undo any hand rotation. */
  resetPlacement() {
    if (!this.group) return;
    const cam = this.renderer.xr.getCamera();
    const pos = new THREE.Vector3(), quat = new THREE.Quaternion();
    cam.getWorldPosition(pos);
    cam.getWorldQuaternion(quat);
    const euler = new THREE.Euler(0, 0, 0, "YXZ").setFromQuaternion(quat);
    this.group.position.copy(pos);
    this.group.quaternion.setFromEuler(new THREE.Euler(0, euler.y, 0, "YXZ"));
    if (this.content && this.content.parent === this.group) {
      this.content.position.set(0, 0, 0);
      this.content.quaternion.identity();
    }
  }

  /** Restore everything to the desktop pose (on session end). */
  resetAll() {
    if (this.content && this.content.parent !== this.group) this.group.attach(this.content);
    this.group.position.set(0, 0, 0);
    this.group.quaternion.identity();
    this.group.scale.setScalar(1);
    this.content.position.set(0, 0, 0);
    this.content.quaternion.identity();
    this.menuGroup.visible = false;
  }

  // --- Per-frame update -----------------------------------------------------

  update() {
    const session = this.renderer.xr.getSession();
    if (!session) return;

    const motionUD = [0, 0];
    for (const source of session.inputSources) {
      const hand = source.handedness === "left" ? 0 : source.handedness === "right" ? 1 : -1;
      if (hand < 0 || !source.gamepad || !source.gamepad.buttons) continue;
      if (source.gamepad.axes[3]) motionUD[hand] = source.gamepad.axes[3];
      source.gamepad.buttons.forEach((button, i) => {
        const b = this.butts[hand][i];
        if (!b) return;
        b.clicked = button.pressed && !b.pressed;
        b.unclicked = !button.pressed && b.pressed;
        b.pressed = button.pressed;
      });
    }

    const mInd = this.motionInd, sInd = 1 - mInd;
    const mCont = this.conts[mInd];
    const mUD = motionUD[mInd], sUD = motionUD[sInd];
    const mButt = this.butts[mInd], sButt = this.butts[sInd];

    // Settings menu (secondary B/Y).
    if (sButt[5].clicked) this.toggleMenu();
    if (this.menuGroup.visible) {
      if (sButt[0].clicked || mButt[0].clicked) {
        if (this.menuIndex === 0) this.motionInd = 1 - this.motionInd;
        else if (this.menuIndex === 1) this.mEnabled = !this.mEnabled;
        else if (this.menuIndex === 2) this.sEnabled = !this.sEnabled;
        else if (this.menuIndex === 3) {
          if (this.confirmDelete) {
            this.toggleMenu();
            this.onDeleteScene();
            return;
          }
          this.confirmDelete = true;
        }
        this._buildMenu();
      }
      if (Math.abs(sUD) > 0.5 && !this._stickMoved) {
        this.menuIndex = (this.menuIndex + (sUD > 0 ? 1 : -1) + 4) % 4;
        this.confirmDelete = false;
        this._stickMoved = true;
        this._buildMenu();
      } else if (Math.abs(sUD) < 0.1) {
        this._stickMoved = false;
      }
      return;  // other inputs are blocked while the menu is open
    }

    // Secondary: thumbstick scales the scene (around its placement point).
    if (this.sEnabled && this.group && sUD) {
      const s = THREE.MathUtils.clamp(this.group.scale.x * (1 - sUD * 0.02), this.minScale, this.maxScale);
      this.group.scale.setScalar(s);
    }

    if (!this.mEnabled || !this.group) return;

    // Main: thumbstick moves the scene along your gaze.
    if (mUD) {
      const dir = new THREE.Vector3();
      this.renderer.xr.getCamera().getWorldDirection(dir);
      this.group.position.addScaledVector(dir, -mUD * 0.01);
    }

    // Main: trigger resets the scene in front of you.
    if (mButt[0].pressed) this.resetPlacement();

    if (mCont) {
      // Grip: drag.
      if (mButt[1].pressed) {
        if (this.lastContPos === null) {
          this.lastContPos = mCont.position.clone();
          this.lastGroupPos = this.group.position.clone();
        }
        this.group.position.copy(this.lastGroupPos).add(new THREE.Vector3().subVectors(mCont.position, this.lastContPos));
      } else {
        this.lastContPos = null;
        this.lastGroupPos = null;
      }
      // Thumbstick click: hold to rotate the scene with the controller.
      if (this.content) {
        if (mButt[3].clicked) mCont.attach(this.content);
        else if (mButt[3].unclicked) this.group.attach(this.content);
      }
    }

    // A/X: short press = next scene, hold 1 s = reload. B/Y: previous scene.
    const a = mButt[4];
    if (a.pressed && !a.processed) {
      a.pressedTime ??= Date.now();
      if (Date.now() - a.pressedTime > 1000) {
        a.processed = true;
        this.onReloadScene();
      }
    }
    if (a.unclicked) {
      if (!a.processed) this.onStepScene(1);
      a.pressedTime = undefined;
      a.processed = false;
    }
    if (mButt[5].unclicked) this.onStepScene(-1);
  }
}
