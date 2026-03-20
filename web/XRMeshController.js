/**
 * XRMeshController handles interaction logic for a 3D mesh in WebXR.
 * 
 * MAIN CONTROLLER (Movement & Navigation):
 * - Thumbstick Up/Down: Move mesh group forward/backward along viewer's gaze.
 * - Index Trigger (0): Reset mesh to center/facing camera.
 * - Grip/Middle Trigger (1): Drag mesh group in 3D space.
 * - Thumbstick Click (3): Attach/Detach mesh to controller for manual rotation.
 * - Button A/X (4): [Short Press] Next image; [Long Press 1s] Reload mesh.
 * - Button B/Y (5): Previous image.
 * 
 * SECONDARY CONTROLLER (Settings & Refinement):
 * - Thumbstick Up/Down: Adjust current slider parameter (Scale, Z-Scale, etc.).
 * - Index Trigger (0): Hold & move controller to adjust Z-Offset (depth).
 * - Grip/Middle Trigger (1): Hold & move controller to adjust Z-Scale (3D intensity).
 * - Thumbstick Click (3): Hold & move controller to adjust Offset X/Y (panning).
 * - Button A/X (4): Toggle video play/pause.
 * - Button B/Y (5): Open/Close Settings Menu.
 * 
 * SETTINGS MENU:
 * - Opened via Secondary Button B/Y (5).
 * - Thumbstick Up/Down: Navigate options.
 * - Index Trigger (0): Toggle/Change selected option.
 * - Options: Swap Role (Left/Right), Toggle Main Enabled, Toggle Secondary Enabled.
 */
import * as THREE from "three";

export class XRMeshController {
  constructor(renderer, camera, scene, options = {}) {
    this.renderer = renderer;
    this.camera = camera;
    this.scene = scene;

    this.group = options.group || null;
    this.mesh = options.mesh || null;
    this.material = options.material || null;
    this.FontLoader = options.FontLoader || (THREE.FontLoader ? THREE.FontLoader : null);
    this.TextGeometry = options.TextGeometry || (THREE.TextGeometry ? THREE.TextGeometry : null);

    // Parameters
    this.config = {
      zoffset: options.zoffset || -2.4,
      ox: options.ox || 0,
      oy: options.oy || 0,
      zscale: options.zscale || 1.4,
      scale: options.scale || 0.45,
      depthThreshold: options.depthThreshold || -0.01,
      basename: options.basename || ""
    };

    // Callbacks
    this.onConfigUpdate = options.onConfigUpdate || (() => { });
    this.onNextImage = options.onNextImage || (() => { });
    this.onLoadMesh = options.onLoadMesh || (() => { });
    this.onToggleVideo = options.onToggleVideo || (() => { });

    // Internal State
    this.conts = [null, null];
    this.butts = [
      Array(12).fill(null).map(() => ({ clicked: false, unclicked: false, pressed: false })),
      Array(12).fill(null).map(() => ({ clicked: false, unclicked: false, pressed: false }))
    ];
    this.mEnabled = true;
    this.sEnabled = true;
    this.motionInd = 1;

    this.sliderModes = ["scale", "z-scale", "z-offset", "oy", "depthT"];
    this.sliderMode = 0;

    this.lastMeshPos = null;
    this.lastContPos = null;
    this.lastZOffsetPos = null;
    this.lastZScalePos = null;
    this.lastOyPos = null;

    this.setInitHeight = false;
    this.photoMeshes = options.photoMeshes || [];

    this.menuGroup = new THREE.Group();
    this.menuGroup.visible = false;
    this.scene.add(this.menuGroup);
    this.menuIndex = 0;

    this._initControllers();
  }

  _createMenu() {
    // Clear old menu
    while(this.menuGroup.children.length > 0) {
      this.menuGroup.remove(this.menuGroup.children[0]);
    }

    const bg = new THREE.Mesh(
      new THREE.PlaneGeometry(0.4, 0.3),
      new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.8 })
    );
    this.menuGroup.add(bg);

    if (!this.FontLoader || !this.TextGeometry) {
      console.error("FontLoader or TextGeometry not provided to XRMeshController");
      return;
    }

    const loader = new this.FontLoader();
    loader.load('https://threejs.org/examples/fonts/helvetiker_regular.typeface.json', (font) => {
      const createText = (str, y, selected) => {
        const geom = new this.TextGeometry(str, { font: font, size: 0.02, height: 0.001 });
        const mat = new THREE.MeshBasicMaterial({ color: selected ? 0xffff00 : 0xffffff });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.position.set(-0.18, y, 0.01);
        this.menuGroup.add(mesh);
      };

      const mLabel = this.motionInd === 0 ? "Left" : "Right";
      const sLabel = this.motionInd === 0 ? "Right" : "Left";

      createText("--- VR SETTINGS ---", 0.1, false);
      createText(`1. Main Role: ${mLabel}`, 0.05, this.menuIndex === 0);
      createText(`2. Main: ${this.mEnabled ? "ON" : "OFF"}`, 0.0, this.menuIndex === 1);
      createText(`3. Secondary: ${this.sEnabled ? "ON" : "OFF"}`, -0.05, this.menuIndex === 2);
      createText("[Trigger to Toggle | Stick to Nav]", -0.12, false);
    });
  }

  toggleMenu() {
    this.menuGroup.visible = !this.menuGroup.visible;
    if (this.menuGroup.visible) {
      const cam = this.renderer.xr.getCamera(this.camera);
      const dir = new THREE.Vector3(0, 0, -0.5);
      dir.applyQuaternion(cam.quaternion);
      this.menuGroup.position.copy(cam.position).add(dir);
      this.menuGroup.lookAt(cam.position);
      this._createMenu();
    }
  }

  _initControllers() {
    const controllers = [this.renderer.xr.getController(0), this.renderer.xr.getController(1)];

    controllers.forEach((controller, index) => {
      controller.addEventListener('connected', event => {
        if (event.data.handedness === 'right') {
          this.conts[1] = controller;
        } else if (event.data.handedness === 'left') {
          this.conts[0] = controller;
        }
      });

      controller.addEventListener('disconnected', event => {
        if (controller === this.conts[1]) {
          this.conts[1] = undefined;
        } else if (controller === this.conts[0]) {
          this.conts[0] = undefined;
        }
      });
      this.scene.add(controller);
    });
  }

  setTarget(group, mesh, splat, material) {
    this.group = group;
    this.mesh = mesh;
    this.splat = splat;
    this.material = material;
    if (this.group) {
      this.group.scale.set(this.config.scale, this.config.scale, this.config.scale);
    }
    this.updateShaderUniforms();
  }

  updateShaderUniforms() {
    if (!this.material || !this.material.userData.shader) return;
    const shader = this.material.userData.shader;
    shader.uniforms.zscale.value = this.config.zscale;
    shader.uniforms.zoffset.value = this.config.zoffset;
    shader.uniforms.oy.value = this.config.oy;
    shader.uniforms.ox.value = this.config.ox;
    shader.uniforms.depthThreshold.value = this.config.depthThreshold;
  }

  setConfig(config) {
    Object.assign(this.config, config);
    if (this.group) {
      this.group.scale.set(this.config.scale, this.config.scale, this.config.scale);
    }
    this.updateShaderUniforms();
  }

  zeroCamera() {
    if (!this.group) return;
    const cam = this.renderer.xr.getCamera(this.camera);
    const leftEye = cam.cameras[0];
    const rightEye = cam.cameras[1];
    const avgPosition = new THREE.Vector3().addVectors(leftEye.position, rightEye.position).multiplyScalar(0.5);
    this.group.position.copy(avgPosition);

    const euler = new THREE.Euler(0, 0, 0, 'YXZ');
    euler.setFromQuaternion(cam.quaternion);
    euler.z = 0;
    euler.x = 0;
    const noRollQuaternion = new THREE.Quaternion().setFromEuler(euler);
    this.group.quaternion.copy(noRollQuaternion);
  }

  update() {
    const session = this.renderer.xr.getSession();
    if (!session) return;

    let motionUD = [0, 0];
    let motionLR = [0, 0];

    for (let index = 0; index < session.inputSources.length; index++) {
      const source = session.inputSources[index];
      let cont = -1;
      if (source.handedness === "left") cont = 0;
      else if (source.handedness === "right") cont = 1;
      if (cont == -1) continue;

      if (!source.gamepad || !source.gamepad.buttons) continue;

      if (source.gamepad.axes[3] !== 0) motionUD[cont] = source.gamepad.axes[3];
      if (source.gamepad.axes[2] !== 0) motionLR[cont] = source.gamepad.axes[2];

      source.gamepad.buttons.forEach((button, i) => {
        const bci = this.butts[cont][i];
        if (!bci) return;
        if (button.pressed) {
          bci.clicked = !bci.pressed;
          bci.unclicked = false;
        } else {
          bci.clicked = false;
          bci.unclicked = bci.pressed;
        }
        bci.pressed = button.pressed;
      });
    }

    const mInd = this.motionInd;
    const sInd = 1 - mInd;
    const mCont = this.conts[mInd];
    const sCont = this.conts[sInd];
    const mUD = motionUD[mInd];
    const sUD = motionUD[sInd];
    const mButt = this.butts[mInd];
    const sButt = this.butts[sInd];

    if (this.setInitHeight) {
      this.zeroCamera();
      this.setInitHeight = false;
    }

    // VR Settings Menu Toggle (Secondary Button 5)
    if (sButt[5].clicked) {
      this.toggleMenu();
    }

    // Handle Menu Interaction
    if (this.menuGroup.visible) {
      if (sButt[0].clicked || mButt[0].clicked) { // Toggle current setting
        if (this.menuIndex === 0) this.motionInd = 1 - this.motionInd;
        else if (this.menuIndex === 1) this.mEnabled = !this.mEnabled;
        else if (this.menuIndex === 2) this.sEnabled = !this.sEnabled;
        this._createMenu();
      }
      if (Math.abs(sUD) > 0.5 && !this._stickMoved) { // Navigate
        this.menuIndex = (this.menuIndex + (sUD > 0 ? 1 : -1) + 3) % 3;
        this._stickMoved = true;
        this._createMenu();
      } else if (Math.abs(sUD) < 0.1) {
        this._stickMoved = false;
      }
      return; // Block other inputs while menu is open
    }

    if (this.sEnabled && this.material && this.material.userData.shader) {
      if (sButt[4].clicked) this.onToggleVideo();
      const shader = this.material.userData.shader;
      if (typeof sUD === "number" && sUD !== 0) {
        const mode = this.sliderModes[this.sliderMode];
        if (mode == "z-scale") this.config.zscale -= sUD * 0.01;
        else if (mode == "z-offset") this.config.zoffset += sUD * 0.01;
        else if (mode == "oy") this.config.oy += sUD * 0.01;
        else if (mode == "scale") {
          this.config.scale -= sUD * 0.01;
          this.group.scale.set(this.config.scale, this.config.scale, this.config.scale);
        }
        else if (mode == "depthT") this.config.depthThreshold += sUD * 0.005;

        this.updateShaderUniforms();
        this.onConfigUpdate(this.config);
      }

      // Trigger-based adjustments
      if (sCont) {
        // Thumbstick press for OX/OY
        if (sButt[3].pressed) {
          if (this.lastOyPos === null) this.lastOyPos = new THREE.Vector3().copy(sCont.position);
          const diff = new THREE.Vector3().subVectors(sCont.position, this.lastOyPos);
          this.config.oy -= diff.y / this.config.scale;
          this.config.ox -= diff.x / this.config.scale;
          this.lastOyPos = sCont.position.clone();
          this.updateShaderUniforms();
        } else if (sButt[3].unclicked) {
          this.lastOyPos = null;
          this.onConfigUpdate(this.config);
        }

        // Index trigger for Z-offset
        if (sButt[0].pressed) {
          if (this.lastZOffsetPos === null) this.lastZOffsetPos = new THREE.Vector3().copy(sCont.position);
          const cam = this.renderer.xr.getCamera(this.camera);
          const dir = new THREE.Vector3();
          cam.getWorldDirection(dir);
          const diff = new THREE.Vector3().subVectors(sCont.position, this.lastZOffsetPos);
          this.config.zoffset -= diff.dot(dir) / this.config.scale;
          this.lastZOffsetPos = sCont.position.clone();
          this.updateShaderUniforms();
        } else if (sButt[0].unclicked) {
          this.lastZOffsetPos = null;
          this.onConfigUpdate(this.config);
        }

        // Middle trigger for Z-scale
        if (sButt[1].pressed) {
          if (this.lastZScalePos === null) this.lastZScalePos = new THREE.Vector3().copy(sCont.position);
          const cam = this.renderer.xr.getCamera(this.camera);
          const dir = new THREE.Vector3();
          cam.getWorldDirection(dir);
          const diff = new THREE.Vector3().subVectors(sCont.position, this.lastZScalePos);
          this.config.zscale -= diff.dot(dir) / this.config.scale;
          this.lastZScalePos = sCont.position.clone();
          this.updateShaderUniforms();
        } else if (sButt[1].unclicked) {
          this.lastZScalePos = null;
          this.onConfigUpdate(this.config);
        }
      }
    }

    if (this.mEnabled && this.group) {
      if (typeof mUD === "number" && mUD !== 0) {
        const cam = this.renderer.xr.getCamera(this.camera);
        const dir = new THREE.Vector3();
        cam.getWorldDirection(dir);
        dir.multiplyScalar(-mUD * 0.01);
        this.group.position.add(dir);
        this.group.updateMatrixWorld(true);
      }

      if (mButt[0].pressed) {
        this.zeroCamera();
        if (this.mesh) {
          this.mesh.position.set(0, 0, 0);
          this.mesh.quaternion.set(0, 0, 0, 1);
        }
        this.photoMeshes.forEach(pm => pm.lookAt(this.camera.position));
      }

      if (mCont) {
        if (mButt[1].pressed) {
          if (this.lastContPos === null) {
            this.lastContPos = new THREE.Vector3().copy(mCont.position);
            this.lastMeshPos = new THREE.Vector3().copy(this.group.position);
          }
          let diff = new THREE.Vector3().subVectors(mCont.position, this.lastContPos);
          this.group.position.addVectors(this.lastMeshPos, diff);
        } else {
          this.lastContPos = null;
          this.lastMeshPos = null;
        }

        if (mButt[3].clicked && this.mesh) {
          mCont.attach(this.mesh);
          mCont.attach(this.splat);
        } else if (mButt[3].unclicked && this.mesh) {
          this.group.attach(this.mesh);
          this.group.attach(this.splat);
        }
      }

      // Navigation and Reload logic
      if (mButt[4].processed === undefined) {
        if (mButt[4].pressed) {
          if (mButt[4].pressedTime === undefined) mButt[4].pressedTime = Date.now();
          if (Date.now() - mButt[4].pressedTime > 1000) {
            mButt[4].pressedTime = undefined;
            mButt[4].processed = true;
            this.onLoadMesh(this.config.basename);
          }
        }
      }
      if (mButt[4].unclicked) {
        if (mButt[4].processed === undefined) this.onNextImage(1);
        mButt[4].pressedTime = undefined;
        mButt[4].processed = undefined;
      }
      if (mButt[5].unclicked) this.onNextImage(-1);
    }
  }
}
