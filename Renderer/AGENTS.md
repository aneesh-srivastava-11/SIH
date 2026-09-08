# AGENTS.md

## 1. What Part B is, in one paragraph
Part B turns geometry into images that a matcher can train on — applying a reflectance law, draping albedo, running the result through a camera, and emitting labelled pairs. The deliverable is not "a pretty render." The deliverable is a directory of image pairs with exact pixel correspondences and a manifest. Success is measured by the render-vs-real evidence panel that proves the pairs are worth training on, not by photorealism.

## 2. Decisions
- **View vector**: Nadir, `v = [0,0,1]`, do not parse roll/pitch. Phase angle is fully determined by sun elevation: `α = 90° - e_sun`.
- **Albedo**: Real drape is the default, procedural is the fallback. Carry an explicit `albedo_valid` mask forward.

## 9. Non-goals — do not let the agent wander here
- **No Hapke.** 
- **No opposition surge.**
- **No Blender / Cycles / CORTO.**
- **No perspective camera.** Ortho only.
- **No IIRS spectral or thermal modelling.**
- **No edits under `geometry/`.**
- **No visual-polish work.**

## Other Rules
- **The interface in §4 is frozen.**
- **Every physical constant, range and magic number lives in `configs/render.yaml`**, never inline.
- **Assumption-tier comments are mandatory.** Any constant not traceable to a source gets `# A-TIER ASSUMPTION: <what, why, what breaks if wrong>`.
