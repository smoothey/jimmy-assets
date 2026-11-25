# Jimmy's 3D assets

The source files for 3D Jimmy: Meshy's models, the Blender files and the
textures. They live here so msi, Balrog and Gandalf all work from the same
copy. The scripts that build him stay in
[smoothey/jimmy](https://github.com/smoothey/jimmy) under
`assistant/widget/art/tools/jimmy3d/`.

## Layout
The same as `Documents\Jimmy 3D` on msi, so the scripts' own paths work:
`v5/` holds `Jimmy5.blend` and Meshy's `jimmy-open.glb`, `v6/` and `v7/` hold
the later Meshy models. `jimmy5.py` finds them next to itself, so a copy of it
sits in `v5/` (kept in step with smoothey/jimmy, checked by md5).

## Big files
Blend, model, picture and archive files are stored with Git LFS
(`.gitattributes`). Renders and clips are not kept: they are made again from
the blend. Install Git LFS once per machine (`git lfs install`).

## Working on a machine
- Clone to `C:\dev\jimmy-assets`, `git pull` before starting.
- Take the machine's Blender lock first (Jimmy store, r2jimmy state
  `blender-<device>`, rule in `governance/loops/CONTRACT.md` in smoothey/jimmy).
- The blend can't be merged. If you change it, commit and push straight away,
  and pull on the other machines before they touch it.
- The blend grows about 16 MB each time it is built and saved, and every
  pushed copy stays in LFS storage (10 GiB included). Clean it out (File,
  Clean Up, Purge Unused Data) and save it compressed before pushing, and
  push it only when a step is agreed, not on every save.

## Big working copies
Copies too big or too frequent for the repo (a working blend between agreed
steps, old renders) go on Google Drive, `G:\My Drive\Dev\Jimmy 3D` (Ant,
25 Nov 2026). Drive is the backup; this repo is what the machines work from.
