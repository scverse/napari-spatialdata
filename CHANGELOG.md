# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.5.1] - 2024-xx-xx

## [0.5.0] - 2024-07-03

### Added

- New annotation widget for managing shapes annotations @melonora #233 #261
- Showing image channel names in var widget for easy channel selection #254

### Fixed

- Fixed scatterplot widget #247
- Fixed contrast limits for images #265

## [0.4.1] - 2024-03-30

### Fixed

- Saving shapes and points with `Shift + E` now saves 2D data (a z dim was wrongly added)

## [0.4.0] - 2024-03-24

### Added

- Multi table support #199 @melonora
- Color shapes and points by values in dataframe in addition to tables #216

### Fixed

- Avoid exception when registering shortcuts #201 @aeisenbarth
- Fix wrong point size when affine specified #193
- Wrong element visiblity when changing coordinate system #207
- Cleaned up data model #180
- Fixes correct matching of geometies and annoations via join #208
- Better docstrings #219

## [0.3.1] - 2023-11-11

### Fix

- removed npe (napari plugin engine) dependency

## [0.3.0] - 2023-11-09

### Added

- New APIs for changing coordinate systems #169
- New APIs for adding single elements #170
- Improved code to link new layers to layers with a `SpatialData` object #173

### Fixed

- Enabled saving annotations #168
- Added safeguard when table is None #177 @aeisenbarth
- Fix plotting annotations when changing element #175
- Fixed installation requirements #185 @goanpeca

## [0.2.8] - 2023-10-30

### Added

- Global parameters POINT_THRESHOLD and POLYGON_THRESHOLD added to allow users to adjust these parameters.

### Fixed

- Updated dependencies
- Updated installation instructions since napari conda-forge no longer installs Qt backend. @psobolewskiPhD
- Allow for viewing SpatialData object with no annotations. @aeisenbarth

## [0.2.7] - 2023-10-02

### Added

- Multiple SpatialData objects support (ported from the "spatialdata" branch)
- Interactive is more ergonomic, has a headless parameter, can be used to display pre-configured elements
- Remembering user layer visibility settings when changing coordinate system

### Fixed

- Fixes in CLI
- Several internal bugfixes and code refactorings
- Fixes in updating affine transformation when changing coordinate system

## [0.2.6] - 2023-07-13

### Fixed

- Wrong tag preventing the release to pip (the previous tag was not of a commit merged to main).

## [0.2.5] - 2023-07-13

### Fixed

- CLI @berombau @LucaMarconato
- Display of points and circles @LucaMarconato
- Issue with name reinitialization @tothmarcella
- Reverted to use PyQt5 @melonora
- RGB/RGBA correctly displayed for 3/4 channel images @rahulbshrestha
- Layer visibility changes when changing coordinate system @melonora
- Performance with multiscale images @LucaMarconato
- Refactored internal code to prepare for future refactoring @melonora

### Thanks

- Reviewers @giovp @timtreis @kevinyamauchi and the users that reported bugs.

## [0.2.4] - 2023-05-23

### Added

- Color circles and polygons with annotations (#73)

## [0.2.3] - 2023-05-11

### Fixed

- Shapes support

## [0.2.1] - 2023-05-04

### Fixed

- Package versioning (#63)

## [0.2.0] - 2023-05-04

### Merged

- Merge pull request #62 from scverse/kevinyamauchi-patch-1 - Install spatialdata from pypi
