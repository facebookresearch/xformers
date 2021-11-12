# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.0.4] - TBD
### Fixed
- Homogenizing the masks, additive or bool [#79][#85][#86]

### Added
- Fused dropout/bias/activation layer [#58]
- Fused layernorm used by default in the factory [#92]


## [0.0.3] - 2021-11-01
### Fixed
- Nystrom causal attention [#75]


## [0.0.2] - 2021-11-01
### Fixed
- More robust blocksparse [#24]

### Added
- Rotary embeddings [#32]
- More flexible layernorm [#50]
