
# AdaGATE

**Adaptive Data Quality Validation for Dynamic Feature-Column Evolution**

AdaGATE is a data quality validation framework designed for machine learning pipelines with evolving feature columns.

The framework combines column-change ratio screening and clustering stability assessment to support adaptive validation decisions under feature-column evolution. It distinguishes between partitions that can proceed to automatic validation and those requiring further review or rejection.

## Source Code

This repository provides the core implementation of AdaGATE.

## Acknowledgments

AdaGATE is developed based on the GATE framework by dm4ml. The original GATE implementation is released under the MIT License.

The AdaGATE extensions introduce dynamic feature-column handling, clustering stability screening, and adaptive structural validation decisions.

## License

This repository is distributed under the MIT License. The original GATE copyright and license notices are retained.

See the [LICENSE](LICENSE) file for details.
