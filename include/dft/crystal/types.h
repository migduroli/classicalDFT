#ifndef DFT_CRYSTAL_TYPES_H
#define DFT_CRYSTAL_TYPES_H

namespace dft::crystal {

  enum class Structure { BCC, FCC, HCP };

  /**
   * @brief Miller-index orientation of the crystal plane orthogonal to the z-axis.
   *
   * For BCC/FCC: _001, _010, _100, _110, _101, _011, _111.
   * For HCP: _001, _010, _100.
   */
  enum class Orientation { _001, _010, _100, _110, _101, _011, _111 };

  enum class ExportFormat { XYZ, CSV };

}  // namespace dft::crystal

#endif  // DFT_CRYSTAL_TYPES_H
