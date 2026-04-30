// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DynamicBuffer.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {

void puncture_field(
    gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& centered_coords,
    const tnsr::I<double, 3>& particle_position,
    const tnsr::I<double, 3>& particle_velocity,
    const tnsr::I<double, 3>& particle_acceleration, const double bh_mass,
    const std::array<double, 3>& spin, const size_t order) {
  if (order == 0) {
    puncture_field_kerr_0(result, centered_coords, particle_position,
                          particle_velocity, particle_acceleration, bh_mass,
                          spin);
  } else if (order == 1) {
    puncture_field_1(result, centered_coords, particle_position,
                     particle_velocity, particle_acceleration, bh_mass);
  } else {
    ERROR(
        "The puncture field is only implemented up to expansion order 1 but "
        "you requested order "
        << order);
  }
}

std::tuple<Scalar<DataVector>, Scalar<DataVector>,
           tnsr::i<DataVector, 3, Frame::Inertial>>
puncture_field(const tnsr::I<DataVector, 3, Frame::Inertial>& centered_coords,
               const tnsr::I<double, 3>& particle_position,
               const tnsr::I<double, 3>& particle_velocity,
               const tnsr::I<double, 3>& particle_acceleration,
               const double bh_mass, const std::array<double, 3>& spin,
               const size_t order) {
  using puncture_vars = Variables<
      tmpl::list<Tags::Psi, ::Tags::dt<Tags::Psi>,
                 ::Tags::deriv<Tags::Psi, tmpl::size_t<3>, Frame::Inertial>>>;
  puncture_vars result{1};
  puncture_field(make_not_null(&result), centered_coords, particle_position,
                 particle_velocity, particle_acceleration, bh_mass, spin,
                 order);

  std::tuple<Scalar<DataVector>, Scalar<DataVector>,
             tnsr::i<DataVector, 3, Frame::Inertial>>
      return_tuple{
          get<Tags::Psi>(result).get(),
          get<::Tags::dt<Tags::Psi>>(result).get(),
          get<::Tags::deriv<Tags::Psi, tmpl::size_t<3>, Frame::Inertial>>(
              result)};

  return return_tuple;
}
}  // namespace CurvedScalarWave::Worldtube
