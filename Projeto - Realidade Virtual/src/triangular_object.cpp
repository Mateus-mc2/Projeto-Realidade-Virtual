#include "triangular_object.h"

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Matrix3d;

namespace util {

TriangularObject::TriangularObject(const Material &material,
                                   const bool is_emissive,
                                   const std::vector<Eigen::Vector3d> &vertices,
                                   const std::vector<Eigen::Vector3i> &faces)
      :  RenderableObject(material, is_emissive),
         kVertices(vertices),
         kFaces(faces) {
  int num_faces = static_cast<int>(faces.size());
  this->planes_coeffs_.resize(num_faces);
  this->linear_systems_.resize(num_faces);
    
  for (int i = 0; i < num_faces; ++i) {
    // Get plane equation (coefficients) - we are assuming the .obj files provide us the 
    // correct orientation of the vertices.
    const Vector3d kA = vertices[faces[i](0)];
    const Vector3d kB = vertices[faces[i](1)];
    const Vector3d kC = vertices[faces[i](2)];

    const Vector3d kAB = kB - kA;
    const Vector3d kAC = kC - kA;
    const Vector3d kNormal = kAB.cross(kAC);

    assert(!(math::IsAlmostEqual(kNormal(0), 0.0, this->kEps) &&
             math::IsAlmostEqual(kNormal(1), 0.0, this->kEps) &&
             math::IsAlmostEqual(kNormal(2), 0.0, this->kEps)));

    // The last coefficient is the additive inverse of the dot product of kA and kNormal.
    this->planes_coeffs_[i] << kNormal(0), kNormal(1), kNormal(2), -kNormal.dot(kA);

    // Get system's inverse matrix.
    Matrix3d linear_system;
    linear_system << kA(0), kB(0), kC(0),
                     kA(1), kB(1), kC(1), 
                     kA(2), kB(2), kC(2);
    this->linear_systems_[i] = linear_system.inverse();
  }
}

bool TriangularObject::IsInnerPoint(const Vector3d &barycentric_coordinates) const {
  double alpha = barycentric_coordinates(0);
  double beta = barycentric_coordinates(1);
  double gamma = barycentric_coordinates(2);

  return (math::IsAlmostEqual(alpha + beta + gamma, 1.0, this->kEps) &&
          alpha >= 0 && beta >= 0 && gamma >= 0 &&
          alpha <= 1 && beta <= 1 && gamma <= 1);
}

double TriangularObject::GetIntersectionParameter(const Ray &ray, Vector3d *normal) const {
  Vector4d ray_origin(ray.origin(0), ray.origin(1), ray.origin(2), 1);

  // Parameters to return.
  double min_t = std::numeric_limits<double>::max();
  Vector3d parameters;

  // Get nearest intersection point - need to check every single face of the object.
  for (int i = 0; i < this->planes_coeffs_.size(); ++i) {
    const Vector3d current_normal(this->planes_coeffs_[i](0),
                                  this->planes_coeffs_[i](1),
                                  this->planes_coeffs_[i](2));
    const double numerator = -(this->planes_coeffs_[i].dot(ray_origin));    
    const double denominator = current_normal.dot(ray.direction);

    // Test if the ray and this plane are parallel (or if this plane contains the ray).
    // Returns a negative (dummy) parameter t if this happens.
    if (math::IsAlmostEqual(denominator, 0.0, this->kEps)) {
      return -1.0;
    }

    double curr_t = numerator / denominator;
    Vector3d intersection_point = ray.origin + curr_t*ray.direction;
    Vector3d barycentric_coords = this->linear_systems_[i]*intersection_point;  // x = A^(-1)*b.

    if (this->IsInnerPoint(barycentric_coords) && min_t > curr_t && curr_t > this->kEps) {
      min_t = curr_t;
      parameters = barycentric_coords;

      (*normal) = current_normal / current_normal.norm();
    }
  }

  return min_t;
}

}  // namespace util